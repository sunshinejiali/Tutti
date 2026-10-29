// csrc/storage_objects/slot_media.cpp

#include "csrc/storage_objects/slot_media.h"

#include <fcntl.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <utility>

namespace tutti::storage_objects {
namespace {

constexpr std::size_t kAlignment = 4096;

// Chunk size for materialisation. Large enough that per-write syscall overhead
// is irrelevant, small enough that the pinned aligned buffer stays modest.
constexpr std::size_t kZeroChunkBytes = 4u * 1024 * 1024;

Status errno_status(const char* what, const std::string& path) {
    std::string message = what;
    message += " (";
    message += path;
    message += "): ";
    message += std::strerror(errno);
    return Status(StatusCode::INTERNAL, message);
}

bool is_aligned(std::uint64_t value) noexcept {
    return value % kAlignment == 0;
}

// RAII descriptor. Deliberately minimal: an open fd escaping on an error path
// would leak a descriptor per failed slot, which at pool scale exhausts the
// process limit and turns a transient error into a hard stop.
class Fd {
public:
    explicit Fd(int fd) : fd_(fd) {}
    ~Fd() { if (fd_ >= 0) ::close(fd_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const noexcept { return fd_; }
    bool valid() const noexcept { return fd_ >= 0; }
private:
    int fd_;
};

// Write `bytes` of zeros at `offset` using O_DIRECT-aligned chunks.
Status write_zeros(int fd, const std::string& path, std::uint64_t offset,
                   std::uint64_t bytes) {
    if (bytes == 0) return {};

    AlignedBuffer buffer(kZeroChunkBytes);
    if (!buffer.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    buffer.zero();

    std::uint64_t written = 0;
    while (written < bytes) {
        const std::uint64_t remaining = bytes - written;
        const std::size_t chunk = remaining < kZeroChunkBytes
                                      ? static_cast<std::size_t>(remaining)
                                      : kZeroChunkBytes;
        // O_DIRECT requires the length to be aligned too. All slot sizes are
        // whole 4096 multiples by construction, so a short tail can only mean a
        // caller passed an unaligned size -- reject rather than silently fall
        // back to buffered IO.
        if (!is_aligned(chunk)) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "zero length must be 4096-aligned for O_DIRECT");
        }
        const ssize_t n = ::pwrite(fd, buffer.data(), chunk,
                                   static_cast<off_t>(offset + written));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pwrite zeros", path);
        }
        if (n == 0) {
            return Status(StatusCode::INTERNAL, "pwrite returned 0: " + path);
        }
        written += static_cast<std::uint64_t>(n);
    }
    return {};
}

std::string parent_of(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

Status filesystem_lock_root(const std::string& path, std::string* root) {
    std::string dir = parent_of(path);
    struct stat current{};
    if (::stat(dir.c_str(), &current) != 0)
        return errno_status("stat segment directory", dir);
    while (dir != "/") {
        const std::string parent = parent_of(dir);
        struct stat above{};
        if (::stat(parent.c_str(), &above) != 0)
            return errno_status("stat segment parent", parent);
        if (above.st_dev != current.st_dev) break;
        dir = parent;
        current = above;
    }
    *root = dir;
    return {};
}

struct SegmentReadyRecord {
    std::uint64_t file_bytes;
    std::uint64_t device;
    std::uint64_t inode;
};

// The ready record of this very inode, if one was published.
bool read_ready_record(const std::string& path, const struct stat& data_stat,
                       SegmentReadyRecord* out) {
    const std::string marker = path + ".ready";
    const Fd fd(::open(marker.c_str(), O_RDONLY | O_CLOEXEC));
    if (!fd.valid()) return false;
    return ::read(fd.get(), out, sizeof(*out)) ==
               static_cast<ssize_t>(sizeof(*out)) &&
           out->device == static_cast<std::uint64_t>(data_stat.st_dev) &&
           out->inode == static_cast<std::uint64_t>(data_stat.st_ino);
}

bool segment_ready(const std::string& path, const struct stat& data_stat,
                   std::uint64_t file_bytes) {
    SegmentReadyRecord recorded{};
    return read_ready_record(path, data_stat, &recorded) &&
           recorded.file_bytes == file_bytes;
}

Status publish_segment_ready(const std::string& path, const struct stat& data_stat,
                             std::uint64_t file_bytes) {
    const std::string pending = path + ".ready.pending";
    const std::string marker = path + ".ready";
    const Fd fd(::open(pending.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                       0644));
    if (!fd.valid()) return errno_status("open segment ready marker", pending);
    const SegmentReadyRecord recorded{
        file_bytes, static_cast<std::uint64_t>(data_stat.st_dev),
        static_cast<std::uint64_t>(data_stat.st_ino)};
    if (::write(fd.get(), &recorded, sizeof(recorded)) !=
        static_cast<ssize_t>(sizeof(recorded))) {
        return errno_status("write segment ready marker", pending);
    }
    if (::fsync(fd.get()) != 0) return errno_status("fsync segment ready marker", pending);
    if (::rename(pending.c_str(), marker.c_str()) != 0) {
        return errno_status("publish segment ready marker", marker);
    }
    return sync_directory(parent_of(path));
}

Status verify_segment_extents(int fd, const std::string& path,
                              std::uint64_t file_bytes) {
    constexpr std::size_t kBatch = 256;
    constexpr std::size_t kMaxExtents = 124;
    std::vector<std::uint8_t> bytes(sizeof(fiemap) + kBatch * sizeof(fiemap_extent));
    auto* map = reinterpret_cast<fiemap*>(bytes.data());
    std::uint64_t cursor = 0;
    std::uint64_t physical_end = 0;
    std::size_t extents = 0;
    while (cursor < file_bytes) {
        std::memset(bytes.data(), 0, bytes.size());
        map->fm_start = cursor;
        map->fm_length = file_bytes - cursor;
        map->fm_flags = FIEMAP_FLAG_SYNC;
        map->fm_extent_count = kBatch;
        if (::ioctl(fd, FS_IOC_FIEMAP, map) != 0)
            return errno_status("fiemap segment file", path);
        if (map->fm_mapped_extents == 0)
            return Status(StatusCode::DATA_LOSS, "segment file has a hole: " + path);
        for (std::uint32_t i = 0; i < map->fm_mapped_extents; ++i) {
            const fiemap_extent& e = map->fm_extents[i];
            constexpr std::uint32_t bad = FIEMAP_EXTENT_UNKNOWN |
                FIEMAP_EXTENT_DELALLOC | FIEMAP_EXTENT_UNWRITTEN |
                FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_SHARED |
                FIEMAP_EXTENT_NOT_ALIGNED | FIEMAP_EXTENT_DATA_ENCRYPTED |
                FIEMAP_EXTENT_DATA_INLINE | FIEMAP_EXTENT_DATA_TAIL;
            if ((e.fe_flags & bad) || e.fe_logical != cursor ||
                e.fe_length == 0 || e.fe_length > file_bytes - cursor ||
                !is_aligned(e.fe_physical) || !is_aligned(e.fe_length) ||
                e.fe_physical > UINT64_MAX - e.fe_length)
                return Status(StatusCode::DATA_LOSS,
                              "segment file has unsafe or missing extents: " + path);
            if (extents == 0 || e.fe_physical != physical_end) ++extents;
            if (extents > kMaxExtents)
                return Status(StatusCode::RESOURCE_EXHAUSTED,
                              "segment file exceeds resolver extent limit: " + path);
            cursor += e.fe_length;
            physical_end = e.fe_physical + e.fe_length;
        }
    }
    return {};
}

Status preallocate_segment(int fd, const std::string& path,
                           std::uint64_t offset, std::uint64_t bytes) {
    // Reserve before writing so ext4 can place the complete range together.
    // These extents are still UNWRITTEN: every byte must subsequently be
    // written and FIEMAP-verified before publishing a ready marker.
    const int error = ::posix_fallocate(fd, static_cast<off_t>(offset),
                                       static_cast<off_t>(bytes));
    if (error == 0 || error == EOPNOTSUPP || error == ENOSYS) return {};
    return Status(error == ENOSPC ? StatusCode::RESOURCE_EXHAUSTED : StatusCode::INTERNAL,
                  "preallocate segment file (" + path + "): " + std::strerror(error));
}

Status rebuild_unpublished_segment(int source, const std::string& path,
                                   const struct stat& source_stat,
                                   std::uint64_t file_bytes) {
    // Missing ready metadata alone does not prove that the file is empty.
    // Preserve all bytes, and keep the old inode until the replacement passes
    // the same FIEMAP checks as a normal materialisation. The mount lock held
    // by the caller serialises cooperative publishers throughout the repair.
    struct statvfs fs{};
    if (::fstatvfs(source, &fs) != 0) return errno_status("statvfs segment rebuild", path);
    if (static_cast<std::uint64_t>(fs.f_bavail) * fs.f_frsize < file_bytes)
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "insufficient free space to rebuild segment file: " + path);
    std::string pattern = path + ".rebuild.XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const Fd replacement(::mkostemp(name.data(), O_DIRECT | O_CLOEXEC));
    if (!replacement.valid()) return errno_status("create segment rebuild", path);
    struct RemovePending {
        const char* path;
        ~RemovePending() { ::unlink(path); }
    } pending{name.data()};
    if (::fchmod(replacement.get(), source_stat.st_mode & 0777) != 0)
        return errno_status("chmod segment rebuild", name.data());
    const Status allocated = preallocate_segment(replacement.get(), name.data(), 0, file_bytes);
    if (!allocated.ok()) return allocated;
    AlignedBuffer buffer(kZeroChunkBytes);
    if (!buffer.valid()) return Status(StatusCode::INTERNAL, "could not allocate segment copy buffer");
    for (std::uint64_t offset = 0; offset < file_bytes;) {
        const std::size_t bytes = static_cast<std::size_t>(
            std::min<std::uint64_t>(kZeroChunkBytes, file_bytes - offset));
        ssize_t read;
        do { read = ::pread(source, buffer.data(), bytes, static_cast<off_t>(offset)); }
        while (read < 0 && errno == EINTR);
        if (read < 0) return errno_status("read segment rebuild", path);
        if (read != static_cast<ssize_t>(bytes))
            return Status(StatusCode::DATA_LOSS, "short read during segment rebuild: " + path);
        ssize_t written;
        do { written = ::pwrite(replacement.get(), buffer.data(), bytes,
                               static_cast<off_t>(offset)); }
        while (written < 0 && errno == EINTR);
        if (written < 0) return errno_status("write segment rebuild", name.data());
        if (written != static_cast<ssize_t>(bytes))
            return Status(StatusCode::DATA_LOSS, "short write during segment rebuild: " + path);
        offset += bytes;
    }
    if (::fsync(replacement.get()) != 0) return errno_status("fsync segment rebuild", name.data());
    const Status verified = verify_segment_extents(replacement.get(), name.data(), file_bytes);
    if (!verified.ok()) return verified;
    struct stat rebuilt{};
    if (::fstat(replacement.get(), &rebuilt) != 0)
        return errno_status("fstat segment rebuild", name.data());
    if (::rename(name.data(), path.c_str()) != 0)
        return errno_status("replace unpublished segment", path);
    return publish_segment_ready(path, rebuilt, file_bytes);
}

} // namespace

// -------------------------------------------------------------------------
// AlignedBuffer
// -------------------------------------------------------------------------

AlignedBuffer::AlignedBuffer(std::size_t bytes) {
    if (bytes == 0) return;
    // Round up so the allocation itself is a whole number of blocks; O_DIRECT
    // cares about the address and the length actually used, and rounding here
    // means callers never have to.
    const std::size_t rounded = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
    void* raw = nullptr;
    if (::posix_memalign(&raw, kAlignment, rounded) != 0) return;
    data_ = static_cast<std::uint8_t*>(raw);
    size_ = rounded;
}

AlignedBuffer::~AlignedBuffer() {
    if (data_ != nullptr) std::free(data_);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) std::free(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void AlignedBuffer::zero() noexcept {
    if (data_ != nullptr) std::memset(data_, 0, size_);
}

// -------------------------------------------------------------------------
// Directory helpers
// -------------------------------------------------------------------------

Status ensure_directory(const std::string& path) {
    if (path.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT, "empty directory path");
    }
    // Create parents first. Walking forward and ignoring EEXIST is both simpler
    // and race-tolerant: several ranks bring up the same namespace root
    // concurrently, so "already exists" is the normal case, not an error.
    std::string partial;
    partial.reserve(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        partial.push_back(path[i]);
        const bool last = (i + 1 == path.size());
        if (path[i] != '/' && !last) continue;
        if (partial == "/" || partial.empty()) continue;
        std::string dir = partial;
        if (dir.size() > 1 && dir.back() == '/') dir.pop_back();
        if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            return errno_status("mkdir", dir);
        }
    }
    return {};
}

Status sync_directory(const std::string& path) {
    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECTORY));
    if (!fd.valid()) return errno_status("open directory", path);
    if (::fsync(fd.get()) != 0) return errno_status("fsync directory", path);
    return {};
}

// -------------------------------------------------------------------------
// Materialisation
// -------------------------------------------------------------------------

Status materialise_segment_file(const std::string& path, std::uint64_t file_bytes) {
    if (file_bytes == 0 || !is_aligned(file_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment file size must be nonzero and 4096-aligned");
    }
    const std::string parent = parent_of(path);
    const Status dir = ensure_directory(parent);
    if (!dir.ok()) return dir;
    // Coordinate all ranks under the same mount, not only workers sharing one
    // file. Concurrent 20 GiB allocations on ext4 fragmented into >2000
    // extents on this deployment, far beyond the resolver's 124-extent cap.
    std::string mount;
    const Status located = filesystem_lock_root(path, &mount);
    if (!located.ok()) return located;
    const std::string lock_path = mount + "/.tutti_segment_precreate.lock";
    const Fd disk_lock(::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644));
    if (!disk_lock.valid()) return errno_status("open segment disk lock", lock_path);
    if (::flock(disk_lock.get(), LOCK_EX) != 0)
        return errno_status("lock segment disk", lock_path);
    const Fd fd(::open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644));
    if (!fd.valid()) return errno_status("open segment file", path);
    // A second precreate worker must not observe a partially zero-filled file
    // as complete merely because ftruncate already set its final size.
    if (::flock(fd.get(), LOCK_EX) != 0) return errno_status("lock segment file", path);
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) return errno_status("fstat segment file", path);
    const std::uint64_t old_size = static_cast<std::uint64_t>(st.st_size);
    if (old_size > file_bytes) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment file exceeds configured size: " + path);
    }
    SegmentReadyRecord recorded{};
    if (read_ready_record(path, st, &recorded)) {
        if (recorded.file_bytes == file_bytes && old_size == file_bytes) return {};
        // A published file of another size belongs to a different geometry.
        // Extending it would silently corrupt that pool; refuse instead.
        if (recorded.file_bytes != file_bytes) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          "segment file was built for another geometry: " + path);
        }
    }
    if (old_size < file_bytes) {
        struct statvfs fs{};
        if (::fstatvfs(fd.get(), &fs) != 0) {
            return errno_status("statvfs segment file", path);
        }
        const std::uint64_t free_bytes =
            static_cast<std::uint64_t>(fs.f_bavail) * fs.f_frsize;
        const std::uint64_t remaining = file_bytes - old_size;
        // A shared file costs its full size at the first slot, so the hard
        // check must cover the whole extension rather than one slot. Avoid a
        // mount-capacity-derived reserve here: deployments may intentionally
        // use a mostly-full volume, and the caller already controls the
        // configured capacity. The kernel's available-block check remains the
        // non-negotiable guard against overshooting the filesystem.
        if (free_bytes < remaining) {
            return Status(StatusCode::RESOURCE_EXHAUSTED,
                          "insufficient free space to materialise segment file: " + path);
        }
        const Status allocated = preallocate_segment(fd.get(), path, old_size, remaining);
        if (!allocated.ok()) return allocated;
        // FIEMAP-backed DMA rejects unwritten/hole extents. Materialise the
        // newly appended range, not just the requested slot, so one shared
        // file is valid for every slot as soon as it is exposed to resolver.
        const Status extended = write_zeros(fd.get(), path, old_size,
                                            file_bytes - old_size);
        if (!extended.ok()) return extended;
    }
    if (::fsync(fd.get()) != 0) return errno_status("fsync segment file", path);
    const Status verified = verify_segment_extents(fd.get(), path, file_bytes);
    if (!verified.ok()) {
        if (verified.code() == StatusCode::RESOURCE_EXHAUSTED ||
            verified.code() == StatusCode::DATA_LOSS)
            return rebuild_unpublished_segment(fd.get(), path, st, file_bytes);
        return verified;
    }
    return publish_segment_ready(path, st, file_bytes);
}

Status segment_file_is_ready(const std::string& path, std::uint64_t file_bytes,
                             bool* out) {
    if (out == nullptr) return Status(StatusCode::INVALID_ARGUMENT, "out must not be null");
    *out = false;
    if (file_bytes == 0 || !is_aligned(file_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "segment file size must be nonzero and 4096-aligned");
    }
    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) {
        if (errno == ENOENT) return {};
        return errno_status("open segment file for probe", path);
    }
    if (::flock(fd.get(), LOCK_SH) != 0) return errno_status("lock segment file probe", path);
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) return errno_status("fstat segment file", path);
    if (static_cast<std::uint64_t>(st.st_size) != file_bytes) return {};
    *out = segment_ready(path, st, file_bytes);
    return {};
}

Status zero_file_range(const std::string& path,
                       std::uint64_t offset,
                       std::uint64_t bytes) {
    if (!is_aligned(offset) || !is_aligned(bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "unaligned zero file range");
    }
    const Fd fd(::open(path.c_str(), O_RDWR | O_DIRECT));
    if (!fd.valid()) return errno_status("open segment file for zero", path);
    const Status written = write_zeros(fd.get(), path, offset, bytes);
    if (!written.ok()) return written;
    if (::fsync(fd.get()) != 0) return errno_status("fsync zero segment file", path);
    return {};
}

// -------------------------------------------------------------------------
// Header IO
// -------------------------------------------------------------------------

Status write_object_header(const std::string& path, std::uint64_t offset,
                           const std::uint8_t* header,
                           std::size_t header_bytes) {
    if (header == nullptr || header_bytes == 0 || !is_aligned(header_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header offset must be 4096-aligned");
    }

    // The caller's buffer may not be aligned, so stage through one that is.
    AlignedBuffer staged(header_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    std::memcpy(staged.data(), header, header_bytes);

    const Fd fd(::open(path.c_str(), O_RDWR | O_DIRECT));
    if (!fd.valid()) return errno_status("open slot for header write", path);

    std::size_t written = 0;
    while (written < header_bytes) {
        const ssize_t n = ::pwrite(fd.get(), staged.data() + written,
                                   header_bytes - written,
                                   static_cast<off_t>(offset + written));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pwrite header", path);
        }
        if (n == 0) {
            return Status(StatusCode::INTERNAL, "pwrite header returned 0: " + path);
        }
        written += static_cast<std::size_t>(n);
    }

    // THE commit point. The payload must already be durable: "payload fsync then
    // header fsync" is precisely what makes a valid header imply durable data.
    if (::fsync(fd.get()) != 0) return errno_status("fsync header", path);
    return {};
}

Status read_object_header(const std::string& path, std::uint64_t offset,
                          std::uint8_t* out, std::size_t out_bytes) {
    if (out == nullptr || out_bytes == 0 || !is_aligned(out_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header buffer must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "header offset must be 4096-aligned");
    }

    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) {
        if (errno == ENOENT) {
            return Status(StatusCode::NOT_FOUND, "slot file absent: " + path);
        }
        return errno_status("open slot for header read", path);
    }

    AlignedBuffer staged(out_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    staged.zero();

    std::size_t read_total = 0;
    while (read_total < out_bytes) {
        const ssize_t n = ::pread(fd.get(), staged.data() + read_total,
                                  out_bytes - read_total,
                                  static_cast<off_t>(offset + read_total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pread header", path);
        }
        if (n == 0) {
            // Short at EOF: a file smaller than its header simply holds no
            // object. That is an absence, not a failure.
            if (read_total == 0) {
                return Status(StatusCode::NOT_FOUND,
                              "slot shorter than header: " + path);
            }
            break;
        }
        read_total += static_cast<std::size_t>(n);
    }

    std::memcpy(out, staged.data(), out_bytes);
    return {};
}

// -------------------------------------------------------------------------
// Checkpoint IO
// -------------------------------------------------------------------------

Status write_checkpoint_container(const std::string& path, std::uint64_t offset,
                                  const std::uint8_t* image,
                                  std::size_t image_bytes) {
    if (image == nullptr || image_bytes == 0 || !is_aligned(image_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint image must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint offset must be 4096-aligned");
    }

    AlignedBuffer staged(image_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    std::memcpy(staged.data(), image, image_bytes);

    const Fd fd(::open(path.c_str(), O_RDWR | O_DIRECT));
    if (!fd.valid()) return errno_status("open checkpoint", path);

    std::size_t written = 0;
    while (written < image_bytes) {
        const ssize_t n = ::pwrite(fd.get(), staged.data() + written,
                                   image_bytes - written,
                                   static_cast<off_t>(offset + written));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pwrite checkpoint", path);
        }
        if (n == 0) {
            return Status(StatusCode::INTERNAL, "pwrite checkpoint returned 0");
        }
        written += static_cast<std::size_t>(n);
    }
    if (::fsync(fd.get()) != 0) return errno_status("fsync checkpoint", path);
    return {};
}

Status read_checkpoint_container(const std::string& path, std::uint64_t offset,
                                 std::uint8_t* out, std::size_t out_bytes) {
    if (out == nullptr || out_bytes == 0 || !is_aligned(out_bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint buffer must be nonzero and 4096-aligned");
    }
    if (!is_aligned(offset)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "checkpoint offset must be 4096-aligned");
    }

    const Fd fd(::open(path.c_str(), O_RDONLY | O_DIRECT));
    if (!fd.valid()) {
        if (errno == ENOENT) {
            return Status(StatusCode::NOT_FOUND, "checkpoint absent: " + path);
        }
        return errno_status("open checkpoint for read", path);
    }

    AlignedBuffer staged(out_bytes);
    if (!staged.valid()) {
        return Status(StatusCode::INTERNAL, "could not allocate aligned buffer");
    }
    staged.zero();

    std::size_t read_total = 0;
    while (read_total < out_bytes) {
        const ssize_t n = ::pread(fd.get(), staged.data() + read_total,
                                  out_bytes - read_total,
                                  static_cast<off_t>(offset + read_total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return errno_status("pread checkpoint", path);
        }
        if (n == 0) break;  // short at EOF: remaining bytes stay zero
        read_total += static_cast<std::size_t>(n);
    }

    std::memcpy(out, staged.data(), out_bytes);
    return {};
}

Status ensure_metadata_file(const std::string& path, std::uint64_t bytes) {
    if (bytes == 0 || !is_aligned(bytes)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "metadata file size must be nonzero and 4096-aligned");
    }
    const std::string dir = parent_of(path);
    const Status dir_status = ensure_directory(dir);
    if (!dir_status.ok()) return dir_status;

    const Fd fd(::open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644));
    if (!fd.valid()) return errno_status("open metadata file", path);

    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) return errno_status("fstat metadata", path);
    if (static_cast<std::uint64_t>(st.st_size) >= bytes) return {};

    if (::ftruncate(fd.get(), static_cast<off_t>(bytes)) != 0) {
        return errno_status("ftruncate metadata", path);
    }
    const Status zeroed = write_zeros(fd.get(), path, 0, bytes);
    if (!zeroed.ok()) return zeroed;
    if (::fsync(fd.get()) != 0) return errno_status("fsync metadata", path);
    return sync_directory(dir);
}

} // namespace tutti::storage_objects
