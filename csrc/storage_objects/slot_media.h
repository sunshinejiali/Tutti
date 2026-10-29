#pragma once

// csrc/storage_objects/slot_media.h -- physical slot operations.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI.
//
// Everything here uses O_DIRECT, per project policy for data files. On a
// GPU-direct path buffered IO is actively harmful, not merely slower:
//   * page cache pollution competes for memory with the KV pool;
//   * writeback competes with GPU DMA for device bandwidth;
//   * after a GPU DMA write the page cache holds STALE data, so a later
//     buffered read can return the old contents -- a correctness bug, not a
//     performance one.
// O_DIRECT therefore requires 4096-alignment of buffer, offset and length,
// which is why every structure in this layer is sized in whole 4096-byte units
// and why buffers come from posix_memalign rather than std::vector.
//
// This layer performs HOST-side metadata IO: object headers, precreating
// space, re-zeroing on reclaim. Normal payload IO belongs to the GPU-direct
// DataPath. Repairing an unpublished segment is the exception: it copies the
// file's existing bytes to preserve data before the file is exposed to DMA.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tutti/status.h>

namespace tutti::storage_objects {

// -------------------------------------------------------------------------
// AlignedBuffer -- posix_memalign-backed, for O_DIRECT.
//
// std::vector cannot be used: its data has no alignment guarantee beyond the
// element type, and an unaligned O_DIRECT buffer fails with EINVAL.
// -------------------------------------------------------------------------
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t bytes);
    ~AlignedBuffer();

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    std::uint8_t* data() noexcept { return data_; }
    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool valid() const noexcept { return data_ != nullptr; }

    void zero() noexcept;

private:
    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

// -------------------------------------------------------------------------
// Segment files
//
// materialise_segment_file creates `path` at exactly `file_bytes` and writes
// REAL ZEROS over it, fsyncs, verifies FIEMAP (no holes/unwritten extents, at
// most 124 extents -- the resolver's limit) and then publishes `<path>.ready`
// (file_bytes, st_dev, st_ino). Preallocation only reserves space; real writes
// must initialise it, because the resolver rejects UNWRITTEN extents.
// Creation is serialised per mount (flock on <mount>/.tutti_segment_precreate.lock):
// concurrent large allocations on ext4 fragment far beyond the extent limit.
// Idempotent: a ready file is left alone.
// A file missing its ready marker is verified in place first. Invalid layouts
// are copied byte-for-byte into a preallocated temporary file and atomically
// replaced only after verification, preserving data if publication crashed.
Status materialise_segment_file(const std::string& path, std::uint64_t file_bytes);

// Probe only (never creates): *out = true iff `path` is `file_bytes` long and
// its ready marker matches. A missing file is (*out = false), not an error.
Status segment_file_is_ready(const std::string& path, std::uint64_t file_bytes,
                             bool* out);

// Rewrite zeros over [offset, offset + bytes) of an existing file, for
// reclamation. Zeroing the header is what invalidates the object: a zero magic
// decodes as "never written" rather than as corruption.
Status zero_file_range(const std::string& path,
                       std::uint64_t offset,
                       std::uint64_t bytes);

// -------------------------------------------------------------------------
// Header IO
//
// The header is exactly one 4096-byte block, so both directions are a single
// aligned operation.
//
// write_object_header fsyncs before returning. This is THE commit point: the
// caller must have already made the payload durable, because the ordering
// "payload fsync, then header fsync" is what makes a valid header mean the
// payload was durable. Reversing it would permit a header that describes data
// which never landed -- the one failure this layer must not have.
Status write_object_header(const std::string& path, std::uint64_t offset,
                           const std::uint8_t* header, std::size_t header_bytes);

// Reads one header block. A short read at EOF is reported as NOT_FOUND rather
// than an error: a file shorter than its header simply has no object.
Status read_object_header(const std::string& path, std::uint64_t offset,
                          std::uint8_t* out, std::size_t out_bytes);

// -------------------------------------------------------------------------
// Checkpoint and bitmap IO
//
// Checkpoint containers are whole 4096-byte multiples by construction, so they
// are written with O_DIRECT like everything else. The residency bitmap is
// deliberately NOT here: it is mmapped, where O_DIRECT is meaningless.
// -------------------------------------------------------------------------
Status write_checkpoint_container(const std::string& path, std::uint64_t offset,
                                  const std::uint8_t* image,
                                  std::size_t image_bytes);

Status read_checkpoint_container(const std::string& path, std::uint64_t offset,
                                 std::uint8_t* out, std::size_t out_bytes);

// Create the file if absent and ensure it is at least `bytes`, precreating
// with real zeros. Used for the checkpoint region.
Status ensure_metadata_file(const std::string& path, std::uint64_t bytes);

// Create a directory and every missing parent. Returns OK if it already exists.
Status ensure_directory(const std::string& path);

// fsync a directory, so a file creation or rename inside it is durable. A file's
// own fsync does not make its directory entry durable.
Status sync_directory(const std::string& path);

} // namespace tutti::storage_objects
