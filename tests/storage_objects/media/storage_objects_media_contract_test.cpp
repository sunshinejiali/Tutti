// tests/storage_objects/media/storage_objects_media_contract_test.cpp
//
// Contract test for the physical media layer (placement is covered by the
// segment-files test).
//
// Needs a real filesystem that supports O_DIRECT. That is not incidental: the
// whole layer exists to keep KV data out of the page cache, so a test on a
// filesystem without O_DIRECT would be testing a configuration this project
// does not support. When O_DIRECT is unavailable the media tests report SKIP
// loudly rather than passing silently -- a green run that never exercised the
// real path would be worse than no run.
//
// Scratch location: TMPDIR, else the build tree, else /tmp. A Tutti host
// typically has a small root filesystem and large data mounts, so hardcoding
// /tmp can fill the root.

#include "csrc/storage_objects/checkpoint_region.h"
#include "csrc/storage_objects/object_header_codec.h"
#include "csrc/storage_objects/slot_media.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace tutti;
using namespace tutti::storage_objects;

int g_failures = 0;
int g_skipped = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

#define REQUIRE(cond)                          \
    do {                                       \
        if (!(cond)) {                         \
            check(false, #cond, __LINE__);     \
            return;                            \
        }                                      \
    } while (0)

#ifndef TUTTI_TEST_TMPDIR_DEFAULT
#  define TUTTI_TEST_TMPDIR_DEFAULT "/tmp"
#endif

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = (base != nullptr && *base != '\0')
                           ? base
                           : TUTTI_TEST_TMPDIR_DEFAULT;
    tmpl += "/tutti_media_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    return dir == nullptr ? std::string() : std::string(dir);
}

// Whether this filesystem accepts O_DIRECT at all. tmpfs does not, and neither
// do some overlay configurations.
bool o_direct_supported(const std::string& dir) {
    const std::string probe = dir + "/.odirect_probe";
    const int fd = ::open(probe.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    const bool ok = (fd >= 0);
    if (fd >= 0) ::close(fd);
    ::unlink(probe.c_str());
    return ok;
}

constexpr std::uint64_t kSegmentBytes = 131072;   // 128 KiB, as deployed
constexpr std::uint32_t kSegmentCount = 80;       // 80 layers
constexpr std::uint64_t kPayload = kSegmentBytes * kSegmentCount;  // 10 MiB

ObjectKey key_of(std::uint8_t tag, std::size_t len = 18) {
    ObjectKey k;
    k.bytes.assign(len, tag);
    return k;
}

// ======================================================================
// 1. AlignedBuffer
// ======================================================================
void test_aligned_buffer() {
    AlignedBuffer buf(4096);
    REQUIRE(buf.valid());
    // O_DIRECT rejects an unaligned buffer address with EINVAL, which is why
    // std::vector cannot be used here.
    CHECK(reinterpret_cast<std::uintptr_t>(buf.data()) % 4096 == 0);
    CHECK(buf.size() == 4096);

    buf.data()[0] = 0xAB;
    buf.zero();
    CHECK(buf.data()[0] == 0);
    CHECK(buf.data()[4095] == 0);

    // A non-multiple request rounds up, so callers never have to.
    AlignedBuffer odd(100);
    REQUIRE(odd.valid());
    CHECK(odd.size() == 4096);
    CHECK(reinterpret_cast<std::uintptr_t>(odd.data()) % 4096 == 0);

    AlignedBuffer big(4096 * 3 + 1);
    REQUIRE(big.valid());
    CHECK(big.size() == 4096 * 4);

    AlignedBuffer empty(0);
    CHECK(!empty.valid());
    CHECK(empty.size() == 0);
    empty.zero();   // must not crash

    // Move must transfer ownership exactly once, or the free is doubled.
    AlignedBuffer src(8192);
    REQUIRE(src.valid());
    src.data()[0] = 0x5A;
    const std::uint8_t* original = src.data();
    AlignedBuffer moved = std::move(src);
    CHECK(!src.valid());
    CHECK(moved.data() == original);
    CHECK(moved.data()[0] == 0x5A);
    AlignedBuffer assigned;
    assigned = std::move(moved);
    CHECK(!moved.valid());
    CHECK(assigned.data() == original);
    CHECK(assigned.data()[0] == 0x5A);
}

// ======================================================================
// 2. Directory helpers
// ======================================================================
void test_directories(const std::string& dir) {
    const std::string nested = dir + "/a/b/c";
    CHECK(ensure_directory(nested).ok());
    struct stat st{};
    CHECK(::stat(nested.c_str(), &st) == 0);
    CHECK(S_ISDIR(st.st_mode));

    // Idempotent. Several ranks bring up the same namespace root concurrently,
    // so "already exists" is the normal case, not an error.
    CHECK(ensure_directory(nested).ok());
    CHECK(ensure_directory(dir).ok());

    CHECK(!ensure_directory("").ok());
    CHECK(sync_directory(dir).ok());
    CHECK(!sync_directory(dir + "/does_not_exist").ok());

    ::rmdir((dir + "/a/b/c").c_str());
    ::rmdir((dir + "/a/b").c_str());
    ::rmdir((dir + "/a").c_str());
}

// ======================================================================
// 3. Materialisation -- must allocate REAL blocks, not a sparse file
// ======================================================================
void test_materialisation(const std::string& dir) {
    const std::string path = dir + "/mat/0.seg";
    // Two small slots; the property under test is block allocation, not size.
    constexpr std::uint64_t kSmallSlot = 4096 * 16;  // 64 KiB
    constexpr std::uint64_t kFile = 2 * kSmallSlot;
    CHECK(materialise_segment_file(path, kFile).ok());

    struct stat st{};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == kFile);
    // THE point of materialisation: blocks must actually be allocated. The
    // resolver maps files to physical extents via FIEMAP and fail-closed rejects
    // UNWRITTEN/DELALLOC extents, because DMA cannot target blocks the
    // filesystem has not committed.
    CHECK(static_cast<std::uint64_t>(st.st_blocks) * 512 >= kFile);
    bool ready = false;
    CHECK(segment_file_is_ready(path, kFile, &ready).ok());
    CHECK(ready);

    // Idempotent: a ready file is left alone, so re-materialising preserves a
    // committed object (tested by consequence rather than mtime).
    const ObjectKey survivor = key_of(0x7E);
    const std::uint64_t payload = kSmallSlot - ObjectHeaderLayout::kHeaderBytes;
    std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
    REQUIRE(encode_object_header(header.data(), header.size(), survivor,
                                 payload, 5, 55));
    REQUIRE(write_object_header(path, kSmallSlot, header.data(), header.size()).ok());
    CHECK(materialise_segment_file(path, kFile).ok());
    std::vector<std::uint8_t> readback(ObjectHeaderLayout::kHeaderBytes);
    CHECK(read_object_header(path, kSmallSlot, readback.data(), readback.size()).ok());
    HeaderExpectation still;
    still.key = &survivor;
    still.payload_bytes = payload;
    still.generation = 5;
    CHECK(decode_object_header(readback.data(), readback.size(), still,
                               nullptr) == HeaderRejection::kNone);

    // Unaligned or zero sizes are refused rather than silently rounded.
    CHECK(!materialise_segment_file(path, 0).ok());
    CHECK(!materialise_segment_file(path, 1000).ok());
    // A file larger than configured is a geometry mismatch, not reusable.
    CHECK(!materialise_segment_file(path, kSmallSlot).ok());
    bool absent_ready = true;
    CHECK(segment_file_is_ready(dir + "/mat/absent.seg", kFile, &absent_ready).ok());
    CHECK(!absent_ready);

    // --- zeroing one slot invalidates its object and only its object ---
    const ObjectKey first = key_of(0x42);
    REQUIRE(encode_object_header(header.data(), header.size(), first, payload, 1, 1));
    CHECK(write_object_header(path, 0, header.data(), header.size()).ok());
    ObjectHeaderFields fields;
    CHECK(read_object_header(path, 0, readback.data(), readback.size()).ok());
    CHECK(peek_object_header(readback.data(), readback.size(), &fields) ==
          HeaderRejection::kNone);

    CHECK(zero_file_range(path, 0, kSmallSlot).ok());
    CHECK(read_object_header(path, 0, readback.data(), readback.size()).ok());
    CHECK(peek_object_header(readback.data(), readback.size(), &fields) ==
          HeaderRejection::kEmptySlot);
    CHECK(read_object_header(path, kSmallSlot, readback.data(), readback.size()).ok());
    CHECK(decode_object_header(readback.data(), readback.size(), still,
                               nullptr) == HeaderRejection::kNone);

    CHECK(!zero_file_range(path, 1, kSmallSlot).ok());
    CHECK(!zero_file_range(dir + "/absent.seg", 0, kSmallSlot).ok());
}

// ======================================================================
// 4. Header IO round trip through real media
// ======================================================================
void test_header_io(const std::string& dir) {
    const std::string path = dir + "/hdr/0.seg";
    constexpr std::uint64_t kSmallSlot = 4096 * 8;
    REQUIRE(materialise_segment_file(path, 2 * kSmallSlot).ok());
    const std::vector<std::string> paths = {path};
    const ObjectKey key = key_of(0xC7);
    const std::uint64_t payload = kSmallSlot - ObjectHeaderLayout::kHeaderBytes;

    std::vector<std::uint8_t> header(ObjectHeaderLayout::kHeaderBytes);
    REQUIRE(encode_object_header(header.data(), header.size(), key, payload, 3, 99));
    CHECK(write_object_header(paths[0], 0, header.data(), header.size()).ok());

    std::vector<std::uint8_t> readback(ObjectHeaderLayout::kHeaderBytes);
    CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
    // Byte-identical: the encoding is deterministic, which is what makes a torn
    // write detectable.
    CHECK(std::memcmp(header.data(), readback.data(), header.size()) == 0);

    HeaderExpectation expect;
    expect.key = &key;
    expect.payload_bytes = payload;
    expect.generation = 3;
    ObjectHeaderFields fields;
    CHECK(decode_object_header(readback.data(), readback.size(), expect, &fields) ==
          HeaderRejection::kNone);
    CHECK(fields.commit_seq == 99);

    // A never-written slot reads as empty, not as an error: this is the expected
    // state of free space and must not look like corruption.
    CHECK(read_object_header(paths[0], kSmallSlot, readback.data(), readback.size()).ok());
    CHECK(peek_object_header(readback.data(), readback.size(), &fields) ==
          HeaderRejection::kEmptySlot);

    // An absent slot is NOT_FOUND, distinct from an IO failure.
    const Status missing =
        read_object_header(dir + "/nope.obj", 0, readback.data(), readback.size());
    CHECK(!missing.ok());
    CHECK(missing.code() == StatusCode::NOT_FOUND);

    // O_DIRECT alignment is enforced, not silently worked around: an unaligned
    // request means the caller's geometry is wrong.
    CHECK(!write_object_header(paths[0], 1, header.data(), header.size()).ok());
    CHECK(!write_object_header(paths[0], 0, header.data(), 100).ok());
    CHECK(!write_object_header(paths[0], 0, nullptr, header.size()).ok());
    CHECK(!read_object_header(paths[0], 1, readback.data(), readback.size()).ok());
    CHECK(!read_object_header(paths[0], 0, readback.data(), 100).ok());

    // Overwriting a header in place works, so a slot can be recommitted without
    // being recreated.
    const ObjectKey second = key_of(0xD8);
    REQUIRE(encode_object_header(header.data(), header.size(), second, payload, 4, 100));
    CHECK(write_object_header(paths[0], 0, header.data(), header.size()).ok());
    CHECK(read_object_header(paths[0], 0, readback.data(), readback.size()).ok());
    HeaderExpectation second_expect;
    second_expect.key = &second;
    second_expect.payload_bytes = payload;
    second_expect.generation = 4;
    CHECK(decode_object_header(readback.data(), readback.size(), second_expect,
                              nullptr) == HeaderRejection::kNone);
    // And the previous key no longer resolves there.
    CHECK(decode_object_header(readback.data(), readback.size(), expect, nullptr) ==
          HeaderRejection::kIdentityMismatch);
}

// ======================================================================
// 5. Checkpoint IO round trip, including the crash-recovery sequence
// ======================================================================
void test_checkpoint_io(const std::string& dir) {
    const std::string path = dir + "/meta/checkpoint.bin";

    std::vector<CheckpointEntry> entries;
    for (std::uint8_t i = 1; i <= 4; ++i) {
        CheckpointEntry e;
        e.key = key_of(i);
        e.slot = i;
        e.payload_bytes = kPayload;
        e.generation = i;
        e.commit_seq = 200 + i;
        entries.push_back(e);
    }

    const std::uint64_t container = checkpoint_container_bytes(64, 18);
    const std::uint64_t region = container * CheckpointLayout::kContainerCount;
    CHECK(ensure_metadata_file(path, region).ok());

    struct stat st{};
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == region);
    // The checkpoint region is materialised too: it is read with O_DIRECT, so
    // its blocks must exist.
    CHECK(static_cast<std::uint64_t>(st.st_blocks) * 512 >= region);

    // Idempotent, and never shrinks an existing region.
    CHECK(ensure_metadata_file(path, region).ok());
    REQUIRE(::stat(path.c_str(), &st) == 0);
    CHECK(static_cast<std::uint64_t>(st.st_size) == region);

    // --- write into container 0, read it back ---
    const std::vector<std::uint8_t> image = encode_checkpoint(entries, 1);
    REQUIRE(image.size() <= container);
    CHECK(write_checkpoint_container(path, checkpoint_container_offset(0, container),
                                    image.data(), image.size()).ok());

    std::vector<std::uint8_t> readback(image.size());
    CHECK(read_checkpoint_container(path, checkpoint_container_offset(0, container),
                                   readback.data(), readback.size()).ok());
    std::uint64_t seq = 0;
    std::vector<CheckpointEntry> decoded;
    CHECK(decode_checkpoint(readback.data(), readback.size(), &seq, &decoded) ==
          CheckpointRejection::kNone);
    CHECK(seq == 1);
    CHECK(decoded.size() == entries.size());

    // --- container 1 is still empty, and reads as such ---
    std::vector<std::uint8_t> empty(CheckpointLayout::kContainerHeaderBytes);
    CHECK(read_checkpoint_container(path, checkpoint_container_offset(1, container),
                                   empty.data(), empty.size()).ok());
    CHECK(decode_checkpoint(empty.data(), empty.size(), &seq, &decoded) ==
          CheckpointRejection::kEmptyContainer);

    // --- the crash-recovery sequence end to end ---
    // Rotation must overwrite the OLDEST container, so the newest valid state
    // survives a crash during the write.
    {
        std::uint64_t seqs[2] = {1, 0};
        CHECK(next_checkpoint_container(seqs, 2) == 1);

        const std::vector<std::uint8_t> second = encode_checkpoint(entries, 2);
        CHECK(write_checkpoint_container(path,
                                        checkpoint_container_offset(1, container),
                                        second.data(), second.size()).ok());
        seqs[1] = 2;
        // Next write targets container 0 again -- never the one holding the
        // newest state.
        CHECK(next_checkpoint_container(seqs, 2) == 0);
        std::uint32_t chosen = 99;
        CHECK(select_checkpoint_container(seqs, 2, &chosen));
        CHECK(chosen == 1);

        // Simulate a crash midway through overwriting container 0: its body is
        // corrupt. Recovery must fall back to container 1, losing nothing that
        // was durable.
        std::vector<std::uint8_t> torn = encode_checkpoint(entries, 3);
        torn[CheckpointLayout::kContainerHeaderBytes + 8] ^= 0xFF;
        CHECK(write_checkpoint_container(path,
                                        checkpoint_container_offset(0, container),
                                        torn.data(), torn.size()).ok());

        std::uint64_t found[2] = {0, 0};
        for (std::uint32_t i = 0; i < 2; ++i) {
            std::vector<std::uint8_t> buf(container);
            if (!read_checkpoint_container(path,
                                           checkpoint_container_offset(i, container),
                                           buf.data(), buf.size()).ok()) {
                continue;
            }
            std::uint64_t s = 0;
            std::vector<CheckpointEntry> tmp;
            if (decode_checkpoint(buf.data(), buf.size(), &s, &tmp) ==
                CheckpointRejection::kNone) {
                found[i] = s;
            }
        }
        CHECK(found[0] == 0);   // torn container is unusable
        CHECK(found[1] == 2);   // previous state intact
        CHECK(select_checkpoint_container(found, 2, &chosen));
        CHECK(chosen == 1);
    }

    // --- alignment is enforced ---
    CHECK(!write_checkpoint_container(path, 1, image.data(), image.size()).ok());
    CHECK(!write_checkpoint_container(path, 0, image.data(), 100).ok());
    CHECK(!read_checkpoint_container(path, 1, readback.data(), readback.size()).ok());
    CHECK(!ensure_metadata_file(path, 100).ok());
    CHECK(!ensure_metadata_file(path, 0).ok());

    const Status absent = read_checkpoint_container(dir + "/meta/nope.bin", 0,
                                                    readback.data(),
                                                    readback.size());
    CHECK(!absent.ok());
    CHECK(absent.code() == StatusCode::NOT_FOUND);
}

} // namespace

int main() {
    // Pure computation: always run.
    test_aligned_buffer();

    const std::string dir = temp_dir();
    if (dir.empty()) {
        std::printf("FAIL: could not create a temporary directory\n");
        return 1;
    }

    test_directories(dir);

    if (o_direct_supported(dir)) {
        test_materialisation(dir);
        test_header_io(dir);
        test_checkpoint_io(dir);
    } else {
        // Reported loudly: a green run that never exercised the O_DIRECT path
        // would be more misleading than no run at all.
        std::printf("SKIP: %s does not support O_DIRECT; media tests not run.\n",
                    dir.c_str());
        std::printf("      Set TMPDIR to a directory on a real block-backed "
                    "filesystem to exercise them.\n");
        ++g_skipped;
    }

    // Best-effort cleanup; leftovers in a temp dir are harmless.
    std::string cmd = "rm -rf '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::printf("note: could not remove %s\n", dir.c_str());
    }

    if (g_failures == 0) {
        std::printf("storage_objects media contract: all checks passed%s\n",
                    g_skipped != 0 ? " (with skips)" : "");
        return 0;
    }
    std::printf("storage_objects media contract: %d failure(s)\n", g_failures);
    return 1;
}
