// RAID0 hardware contract: files on an md RAID0 over Tutti-owned NVMe
// namespaces, read and written by the GPU directly on the members.
//
// The kernel md + ext4 path is the oracle. Every byte the GPU writes must
// read back identically through the filesystem (O_DIRECT, so no page cache
// can mask a wrong member/offset), and every byte the filesystem writes must
// read back identically on the GPU. Requests have random lengths and
// offsets, so they cross RAID0 chunk boundaries at every phase, and each
// batch mixes two files (two RAID0 targets in one device table).
//
// Needs: tutti_daemon up with raid0_arrays (config/local/tutti_daemon_kv0.yaml),
// the array mounted. Skips (77) when the directory is not on md RAID0.
//
//   tutti_raid0_local_nvme_contract_test [--directory DIR] [--config YAML]

#include <tutti/cuda_like.h>
#include <tutti/io_types.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>

#include "csrc/common/md_raid0.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tutti;

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (cond) {                                                   \
            ++g_pass;                                                 \
        } else {                                                      \
            ++g_fail;                                                 \
            std::fprintf(stderr, "[FAIL] %s:%d ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
        }                                                             \
    } while (0)

constexpr std::uint64_t kFileBytes = 256ull << 20;
constexpr std::uint64_t kBlock = 4096;
constexpr int kFiles = 2;

struct AlignedBuf {
    void* p = nullptr;
    explicit AlignedBuf(std::size_t n) {
        if (::posix_memalign(&p, kBlock, n) != 0) p = nullptr;
    }
    ~AlignedBuf() { std::free(p); }
    std::uint8_t* data() const { return static_cast<std::uint8_t*>(p); }
};

void fill_random(std::uint8_t* p, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto* w = reinterpret_cast<std::uint64_t*>(p);
    for (std::size_t i = 0; i < n / 8; ++i) w[i] = rng();
}

bool write_file(const std::string& path, const std::uint8_t* data,
                std::uint64_t bytes, bool create) {
    const int fd = ::open(path.c_str(),
                          O_RDWR | O_DIRECT | (create ? O_CREAT | O_TRUNC : 0),
                          0644);
    if (fd < 0) return false;
    bool ok = !create || ::fallocate(fd, 0, 0, static_cast<off_t>(bytes)) == 0;
    for (std::uint64_t off = 0; ok && off < bytes; off += 8u << 20) {
        const std::size_t n = std::min<std::uint64_t>(8u << 20, bytes - off);
        ok = ::pwrite(fd, data + off, n, static_cast<off_t>(off)) ==
             static_cast<ssize_t>(n);
    }
    ok = ok && ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

bool read_file(const std::string& path, std::uint8_t* data, std::uint64_t bytes) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) return false;
    bool ok = true;
    for (std::uint64_t off = 0; ok && off < bytes; off += 8u << 20) {
        const std::size_t n = std::min<std::uint64_t>(8u << 20, bytes - off);
        ok = ::pread(fd, data + off, n, static_cast<off_t>(off)) ==
             static_cast<ssize_t>(n);
    }
    ::close(fd);
    return ok;
}

struct Piece {
    int file;
    std::uint64_t offset;
    std::uint64_t length;
};

// Every byte of every file exactly once, in random-length pieces, shuffled.
std::vector<Piece> cover(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Piece> pieces;
    for (int f = 0; f < kFiles; ++f) {
        for (std::uint64_t off = 0; off < kFileBytes;) {
            std::uint64_t len = (1 + rng() % 384) * kBlock;  // 4 KiB .. 1.5 MiB
            len = std::min(len, kFileBytes - off);
            pieces.push_back({f, off, len});
            off += len;
        }
    }
    std::shuffle(pieces.begin(), pieces.end(), rng);
    return pieces;
}

// Submits all pieces in batches; memory offset = file * kFileBytes + offset.
bool run(StorageRuntime* rt, IoDirection dir, MemoryHandle memory,
         const TargetHandle* targets, const std::vector<Piece>& pieces,
         const HostSubmitContext& context, std::string* error) {
    constexpr std::size_t kBatch = 48;
    for (std::size_t first = 0; first < pieces.size(); first += kBatch) {
        std::vector<IoRequest> reqs;
        for (std::size_t i = first; i < std::min(first + kBatch, pieces.size()); ++i) {
            const Piece& p = pieces[i];
            reqs.push_back({dir, memory, p.file * kFileBytes + p.offset,
                            targets[p.file], p.offset, p.length});
        }
        auto submitted = rt->submit(reqs.data(), reqs.size(), context);
        if (!submitted.io.has_value()) {
            *error = "submit: " + submitted.status.message();
            return false;
        }
        for (const auto& s : submitted.initial_states) {
            if (s.state != IoRequestState::ACCEPTED) {
                *error = "request not accepted: " + s.status.message();
                rt->wait(submitted.io.value(), 60000);
                rt->release_io(submitted.io.value());
                return false;
            }
        }
        auto waited = rt->wait(submitted.io.value(), 60000);
        const bool ok = waited.result && waited.result->state == IoState::COMPLETED;
        rt->release_io(submitted.io.value());
        if (!ok) {
            *error = "wait did not complete";
            return false;
        }
    }
    return true;
}

std::uint64_t first_mismatch(const std::uint8_t* a, const std::uint8_t* b,
                             std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; i += kBlock) {
        if (std::memcmp(a + i, b + i, kBlock) != 0) return i;
    }
    return UINT64_MAX;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "/mnt/tutti_md0";
    std::string config = TUTTI_RAID0_TEST_DEFAULT_CONFIG;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--directory") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--directory DIR] [--config YAML]\n", argv[0]);
            return 2;
        }
    }

    namespace md = tutti::detail::md_raid0;
    md::Geometry geometry;
    std::string why;
    if (!md::probe_path(dir, &geometry, &why)) {
        std::printf("[SKIP] %s is not on a usable md RAID0: %s\n", dir.c_str(), why.c_str());
        return 77;
    }
    std::printf("array %s: %zu members, chunk %llu KiB, member data %llu GiB\n",
                geometry.md_name.c_str(), geometry.members.size(),
                static_cast<unsigned long long>(geometry.chunk_bytes >> 10),
                static_cast<unsigned long long>(geometry.member_data_bytes >> 30));
    for (std::size_t i = 0; i < geometry.members.size(); ++i) {
        std::printf("  slot %zu: %s data_offset=%llu\n", i,
                    geometry.members[i].block_name.c_str(),
                    static_cast<unsigned long long>(geometry.members[i].data_offset_bytes));
    }

    const std::string work = dir + "/raid0_contract." + std::to_string(::getpid());
    if (::mkdir(work.c_str(), 0755) != 0) {
        std::fprintf(stderr, "mkdir %s: %s\n", work.c_str(), std::strerror(errno));
        return 1;
    }
    std::vector<std::string> paths;
    for (int f = 0; f < kFiles; ++f) paths.push_back(work + "/f" + std::to_string(f) + ".bin");

    const std::uint64_t total = kFileBytes * kFiles;
    AlignedBuf host(total), check(total);
    CHECK(host.p && check.p, "host buffers");

    // Files with known content, written by the kernel.
    fill_random(host.data(), total, 1);
    for (int f = 0; f < kFiles; ++f) {
        CHECK(write_file(paths[f], host.data() + f * kFileBytes, kFileBytes, true),
              "create %s: %s", paths[f].c_str(), std::strerror(errno));
    }

    auto created = TuttiRuntime::create(config);
    if (!created.ok()) {
        std::fprintf(stderr, "runtime: %s\n", created.status().message().c_str());
        return 1;
    }
    auto owner = std::move(created).value();
    StorageRuntime* rt = owner->storage_runtime();
    cudaSetDevice(rt->accel_id());

    TargetHandle targets[kFiles];
    for (int f = 0; f < kFiles; ++f) {
        auto opened = rt->open("file://" + paths[f], OpenOptions{"file"});
        CHECK(opened.ok(), "open %s: %s", paths[f].c_str(),
              opened.ok() ? "" : opened.status().message().c_str());
        if (!opened.ok()) return 1;
        targets[f] = opened.value();
    }

    void* raw = nullptr;
    CHECK(cudaMalloc(&raw, total + 65536) == cudaSuccess, "cudaMalloc");
    void* gpu = reinterpret_cast<void*>(
        (reinterpret_cast<std::uintptr_t>(raw) + 65535) & ~std::uintptr_t(65535));
    auto mem = rt->register_memory({gpu, total, MemoryKind::DEVICE,
                                    MemoryOwnership::CALLER_OWNED, rt->accel_id(),
                                    TUTTI_COMPILED_ACCELERATOR_PROFILE, kBlock});
    CHECK(mem.ok(), "register_memory: %s", mem.ok() ? "" : mem.status().message().c_str());
    if (!mem.ok()) return 1;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    const HostSubmitContext context{ExecutionDomain::DEVICE_EXECUTION, rt->accel_id(), stream};
    std::string error;

    // 1. Kernel wrote, GPU reads.
    cudaMemset(gpu, 0xA5, total);
    cudaDeviceSynchronize();
    bool ok = run(rt, IoDirection::READ, mem.value(), targets, cover(2), context, &error);
    CHECK(ok, "GPU read: %s", error.c_str());
    cudaMemcpy(check.data(), gpu, total, cudaMemcpyDeviceToHost);
    std::uint64_t bad = first_mismatch(host.data(), check.data(), total);
    CHECK(bad == UINT64_MAX, "GPU read differs from the filesystem at byte %llu",
          static_cast<unsigned long long>(bad));
    std::printf("[%s] GPU read == kernel write (%llu MiB, 2 files, random pieces)\n",
                bad == UINT64_MAX && ok ? " OK " : "FAIL",
                static_cast<unsigned long long>(total >> 20));

    // 2. GPU writes, kernel reads.
    fill_random(host.data(), total, 3);
    cudaMemcpy(gpu, host.data(), total, cudaMemcpyHostToDevice);
    ok = run(rt, IoDirection::WRITE, mem.value(), targets, cover(4), context, &error);
    CHECK(ok, "GPU write: %s", error.c_str());
    for (int f = 0; f < kFiles; ++f) {
        CHECK(read_file(paths[f], check.data() + f * kFileBytes, kFileBytes),
              "pread %s", paths[f].c_str());
    }
    bad = first_mismatch(host.data(), check.data(), total);
    CHECK(bad == UINT64_MAX, "filesystem read differs from GPU write at byte %llu",
          static_cast<unsigned long long>(bad));
    std::printf("[%s] kernel read == GPU write\n", bad == UINT64_MAX && ok ? " OK " : "FAIL");

    // 3. Kernel overwrites in place (same extents), GPU sees it.
    fill_random(host.data(), total, 5);
    for (int f = 0; f < kFiles; ++f) {
        CHECK(write_file(paths[f], host.data() + f * kFileBytes, kFileBytes, false),
              "rewrite %s", paths[f].c_str());
    }
    ok = run(rt, IoDirection::READ, mem.value(), targets, cover(6), context, &error);
    CHECK(ok, "GPU re-read: %s", error.c_str());
    cudaMemcpy(check.data(), gpu, total, cudaMemcpyDeviceToHost);
    bad = first_mismatch(host.data(), check.data(), total);
    CHECK(bad == UINT64_MAX, "GPU re-read differs at byte %llu",
          static_cast<unsigned long long>(bad));
    std::printf("[%s] GPU re-read after in-place kernel rewrite\n",
                bad == UINT64_MAX && ok ? " OK " : "FAIL");

    // 4. Out-of-file request is rejected, not sent to a member.
    {
        IoRequest r{IoDirection::READ, mem.value(), 0, targets[0], kFileBytes, kBlock};
        auto submitted = rt->submit(&r, 1, context);
        const bool rejected = !submitted.io.has_value() ||
            (submitted.initial_states.size() == 1 &&
             submitted.initial_states[0].state != IoRequestState::ACCEPTED);
        if (submitted.io.has_value()) {
            rt->wait(submitted.io.value(), 60000);
            rt->release_io(submitted.io.value());
        }
        CHECK(rejected, "read past end of file was accepted");
    }

    rt->unregister_memory(mem.value());
    for (int f = 0; f < kFiles; ++f) rt->close(targets[f]);
    cudaStreamDestroy(stream);
    cudaFree(raw);

    if (g_fail == 0) {
        for (const auto& p : paths) ::unlink(p.c_str());
        ::rmdir(work.c_str());
    }
    std::printf("raid0 contract: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
