// async_completion_bench — correctness and compute-overlap benchmark for
// the submit/drain split (doc/design/async-nvme-completion.md).
//
// The mode comes from the environment, like every other component:
//   TUTTI_ASYNC_COMPLETION=1  StripedAsyncDataPath (submit-only IO kernel)
//   unset                     StripedDataPath      (submit-and-wait kernel)
//
// --verify   write a GPU pattern through Tutti, read it back into another
//            buffer, compare on the GPU; then a stress pass with many more
//            entries per batch than the queues hold, reads and writes on two
//            streams at once, read back and compared again.
// --overlap  layer-wise reads overlapped with cuBLAS GEMMs on another stream
//            (the vLLM pattern: all layers' reads submitted up front, compute
//            waits for layer L before its GEMMs). Reports the GEMM time with
//            no IO, and with IO in flight.

#define _GNU_SOURCE

#include <tutti/io_types.h>
#include <tutti/storage_runtime.h>
#include <tutti/tutti_runtime.h>
#include <tutti/cuda_like.h>

#include <cublas_v2.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tutti;

#define FAIL(...) do { std::fprintf(stderr, "[FAIL] " __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } while (0)
#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) FAIL("%s: %s", #call, cudaGetErrorString(e_)); } while (0)
#define CUBLAS_OK(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) FAIL("%s: %d", #call, (int)s_); } while (0)

namespace {

constexpr std::uint64_t kEntryBytes = 128 * 1024;

} // namespace

// async_completion_bench_kernels.cu
void bench_fill_pattern(std::uint64_t* p, std::uint64_t words, std::uint64_t seed, cudaStream_t s);
void bench_count_mismatch(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t words,
                          unsigned long long* bad, cudaStream_t s);
int bench_persistent_setup(int device, int* sm_count);
void bench_persistent_compute(float* sink, std::uint32_t iters, int sm_count, int smem,
                              cudaStream_t s);

namespace {

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

bool ensure_file(const std::string& path, std::uint64_t bytes) {
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0 && static_cast<std::uint64_t>(st.st_size) == bytes) return true;
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) return false;
    if (::fallocate(fd, 0, 0, static_cast<off_t>(bytes)) != 0) { ::close(fd); return false; }
    void* raw = nullptr;
    if (::posix_memalign(&raw, 4096, 4 << 20) != 0) { ::close(fd); return false; }
    std::memset(raw, 0, 4 << 20);
    for (std::uint64_t off = 0; off < bytes; off += 4 << 20) {
        const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(4 << 20, bytes - off));
        if (::pwrite(fd, raw, n, static_cast<off_t>(off)) != static_cast<ssize_t>(n)) {
            std::free(raw); ::close(fd); return false;
        }
    }
    std::free(raw);
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

struct Bench {
    StorageRuntime* rt = nullptr;
    MemoryHandle mem{};
    std::vector<TargetHandle> targets;
    std::uint64_t file_bytes = 0;
    bool async_mode = false;

    // Submit `reqs` (all accepted or fail), return the handle.
    IoHandle submit(const std::vector<IoRequest>& reqs, cudaStream_t stream) {
        HostSubmitContext ctx{ExecutionDomain::DEVICE_EXECUTION, rt->accel_id(), stream};
        auto out = rt->submit(reqs.data(), reqs.size(), ctx);
        if (!out.io.has_value()) FAIL("submit: %s", out.status.message().c_str());
        for (std::size_t i = 0; i < out.initial_states.size(); ++i) {
            if (out.initial_states[i].state != IoRequestState::ACCEPTED)
                FAIL("request %zu rejected: %s", i, out.initial_states[i].status.message().c_str());
        }
        return out.io.value();
    }
    // Device-side wait on `stream` (async mode), as the vLLM worker does.
    void stream_wait(const IoHandle& h, cudaStream_t stream) {
        Status st = rt->enqueue_wait(&h, 1, stream);
        if (!st.ok()) FAIL("enqueue_wait: %s", st.message().c_str());
    }
    void host_wait(const IoHandle& h) {
        auto w = rt->wait(h, 120000);
        if (!w.result || w.result->state != IoState::COMPLETED)
            FAIL("wait: %s", w.result ? w.result->status.message().c_str() : "no result");
        rt->release_io(h);
    }
    // `count` requests of kEntryBytes, entry i at buffer offset
    // (first + i) * kEntryBytes, spread round-robin over the files at
    // distinct aligned offsets.
    std::vector<IoRequest> requests(IoDirection dir, std::uint64_t first, std::uint64_t count,
                                    std::uint64_t file_slot_base) const {
        std::vector<IoRequest> r;
        r.reserve(count);
        const std::uint64_t per_file = file_bytes / kEntryBytes;
        for (std::uint64_t i = 0; i < count; ++i) {
            const std::uint64_t global = file_slot_base + first + i;
            const std::uint64_t file = global % targets.size();
            const std::uint64_t slot = (global / targets.size()) % per_file;
            r.push_back({dir, mem, (first + i) * kEntryBytes, targets[file], slot * kEntryBytes,
                         kEntryBytes});
        }
        return r;
    }
};

// Write src through Tutti, read back into dst, compare. `batch` entries per
// submit; reads and writes interleaved on two streams in the stress pass.
void verify(Bench& b, std::uint64_t* d_src, std::uint64_t* d_dst, std::uint64_t entries,
            std::uint64_t batch, bool two_streams, const char* label) {
    const std::uint64_t words = entries * kEntryBytes / 8;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CUDA_OK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    bench_fill_pattern(d_src, words, 0x5eed0000ull + entries, s0);
    CUDA_OK(cudaMemsetAsync(d_dst, 0, words * 8, s0));
    CUDA_OK(cudaStreamSynchronize(s0));
    // The registered region is [pool, pool + entries * kEntryBytes): src is
    // the first half, dst the second; requests address src, reads land in dst
    // by offsetting memory by the half size.
    const std::uint64_t half = entries * kEntryBytes;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<IoHandle> writes;
    for (std::uint64_t first = 0; first < entries; first += batch) {
        const std::uint64_t n = std::min(batch, entries - first);
        auto reqs = b.requests(IoDirection::WRITE, first, n, 0);
        cudaStream_t s = (two_streams && (first / batch) % 2) ? s1 : s0;
        IoHandle h = b.submit(reqs, s);
        if (b.async_mode) b.stream_wait(h, s);
        writes.push_back(h);
    }
    for (auto& h : writes) b.host_wait(h);
    std::vector<IoHandle> reads;
    for (std::uint64_t first = 0; first < entries; first += batch) {
        const std::uint64_t n = std::min(batch, entries - first);
        auto reqs = b.requests(IoDirection::READ, first, n, 0);
        for (auto& r : reqs) r.memory_offset += half;
        cudaStream_t s = (two_streams && (first / batch) % 2) ? s1 : s0;
        IoHandle h = b.submit(reqs, s);
        if (b.async_mode) b.stream_wait(h, s);
        reads.push_back(h);
    }
    for (auto& h : reads) b.host_wait(h);
    const double secs = seconds_since(t0);
    unsigned long long* d_bad = nullptr;
    CUDA_OK(cudaMalloc(&d_bad, sizeof(*d_bad)));
    CUDA_OK(cudaMemsetAsync(d_bad, 0, sizeof(*d_bad), s0));
    CUDA_OK(cudaStreamSynchronize(s1));
    bench_count_mismatch(d_src, d_dst, words, d_bad, s0);
    unsigned long long bad = 0;
    CUDA_OK(cudaMemcpyAsync(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost, s0));
    CUDA_OK(cudaStreamSynchronize(s0));
    cudaFree(d_bad);
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    std::printf("[%s] verify %s: %llu entries x 128 KiB, batch %llu%s, %.2f GiB/s (write+read), "
                "mismatched words %llu\n",
                bad == 0 ? " OK " : "FAIL", label, (unsigned long long)entries,
                (unsigned long long)batch, two_streams ? ", two streams" : "",
                2.0 * half / (1ull << 30) / secs, bad);
    if (bad != 0) std::exit(1);
}

struct Gemm {
    bool persistent = true;     // --compute persistent (default) | cublas
    int sm_count = 0, smem = 0;
    std::uint32_t iters = 0;
    float* sink = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t h = nullptr;
    std::uint16_t *a = nullptr, *bm = nullptr, *c = nullptr;
    int n = 0;
    // Persistent: calibrate `iters` so one kernel takes about `kernel_us`.
    void init_persistent(int device, double kernel_us, cudaStream_t s) {
        stream = s;
        smem = bench_persistent_setup(device, &sm_count);
        CUDA_OK(cudaMalloc(&sink, sm_count * sizeof(float)));
        cudaEvent_t e0, e1;
        CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1));
        std::uint32_t probe = 1u << 16;
        bench_persistent_compute(sink, probe, sm_count, smem, s);
        CUDA_OK(cudaEventRecord(e0, s));
        bench_persistent_compute(sink, probe, sm_count, smem, s);
        CUDA_OK(cudaEventRecord(e1, s));
        CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
        iters = static_cast<std::uint32_t>(probe * (kernel_us / 1000.0) / ms);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        std::printf("persistent compute: grid=%d (SMs) smem/block=%d B, %u iters ~ %.0f us/kernel\n",
                    sm_count, smem, iters, kernel_us);
    }
    void init(int size, cudaStream_t s) {
        persistent = false;
        n = size;
        CUBLAS_OK(cublasCreate(&h));
        CUBLAS_OK(cublasSetStream(h, s));
        const std::size_t bytes = (std::size_t)n * n * sizeof(std::uint16_t);
        CUDA_OK(cudaMalloc(&a, bytes)); CUDA_OK(cudaMalloc(&bm, bytes)); CUDA_OK(cudaMalloc(&c, bytes));
        CUDA_OK(cudaMemsetAsync(a, 0, bytes, s)); CUDA_OK(cudaMemsetAsync(bm, 0, bytes, s));
    }
    void run() {
        if (persistent) { bench_persistent_compute(sink, iters, sm_count, smem, stream); return; }
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a, CUDA_R_16BF, n,
                               bm, CUDA_R_16BF, n, &beta, c, CUDA_R_16BF, n,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
};

struct OverlapResult {
    double wall_ms = 0;       // first compute start .. last compute end (GPU)
    double gemm_ms = 0;       // sum of per-layer GEMM time (GPU)
    double io_gib_s = 0;
};

// One step: reads of every layer submitted up front on io_stream; compute
// waits for layer L, then runs `gemms` GEMMs. with_io=false: GEMMs only.
OverlapResult overlap_step(Bench& b, Gemm& g, cudaStream_t io, cudaStream_t compute,
                           std::uint32_t layers, std::uint64_t entries_per_layer,
                           std::uint32_t gemms, bool with_io, std::uint64_t file_base) {
    std::vector<cudaEvent_t> submitted(layers), l0(layers), l1(layers);
    for (std::uint32_t l = 0; l < layers; ++l) {
        CUDA_OK(cudaEventCreate(&submitted[l]));
        CUDA_OK(cudaEventCreate(&l0[l]));
        CUDA_OK(cudaEventCreate(&l1[l]));
    }
    CUDA_OK(cudaStreamSynchronize(io));
    CUDA_OK(cudaStreamSynchronize(compute));
    auto t0 = std::chrono::steady_clock::now();
    std::vector<IoHandle> handles;
    if (with_io) {
        for (std::uint32_t l = 0; l < layers; ++l) {
            auto reqs = b.requests(IoDirection::READ, (std::uint64_t)l * entries_per_layer,
                                   entries_per_layer, file_base);
            handles.push_back(b.submit(reqs, io));
            CUDA_OK(cudaEventRecord(submitted[l], io));
        }
    }
    for (std::uint32_t l = 0; l < layers; ++l) {
        if (with_io) {
            if (b.async_mode) b.stream_wait(handles[l], compute);
            else CUDA_OK(cudaStreamWaitEvent(compute, submitted[l], 0));
        }
        CUDA_OK(cudaEventRecord(l0[l], compute));
        for (std::uint32_t k = 0; k < gemms; ++k) g.run();
        CUDA_OK(cudaEventRecord(l1[l], compute));
    }
    CUDA_OK(cudaStreamSynchronize(compute));
    for (auto& h : handles) b.host_wait(h);
    const double host_s = seconds_since(t0);
    OverlapResult r;
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, l0[0], l1[layers - 1]));
    r.wall_ms = ms;
    for (std::uint32_t l = 0; l < layers; ++l) {
        CUDA_OK(cudaEventElapsedTime(&ms, l0[l], l1[l]));
        r.gemm_ms += ms;
    }
    if (with_io) {
        r.io_gib_s = (double)layers * entries_per_layer * kEntryBytes / (1ull << 30) / host_s;
    }
    for (std::uint32_t l = 0; l < layers; ++l) {
        cudaEventDestroy(submitted[l]); cudaEventDestroy(l0[l]); cudaEventDestroy(l1[l]);
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    std::string config;
    std::string directory;
    std::uint32_t files = 4;
    std::uint64_t file_gib = 8;
    bool do_verify = false, do_overlap = false;
    std::uint32_t layers = 80, gemms = 4, rounds = 3;
    std::uint64_t entries_per_layer = 256;
    int gemm_n = 4096;
    std::string compute = "persistent";
    double kernel_us = 2500;
    std::uint64_t verify_entries = 16384;
    std::uint64_t stress_batch = 8192;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { if (i + 1 >= argc) FAIL("%s needs a value", argv[i]); return argv[++i]; };
        const char* a = argv[i];
        if (!std::strcmp(a, "--config")) config = next();
        else if (!std::strcmp(a, "--directory")) directory = next();
        else if (!std::strcmp(a, "--files")) files = std::strtoul(next(), nullptr, 10);
        else if (!std::strcmp(a, "--file-gib")) file_gib = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--verify")) do_verify = true;
        else if (!std::strcmp(a, "--overlap")) do_overlap = true;
        else if (!std::strcmp(a, "--layers")) layers = std::strtoul(next(), nullptr, 10);
        else if (!std::strcmp(a, "--entries-per-layer")) entries_per_layer = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--gemms")) gemms = std::strtoul(next(), nullptr, 10);
        else if (!std::strcmp(a, "--gemm-n")) gemm_n = std::atoi(next());
        else if (!std::strcmp(a, "--compute")) compute = next();
        else if (!std::strcmp(a, "--kernel-us")) kernel_us = std::atof(next());
        else if (!std::strcmp(a, "--rounds")) rounds = std::strtoul(next(), nullptr, 10);
        else if (!std::strcmp(a, "--verify-entries")) verify_entries = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--stress-batch")) stress_batch = std::strtoull(next(), nullptr, 10);
        else FAIL("unknown argument %s", a);
    }
    if (config.empty() || directory.empty()) FAIL("--config and --directory are required");
    if (!do_verify && !do_overlap) do_verify = do_overlap = true;

    Bench b;
    b.async_mode = std::getenv("TUTTI_ASYNC_COMPLETION") && !std::strcmp(std::getenv("TUTTI_ASYNC_COMPLETION"), "1");
    b.file_bytes = file_gib << 30;
    auto created = TuttiRuntime::create(config);
    if (!created.ok()) FAIL("runtime: %s", created.status().message().c_str());
    auto owner = std::move(created).value();
    b.rt = owner->storage_runtime();
    CUDA_OK(cudaSetDevice(b.rt->accel_id()));
    std::printf("mode=%s gpu=%d\n", b.async_mode ? "async (submit-only)" : "sync (submit-and-wait)",
                b.rt->accel_id());

    const std::string dir = directory + "/async_completion_bench";
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) FAIL("mkdir %s", dir.c_str());
    for (std::uint32_t f = 0; f < files; ++f) {
        const std::string path = dir + "/" + std::to_string(f) + ".bin";
        if (!ensure_file(path, b.file_bytes)) FAIL("create %s: %s", path.c_str(), std::strerror(errno));
        auto t = b.rt->open("file://" + path, OpenOptions{"file"});
        if (!t.ok()) FAIL("open %s: %s", path.c_str(), t.status().message().c_str());
        b.targets.push_back(t.value());
    }

    const std::uint64_t overlap_entries = (std::uint64_t)layers * entries_per_layer;
    const std::uint64_t pool_entries = std::max(2 * verify_entries, overlap_entries);
    void* pool = nullptr;
    CUDA_OK(cudaMalloc(&pool, pool_entries * kEntryBytes));
    auto m = b.rt->register_memory({pool, pool_entries * kEntryBytes, MemoryKind::DEVICE,
                                    MemoryOwnership::CALLER_OWNED, b.rt->accel_id(),
                                    TUTTI_COMPILED_ACCELERATOR_PROFILE, 4096});
    if (!m.ok()) FAIL("register_memory: %s", m.status().message().c_str());
    b.mem = m.value();

    if (do_verify) {
        auto* src = static_cast<std::uint64_t*>(pool);
        auto* dst = reinterpret_cast<std::uint64_t*>(static_cast<char*>(pool) + verify_entries * kEntryBytes);
        verify(b, src, dst, verify_entries, 256, false, "batch=256");
        verify(b, src, dst, verify_entries, stress_batch, true, "queue-overflow stress");
    }
    if (do_overlap) {
        cudaStream_t io = nullptr, compute_stream = nullptr;
        CUDA_OK(cudaStreamCreateWithFlags(&io, cudaStreamNonBlocking));
        CUDA_OK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
        Gemm g;
        if (compute == "cublas") g.init(gemm_n, compute_stream);
        else g.init_persistent(b.rt->accel_id(), kernel_us, compute_stream);
        overlap_step(b, g, io, compute_stream, layers, entries_per_layer, gemms, false, 0);  // warm up
        overlap_step(b, g, io, compute_stream, layers, entries_per_layer, gemms, true, 0);
        for (std::uint32_t r = 0; r < rounds; ++r) {
            const auto base = overlap_step(b, g, io, compute_stream, layers, entries_per_layer, gemms, false, 0);
            // Different file offsets each round: no reuse of anything cached.
            const auto io_r = overlap_step(b, g, io, compute_stream, layers, entries_per_layer, gemms, true,
                                           (r + 1) * 7919);
            std::printf("overlap round %u: layers=%u x %llu entries (%.1f MiB/layer), %u %s kernels/layer | "
                        "compute only: %.1f ms | with IO: compute %.1f ms (+%.1f%%), compute wall %.1f ms, "
                        "IO %.2f GiB/s\n",
                        r, layers, (unsigned long long)entries_per_layer,
                        entries_per_layer * kEntryBytes / 1048576.0, gemms, compute.c_str(), base.gemm_ms,
                        io_r.gemm_ms, 100.0 * (io_r.gemm_ms / base.gemm_ms - 1.0), io_r.wall_ms,
                        io_r.io_gib_s);
        }
    }
    b.rt->unregister_memory(b.mem);
    cudaFree(pool);
    return 0;
}
