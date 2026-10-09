// Pattern fill / compare kernels for async_completion_bench.cpp.
#include <cstdint>
#include <cuda_runtime.h>

namespace {

__global__ void fill_pattern(std::uint64_t* p, std::uint64_t words, std::uint64_t seed) {
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x; i < words;
         i += (std::uint64_t)gridDim.x * blockDim.x) {
        std::uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        p[i] = x;
    }
}

__global__ void count_mismatch(const std::uint64_t* a, const std::uint64_t* b,
                               std::uint64_t words, unsigned long long* bad) {
    unsigned long long local = 0;
    for (std::uint64_t i = blockIdx.x * (std::uint64_t)blockDim.x + threadIdx.x; i < words;
         i += (std::uint64_t)gridDim.x * blockDim.x) {
        local += a[i] != b[i];
    }
    if (local) atomicAdd(bad, local);
}

} // namespace

void bench_fill_pattern(std::uint64_t* p, std::uint64_t words, std::uint64_t seed, cudaStream_t s) {
    fill_pattern<<<1024, 256, 0, s>>>(p, words, seed);
}

void bench_count_mismatch(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t words,
                          unsigned long long* bad, cudaStream_t s) {
    count_mismatch<<<1024, 256, 0, s>>>(a, b, words, bad);
}

// Stand-in for a persistent GEMM (cutlass persistent / MoE grouped GEMM):
// grid = SM count, each block reserves the largest shared memory a block
// may use, so no other block (an IO kernel's) fits beside it on its SM.
// If any SM is occupied, one block waits for a second wave and the kernel
// takes about twice as long — the effect seen in vLLM.
namespace {
__global__ void persistent_compute(float* sink, std::uint32_t iters) {
    extern __shared__ float smem[];
    float a = threadIdx.x * 1e-3f, b = blockIdx.x * 1e-3f;
    for (std::uint32_t i = 0; i < iters; ++i) {
        a = fmaf(a, 1.0000001f, b);
        b = fmaf(b, 0.9999999f, a);
    }
    smem[threadIdx.x] = a + b;
    __syncthreads();
    if (threadIdx.x == 0) sink[blockIdx.x] = smem[0];
}
} // namespace

int bench_persistent_setup(int device, int* sm_count) {
    int smem = 0;
    cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(persistent_compute, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    return smem;
}

void bench_persistent_compute(float* sink, std::uint32_t iters, int sm_count, int smem,
                              cudaStream_t s) {
    persistent_compute<<<sm_count, 256, smem, s>>>(sink, iters);
}
