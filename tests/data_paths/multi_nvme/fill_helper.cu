// fill_helper.cu — GPU kernel for DMA-visible buffer fills
#include <cstdint>
#include <tutti/cuda_like.h>

__global__ void fill_pattern_kernel(unsigned char* buf, unsigned char val,
                                    std::uint64_t n) {
    std::uint64_t stride = (std::uint64_t)gridDim.x * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        buf[i] = val;
    }
    __threadfence_system();
}

// Position-dependent pattern: byte at logical offset `base_offset + i`
// gets value (base_offset + i) mod 251 (a prime, avoids trivial repeats).
// Used by the roundtrip test to detect any offset scrambling across shards.
__global__ void fill_position_pattern_kernel(unsigned char* buf,
                                             std::uint64_t base_offset,
                                             std::uint64_t n) {
    std::uint64_t stride = (std::uint64_t)gridDim.x * blockDim.x;
    for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        buf[i] = static_cast<unsigned char>((base_offset + i) % 251u);
    }
    __threadfence_system();
}

extern "C" void launch_fill_pattern_gpu(void* buf, unsigned char val,
                                        std::uint64_t n, void* stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    fill_pattern_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (unsigned char*)buf, val, (std::uint64_t)n);
}

extern "C" void launch_fill_position_pattern_gpu(void* buf,
                                                  std::uint64_t base_offset,
                                                  std::uint64_t n, void* stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    fill_position_pattern_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (unsigned char*)buf, base_offset, (std::uint64_t)n);
}
