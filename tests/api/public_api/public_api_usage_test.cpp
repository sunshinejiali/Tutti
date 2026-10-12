// tests/api/public_api/public_api_usage_test.cpp
//
// Proves that linking only tutti_api propagates all usage requirements:
//   - Public include path (tutti/cuda_like.h is found)
//   - Profile definition (TUTTI_USE_HOST or TUTTI_USE_CUDA)
//   - CUDA-like API availability (cudaGetDeviceCount is callable)
//
// No manual include directories or compile definitions are set in CMake.

#include <tutti/cuda_like.h>
#include <tutti/storage_runtime.h>

#include <cstdio>

int main() {
    // Compile-time: exactly one profile macro must be defined
    // (inherited from tutti_api -> tutti_cuda_like)
#if defined(TUTTI_USE_CUDA) && defined(TUTTI_USE_HOST)
#error "Both TUTTI_USE_CUDA and TUTTI_USE_HOST are defined"
#endif
#if !defined(TUTTI_USE_CUDA) && !defined(TUTTI_USE_HOST)
#error "Neither TUTTI_USE_CUDA nor TUTTI_USE_HOST is defined"
#endif
    // HOST profile: only TUTTI_USE_HOST is set (TUTTI_USE_CUDA is absent)
    // CUDA profile: only TUTTI_USE_CUDA is set (TUTTI_USE_HOST is absent)

    // Call a CUDA-like API to prove usage requirements are propagated.
    int deviceCount = -1;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    auto runtime = tutti::StorageRuntime::create();
    if (!runtime.ok()) {
        return 1;
    }

    if (err != cudaSuccess) {
        // CUDA profile without a GPU may fail here; the fact that the call
        // compiled and linked already proves usage requirements are correct.
        printf("PASS: tutti_api usage requirements verified "
               "(cudaGetDeviceCount err=%d, no device)\n", (int)err);
        return 0;
    }

    printf("PASS: tutti_api usage requirements verified (deviceCount=%d)\n",
           deviceCount);
    return 0;
}
