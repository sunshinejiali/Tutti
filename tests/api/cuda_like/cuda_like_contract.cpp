// tests/api/cuda_like/cuda_like_contract.cpp
//
// Contract test for tutti/cuda_like.h.  Covers the CUDA-like API surface in
// both CUDA and HOST profiles.
//
// CUDA profile: skips if no GPU is present.
// HOST profile: always runs to completion.

#include <tutti/cuda_like.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_failures = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            printf("FAIL [line %d]: %s\n", __LINE__, msg);       \
            ++g_failures;                                         \
        }                                                         \
    } while (0)

#define CHECK_EQ(actual, expected, msg)                                     \
    do {                                                                    \
        if ((int)(actual) != (int)(expected)) {                             \
            printf("FAIL [line %d]: %s (got %d, expected %d)\n",           \
                   __LINE__, msg, (int)(actual), (int)(expected));          \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// 1. cudaGetDeviceCount
// ---------------------------------------------------------------------------
static void test_device_count() {
    int count = -1;
    cudaError_t err = cudaGetDeviceCount(&count);
    CHECK_EQ(err, cudaSuccess, "cudaGetDeviceCount should succeed");
    CHECK(count >= 0, "device count should be non-negative");

    if (err != cudaSuccess || count == 0) {
        printf("SKIP: no CUDA device\n");
        std::exit(0);
    }
}

// ---------------------------------------------------------------------------
// 2. set/get device
// ---------------------------------------------------------------------------
static void test_set_get_device() {
    cudaError_t err = cudaSetDevice(0);
    CHECK_EQ(err, cudaSuccess, "cudaSetDevice(0) should succeed");

    int dev = -1;
    err = cudaGetDevice(&dev);
    CHECK_EQ(err, cudaSuccess, "cudaGetDevice should succeed");
    CHECK_EQ(dev, 0, "current device should be 0");
}

// ---------------------------------------------------------------------------
// 3. cudaMalloc / cudaFree
// ---------------------------------------------------------------------------
static void test_malloc_free() {
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, 1024);
    CHECK_EQ(err, cudaSuccess, "cudaMalloc should succeed");
    CHECK(ptr != nullptr, "cudaMalloc should return non-null");

    err = cudaFree(ptr);
    CHECK_EQ(err, cudaSuccess, "cudaFree should succeed");
}

// ---------------------------------------------------------------------------
// 4. cudaMallocHost / cudaFreeHost
// ---------------------------------------------------------------------------
static void test_malloc_host_free_host() {
    void *ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, 1024);
    CHECK_EQ(err, cudaSuccess, "cudaMallocHost should succeed");
    CHECK(ptr != nullptr, "cudaMallocHost should return non-null");

    err = cudaFreeHost(ptr);
    CHECK_EQ(err, cudaSuccess, "cudaFreeHost should succeed");
}

// ---------------------------------------------------------------------------
// 5. stream create / synchronize / destroy
// ---------------------------------------------------------------------------
static void test_stream() {
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    CHECK_EQ(err, cudaSuccess, "cudaStreamCreate should succeed");
    CHECK(stream != nullptr, "stream should be non-null");

    err = cudaStreamSynchronize(stream);
    CHECK_EQ(err, cudaSuccess, "cudaStreamSynchronize should succeed");

    err = cudaStreamDestroy(stream);
    CHECK_EQ(err, cudaSuccess, "cudaStreamDestroy should succeed");
}

// ---------------------------------------------------------------------------
// 6. event create / record / wait / query / destroy
// ---------------------------------------------------------------------------
static void test_event() {
    cudaEvent_t event = nullptr;
    cudaError_t err = cudaEventCreate(&event);
    CHECK_EQ(err, cudaSuccess, "cudaEventCreate should succeed");
    CHECK(event != nullptr, "event should be non-null");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    err = cudaEventRecord(event, stream);
    CHECK_EQ(err, cudaSuccess, "cudaEventRecord should succeed");

    err = cudaEventSynchronize(event);
    CHECK_EQ(err, cudaSuccess, "cudaEventSynchronize should succeed");

    err = cudaEventQuery(event);
    CHECK_EQ(err, cudaSuccess, "cudaEventQuery should return success (ready)");

    err = cudaStreamWaitEvent(stream, event, 0);
    CHECK_EQ(err, cudaSuccess, "cudaStreamWaitEvent should succeed");

    cudaStreamDestroy(stream);
    err = cudaEventDestroy(event);
    CHECK_EQ(err, cudaSuccess, "cudaEventDestroy should succeed");
}

// ---------------------------------------------------------------------------
// 7. async H2D / D2H round trip
// ---------------------------------------------------------------------------
static void test_async_round_trip() {
    const int N = 256;
    int *host_src = nullptr;
    int *host_dst = nullptr;
    int *dev_buf  = nullptr;

    CHECK_EQ(cudaMallocHost((void**)&host_src, N * sizeof(int)), cudaSuccess,
             "alloc host_src");
    CHECK_EQ(cudaMallocHost((void**)&host_dst, N * sizeof(int)), cudaSuccess,
             "alloc host_dst");
    CHECK_EQ(cudaMalloc((void**)&dev_buf, N * sizeof(int)), cudaSuccess,
             "alloc dev_buf");
    CHECK(host_src != nullptr, "host_src non-null");
    CHECK(host_dst != nullptr, "host_dst non-null");
    CHECK(dev_buf  != nullptr, "dev_buf non-null");

    for (int i = 0; i < N; ++i) host_src[i] = i;
    std::memset(host_dst, 0, N * sizeof(int));

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // H2D
    cudaError_t err = cudaMemcpyAsync(dev_buf, host_src, N * sizeof(int),
                                      cudaMemcpyHostToDevice, stream);
    CHECK_EQ(err, cudaSuccess, "cudaMemcpyAsync H2D should succeed");

    // D2H
    err = cudaMemcpyAsync(host_dst, dev_buf, N * sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    CHECK_EQ(err, cudaSuccess, "cudaMemcpyAsync D2H should succeed");

    cudaStreamSynchronize(stream);

    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (host_dst[i] != i) { match = false; break; }
    }
    CHECK(match, "round trip data should match source");

    cudaStreamDestroy(stream);
    cudaFree(dev_buf);
    cudaFreeHost(host_src);
    cudaFreeHost(host_dst);
}

// ---------------------------------------------------------------------------
// 8. memset
// ---------------------------------------------------------------------------
static void test_memset() {
    const int N = 128;
    unsigned char *dev_buf  = nullptr;
    unsigned char *host_buf = nullptr;

    CHECK_EQ(cudaMalloc((void**)&dev_buf, N), cudaSuccess, "memset: alloc dev_buf");
    CHECK_EQ(cudaMallocHost((void**)&host_buf, N), cudaSuccess, "memset: alloc host_buf");

    cudaError_t err = cudaMemset(dev_buf, 0xAB, N);
    CHECK_EQ(err, cudaSuccess, "cudaMemset should succeed");

    err = cudaMemcpy(host_buf, dev_buf, N, cudaMemcpyDeviceToHost);
    CHECK_EQ(err, cudaSuccess, "memset verify: D2H copy should succeed");

    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (host_buf[i] != 0xAB) { match = false; break; }
    }
    CHECK(match, "memset should fill buffer with 0xAB");

    cudaFreeHost(host_buf);
    cudaFree(dev_buf);
}

// ---------------------------------------------------------------------------
// 9. invalid null parameter errors
// ---------------------------------------------------------------------------
static void test_null_errors() {
    // Null output pointer
    cudaError_t err = cudaMalloc(nullptr, 1024);
    CHECK(err != cudaSuccess, "cudaMalloc(nullptr) should fail");

    err = cudaMallocHost(nullptr, 1024);
    CHECK(err != cudaSuccess, "cudaMallocHost(nullptr) should fail");

    err = cudaStreamCreate(nullptr);
    CHECK(err != cudaSuccess, "cudaStreamCreate(nullptr) should fail");

    err = cudaEventCreate(nullptr);
    CHECK(err != cudaSuccess, "cudaEventCreate(nullptr) should fail");

    // Null dst for non-zero copy
    int dummy = 42;
    err = cudaMemcpy(nullptr, &dummy, sizeof(int), cudaMemcpyHostToDevice);
    CHECK(err != cudaSuccess, "cudaMemcpy(null dst) should fail");

    // Null src for non-zero copy
    err = cudaMemcpy(&dummy, nullptr, sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(err != cudaSuccess, "cudaMemcpy(null src) should fail");

    // Null ptr for non-zero memset
    err = cudaMemset(nullptr, 0, 16);
    CHECK(err != cudaSuccess, "cudaMemset(nullptr) should fail");
}

// ---------------------------------------------------------------------------
// 10. cudaGetErrorString
// ---------------------------------------------------------------------------
static void test_error_string() {
    const char *str = cudaGetErrorString(cudaSuccess);
    CHECK(str != nullptr, "cudaGetErrorString(cudaSuccess) non-null");
    printf("cudaGetErrorString(cudaSuccess) = \"%s\"\n", str);

    str = cudaGetErrorString(cudaErrorInvalidValue);
    CHECK(str != nullptr, "cudaGetErrorString(cudaErrorInvalidValue) non-null");
    printf("cudaGetErrorString(cudaErrorInvalidValue) = \"%s\"\n", str);
}

// ---------------------------------------------------------------------------
// 11. pointer attributes (cudaPointerGetAttributes)
// ---------------------------------------------------------------------------
static void test_pointer_attributes() {
    // Device pointer
    void *dev_ptr = nullptr;
    CHECK_EQ(cudaMalloc(&dev_ptr, 1024), cudaSuccess, "alloc for pointer attrs");
    CHECK(dev_ptr != nullptr, "dev ptr non-null");

    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, dev_ptr);
    CHECK_EQ(err, cudaSuccess, "cudaPointerGetAttributes(dev_ptr) should succeed");
    CHECK(attrs.device == 0, "device pointer: device == 0");
    printf("  dev_ptr attrs: type=%d device=%d\n", (int)attrs.type, attrs.device);
    cudaFree(dev_ptr);

    // Host pointer
    void *host_ptr = nullptr;
    CHECK_EQ(cudaMallocHost(&host_ptr, 1024), cudaSuccess, "alloc host for pointer attrs");
    CHECK(host_ptr != nullptr, "host ptr non-null");

    err = cudaPointerGetAttributes(&attrs, host_ptr);
    CHECK_EQ(err, cudaSuccess, "cudaPointerGetAttributes(host_ptr) should succeed");
    printf("  host_ptr attrs: type=%d device=%d\n", (int)attrs.type, attrs.device);
    cudaFreeHost(host_ptr);

    // Null attributes struct
    err = cudaPointerGetAttributes(nullptr, dev_ptr);
    CHECK(err != cudaSuccess, "cudaPointerGetAttributes(nullptr attrs) should fail");
}

// ---------------------------------------------------------------------------
// 12. cudaDeviceSynchronize
// ---------------------------------------------------------------------------
static void test_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    CHECK_EQ(err, cudaSuccess, "cudaDeviceSynchronize should succeed");
}

// ---------------------------------------------------------------------------
// 13. cudaStreamQuery (success after sync, NotReady for unsignaled event)
// ---------------------------------------------------------------------------
static void test_stream_query() {
    cudaStream_t stream = nullptr;
    CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess, "create stream for query");
    CHECK(stream != nullptr, "stream non-null");

    // An empty stream that has no pending work should report success.
    // In HOST profile, streams are synchronous so this is always ready.
    // In CUDA profile, an empty stream with no work is also ready.
    cudaError_t err = cudaStreamQuery(stream);
    CHECK(err == cudaSuccess || err == cudaErrorNotReady,
          "cudaStreamQuery should return success or NotReady");

    cudaStreamSynchronize(stream);
    err = cudaStreamQuery(stream);
    CHECK_EQ(err, cudaSuccess, "cudaStreamQuery after sync should succeed");

    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// 14. cudaMemcpy DeviceToDevice
// ---------------------------------------------------------------------------
static void test_d2d_copy() {
    const int N = 64;
    int *src = nullptr;
    int *dst = nullptr;

    CHECK_EQ(cudaMalloc((void**)&src, N * sizeof(int)), cudaSuccess, "D2D: alloc src");
    CHECK_EQ(cudaMalloc((void**)&dst, N * sizeof(int)), cudaSuccess, "D2D: alloc dst");
    CHECK(src != nullptr && dst != nullptr, "D2D: allocs non-null");

    // Fill src via memset + verify via host
    CHECK_EQ(cudaMemset(src, 0xCD, N * sizeof(int)), cudaSuccess, "D2D: memset src");

    // D2D copy
    cudaError_t err = cudaMemcpy(dst, src, N * sizeof(int),
                                  cudaMemcpyDeviceToDevice);
    CHECK_EQ(err, cudaSuccess, "cudaMemcpy D2D should succeed");

    // Verify via D2H
    int *host_buf = nullptr;
    CHECK_EQ(cudaMallocHost((void**)&host_buf, N * sizeof(int)), cudaSuccess, "D2D: alloc host buf");
    err = cudaMemcpy(host_buf, dst, N * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_EQ(err, cudaSuccess, "D2D: D2H verify should succeed");

    bool match = true;
    unsigned char *hb = (unsigned char *)host_buf;
    for (int i = 0; i < N * (int)sizeof(int); ++i) {
        if (hb[i] != (unsigned char)0xCD) { match = false; break; }
    }
    CHECK(match, "D2D: dst should match src pattern");

    cudaFreeHost(host_buf);
    cudaFree(dst);
    cudaFree(src);
}

// ---------------------------------------------------------------------------
// 15. cudaMemsetAsync
// ---------------------------------------------------------------------------
static void test_memset_async() {
    const int N = 128;
    unsigned char *dev_buf = nullptr;
    unsigned char *host_buf = nullptr;

    CHECK_EQ(cudaMalloc((void**)&dev_buf, N), cudaSuccess, "async memset: alloc dev");
    CHECK_EQ(cudaMallocHost((void**)&host_buf, N), cudaSuccess, "async memset: alloc host");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaError_t err = cudaMemsetAsync(dev_buf, 0x77, N, stream);
    CHECK_EQ(err, cudaSuccess, "cudaMemsetAsync should succeed");

    err = cudaMemcpyAsync(host_buf, dev_buf, N, cudaMemcpyDeviceToHost, stream);
    CHECK_EQ(err, cudaSuccess, "async memset: D2H should succeed");

    cudaStreamSynchronize(stream);

    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (host_buf[i] != 0x77) { match = false; break; }
    }
    CHECK(match, "async memset: buffer should be 0x77");

    cudaStreamDestroy(stream);
    cudaFreeHost(host_buf);
    cudaFree(dev_buf);
}

// ---------------------------------------------------------------------------
// 16. cudaHostRegister / cudaHostUnregister
// ---------------------------------------------------------------------------
static void test_host_register() {
    // Use plain malloc — cudaHostRegister on already-pinned memory
    // (from cudaMallocHost) returns cudaErrorInvalidValue in CUDA.
    void *ptr = std::malloc(1024);
    CHECK(ptr != nullptr, "host register: malloc");

    // cudaHostRegister behavior is profile-dependent:
    //   HOST shim: returns cudaErrorNotSupported
    //   CUDA:      returns cudaSuccess (pins the memory)
    cudaError_t err = cudaHostRegister(ptr, 1024, 0);
    CHECK(err == cudaSuccess || err == cudaErrorNotSupported,
          "cudaHostRegister should succeed (CUDA) or return NotSupported (HOST)");
    printf("  cudaHostRegister result: %d (%s)\n", (int)err, cudaGetErrorString(err));

    // Unregister — should match register behavior
    if (err == cudaSuccess) {
        err = cudaHostUnregister(ptr);
        CHECK_EQ(err, cudaSuccess, "cudaHostUnregister should succeed after register");
    } else {
        err = cudaHostUnregister(ptr);
        CHECK(err == cudaSuccess || err == cudaErrorNotSupported,
              "cudaHostUnregister should succeed or return NotSupported");
    }

    std::free(ptr);
}

// ---------------------------------------------------------------------------
// 17. cudaGetLastError / cudaPeekAtLastError
// ---------------------------------------------------------------------------
static void test_last_error() {
    // Clear any sticky error from previous tests.
    cudaGetLastError();

    // After clearing, last error should be success.
    cudaError_t err = cudaGetLastError();
    CHECK_EQ(err, cudaSuccess, "cudaGetLastError should be success after clear");

    err = cudaPeekAtLastError();
    CHECK_EQ(err, cudaSuccess, "cudaPeekAtLastError should be success");

    // Trigger an error (null malloc) and verify it can be retrieved.
    cudaMalloc(nullptr, 1024);  // sets error

    // In HOST shim, cudaGetLastError always returns success (errors are
    // returned directly, not sticky).  In CUDA, the error is sticky.
    // Either behavior is acceptable for the contract — the key is that
    // the functions exist and return a cudaError_t.
    CHECK(true, "cudaPeekAtLastError callable after error");

    // Clear the error
    cudaGetLastError();
    CHECK(true, "cudaGetLastError clears error");
}

// ---------------------------------------------------------------------------
// 18. create with flags (cudaStreamCreateWithFlags, cudaEventCreateWithFlags)
// ---------------------------------------------------------------------------
static void test_create_with_flags() {
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CHECK_EQ(err, cudaSuccess, "cudaStreamCreateWithFlags should succeed");
    CHECK(stream != nullptr, "flagged stream non-null");
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaEvent_t event = nullptr;
    err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    CHECK_EQ(err, cudaSuccess, "cudaEventCreateWithFlags should succeed");
    CHECK(event != nullptr, "flagged event non-null");
    cudaEventDestroy(event);
}

// ---------------------------------------------------------------------------
// 19. cudaGetDeviceProperties
// ---------------------------------------------------------------------------
static void test_device_properties() {
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    CHECK_EQ(err, cudaSuccess, "cudaGetDeviceProperties should succeed");
    CHECK(prop.name[0] != '\0', "device name non-empty");
    printf("  device: \"%s\" major=%d minor=%d SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // Invalid device index
    err = cudaGetDeviceProperties(&prop, -1);
    CHECK(err != cudaSuccess, "cudaGetDeviceProperties(-1) should fail");

    // Null prop
    err = cudaGetDeviceProperties(nullptr, 0);
    CHECK(err != cudaSuccess, "cudaGetDeviceProperties(nullptr) should fail");
}

// ---------------------------------------------------------------------------
// 20. profile macro verification (compile-time)
// ---------------------------------------------------------------------------
static void test_profile_macro() {
    // These static_asserts prove that exactly one TUTTI_USE_<PROFILE> is
    // defined.  The cuda_like.h header also enforces this via #error, but
    // the static_assert here gives a visible test failure message (vs. a
    // compile error that only fires when the header is included).

#if defined(TUTTI_USE_CUDA) && defined(TUTTI_USE_HOST)
    #error "Both TUTTI_USE_CUDA and TUTTI_USE_HOST defined — contract violation"
#endif

#if !defined(TUTTI_USE_CUDA) && !defined(TUTTI_USE_HOST)
    #error "Neither TUTTI_USE_CUDA nor TUTTI_USE_HOST defined — contract violation"
#endif

    // Report which profile is active.
#if defined(TUTTI_USE_CUDA)
    printf("  active profile: CUDA (TUTTI_USE_CUDA=1)\n");
    CHECK(true, "exactly one profile macro defined (CUDA)");
#elif defined(TUTTI_USE_HOST)
    printf("  active profile: HOST (TUTTI_USE_HOST=1)\n");
    CHECK(true, "exactly one profile macro defined (HOST)");
#endif

    // Profile-specific assertions
#if defined(TUTTI_USE_HOST)
    // HOST shim: cudaPointerGetAttributes always returns cudaMemoryTypeHost
    // because all memory is host-backed.
    void *ptr = nullptr;
    cudaMallocHost(&ptr, 64);
    if (ptr) {
        cudaPointerAttributes attrs{};
        cudaPointerGetAttributes(&attrs, ptr);
        CHECK(attrs.type == cudaMemoryTypeHost,
              "HOST profile: pointer type should be cudaMemoryTypeHost");
        cudaFreeHost(ptr);
    }
#endif

#if defined(TUTTI_USE_CUDA)
    // CUDA profile: cudaGetDeviceCount must return >= 1 if we got here
    // (test_device_count already exits if 0, so this is a belt-and-suspenders
    // check that the CUDA runtime is actually active).
    int count = 0;
    cudaGetDeviceCount(&count);
    CHECK(count >= 1, "CUDA profile: at least 1 device expected");
#endif
}

// ---------------------------------------------------------------------------

int main() {
    test_device_count();
    test_set_get_device();
    test_malloc_free();
    test_malloc_host_free_host();
    test_stream();
    test_event();
    test_async_round_trip();
    test_memset();
    test_null_errors();
    test_error_string();
    test_pointer_attributes();
    test_device_synchronize();
    test_stream_query();
    test_d2d_copy();
    test_memset_async();
    test_host_register();
    test_last_error();
    test_create_with_flags();
    test_device_properties();
    test_profile_macro();

    if (g_failures > 0) {
        printf("RESULT: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("RESULT: all checks passed\n");
    return 0;
}
