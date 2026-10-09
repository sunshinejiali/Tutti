# tutti/examples/nvme_bw_probe/
#
# Single-drive 32 KiB read-bandwidth probe.
#
# Goal: find how to saturate ONE drive with small (default 32 KiB) IOs from a
# single GPU kernel, matching the fio libaio baseline (~6.1 GB/s @ QD128).
#
# Knobs:
#   --io-kb N      IO size in KiB (default 32)
#   --depth D      burst batch size = in-flight depth per submit (default 128)
#   --total-gb N   total bytes to read (default 32)
#   --file-gb N    backing file size = random footprint (default 64)
#   --buf-mb N     registered GPU buffer size (default 512)
#   --seed N       RNG seed for random offsets (default 7)
#   --fresh        recreate the backing file even if it exists
#   --sequential   sequential offsets instead of random
#
# The worker-pool kernel model is enabled with TUTTI_POOL_WORKERS=N (env,
# read by LocalNvmeDataPath); then --depth is the batch size handed to the
# pool and N workers bound the true in-flight depth.
#
# Manual hardware workload: needs GPU + snvme + libnvm + a running
# tutti_daemon. Not a CTest case.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_nvme_bw_probe
    nvme_bw_probe.cpp
)

add_executable(tutti_nvme_segment_bw_probe
    nvme_segment_bw_probe.cpp
)

target_link_libraries(tutti_nvme_bw_probe PRIVATE
    tutti_runtime
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

target_link_libraries(tutti_nvme_segment_bw_probe PRIVATE
    tutti_runtime
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

target_compile_definitions(tutti_nvme_bw_probe PRIVATE
    TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/tutti_nvme_bw_probe.yaml"
    TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG_STRIPED="${CMAKE_CURRENT_SOURCE_DIR}/tutti_nvme_bw_probe_striped.yaml"
)

target_compile_definitions(tutti_nvme_segment_bw_probe PRIVATE
    TUTTI_NVME_BW_PROBE_DEFAULT_CONFIG_STRIPED="${CMAKE_CURRENT_SOURCE_DIR}/tutti_nvme_bw_probe_striped.yaml"
)

set_target_properties(tutti_nvme_bw_probe PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(tutti_nvme_segment_bw_probe PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_nvme_bw_probe PRIVATE
    -Wall -Wextra
)

target_compile_options(tutti_nvme_segment_bw_probe PRIVATE
    -Wall -Wextra
)

message(STATUS "Configured tutti_nvme_bw_probe (single-drive bandwidth probe)")

# Submit/drain split (doc/design/async-nvme-completion.md): correctness and
# GEMM-overlap benchmark. Mode from TUTTI_ASYNC_COMPLETION (1 = submit-only).
find_package(CUDAToolkit REQUIRED)
add_executable(tutti_async_completion_bench
    async_completion_bench.cpp
    async_completion_bench_kernels.cu
)
target_link_libraries(tutti_async_completion_bench PRIVATE
    tutti_runtime
    CUDA::cublas
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)
set_target_properties(tutti_async_completion_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
