cmake_minimum_required(VERSION 3.18)

add_executable(tutti_async_completion_contract_test
    async_completion_contract_test.cpp
    async_completion_test_kernels.cu
)
target_include_directories(tutti_async_completion_contract_test PRIVATE ${TUTTI_REPOSITORY_ROOT})
target_link_libraries(tutti_async_completion_contract_test PRIVATE
    tutti_striped_local_nvme_datapath
    tutti_spi
    libnvm
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)
set_target_properties(tutti_async_completion_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_test(NAME tutti_async_completion_contract_test COMMAND tutti_async_completion_contract_test)
set_tests_properties(tutti_async_completion_contract_test PROPERTIES
    LABELS "hardware;gpu_only;async_completion"
    SKIP_RETURN_CODE 77
    TIMEOUT 30
)

set_property(TEST tutti_async_completion_contract_test APPEND PROPERTY LABELS module_data_paths)
