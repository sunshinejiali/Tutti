# tests/storage_runtime/raid0/CMakeLists.txt
#
# md RAID0 hardware contract: GPU member-direct IO vs the kernel md/ext4 path.
# Requires tutti_daemon with raid0_arrays and the array mounted; skips (77)
# otherwise.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_raid0_local_nvme_contract_test
    raid0_local_nvme_contract_test.cpp
)

target_link_libraries(tutti_raid0_local_nvme_contract_test PRIVATE
    tutti_runtime
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

target_include_directories(tutti_raid0_local_nvme_contract_test PRIVATE
    ${TUTTI_REPOSITORY_ROOT}   # csrc/common/md_raid0.h
)

target_compile_definitions(tutti_raid0_local_nvme_contract_test PRIVATE
    TUTTI_RAID0_TEST_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/tutti_raid0_test.yaml"
)

set_target_properties(tutti_raid0_local_nvme_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_raid0_local_nvme_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_raid0_local_nvme_contract_test
         COMMAND tutti_raid0_local_nvme_contract_test)

set_tests_properties(tutti_raid0_local_nvme_contract_test PROPERTIES
    LABELS "hardware;striped;raid0;runtime_e2e"
    SKIP_RETURN_CODE 77
)

set_property(TEST tutti_raid0_local_nvme_contract_test APPEND PROPERTY LABELS module_storage_runtime)
