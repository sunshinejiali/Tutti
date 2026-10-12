# tests/device_manager/snvme_identity/CMakeLists.txt
#
# Regression for 2026-10-07: the preset must open the character device of the
# controller it was configured for (chrdev and controller numbers diverge
# when two daemons bring controllers up). Needs both md daemons up; skips (77)
# without snvme devices.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_snvme_identity_contract_test
    snvme_identity_contract_test.cpp
)

target_link_libraries(tutti_snvme_identity_contract_test PRIVATE
    tutti_presets
    tutti_runtime
    tutti_striped_local_nvme_datapath
    tutti_spi
    libnvm
    ${TUTTI_ACCEL_RUNTIME_LIBS}
)

target_include_directories(tutti_snvme_identity_contract_test PRIVATE
    ${TUTTI_REPOSITORY_ROOT}   # csrc/common/snvme_identity.h
)

set_target_properties(tutti_snvme_identity_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CUDA
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_snvme_identity_contract_test PRIVATE -Wall -Wextra)

add_test(NAME tutti_snvme_identity_contract_test
         COMMAND tutti_snvme_identity_contract_test)

set_tests_properties(tutti_snvme_identity_contract_test PROPERTIES
    LABELS "hardware;striped;raid0"
    SKIP_RETURN_CODE 77
)

set_property(TEST tutti_snvme_identity_contract_test APPEND PROPERTY LABELS module_device_manager)
