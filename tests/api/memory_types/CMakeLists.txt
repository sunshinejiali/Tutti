# tests/api/memory_types/CMakeLists.txt
#
# Hardware-free: plain C++17, links only tutti_api for public headers + profile macro.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_memory_types_contract_test
    memory_types_contract_test.cpp
)

target_link_libraries(tutti_memory_types_contract_test PRIVATE
    tutti_api
)

set_target_properties(tutti_memory_types_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(NAME tutti_memory_types_contract_test
         COMMAND tutti_memory_types_contract_test)

message(STATUS "Configured tutti_memory_types_contract_test (hardware-free)")

set_property(TEST tutti_memory_types_contract_test APPEND PROPERTY LABELS module_api)
