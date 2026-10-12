# tests/api/io_types/CMakeLists.txt
#
# Hardware-free: plain C++17, links only tutti_api for public headers + profile macro.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_io_types_contract_test
    io_types_contract_test.cpp
)

target_link_libraries(tutti_io_types_contract_test PRIVATE
    tutti_api
)

set_target_properties(tutti_io_types_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_io_types_contract_test PRIVATE
    -Wall -Wextra
)

add_test(NAME tutti_io_types_contract_test
         COMMAND tutti_io_types_contract_test)

set_property(TEST tutti_io_types_contract_test APPEND PROPERTY LABELS module_api)
