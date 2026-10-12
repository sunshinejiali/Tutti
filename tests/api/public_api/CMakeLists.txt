# tests/api/public_api/CMakeLists.txt -- public API usage test
#
# Proves that linking only tutti_api is sufficient to obtain public headers,
# profile definition, and CUDA-like API availability — no manual include
# directories, compile definitions, or vendor library links required.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_public_api_usage_test
    public_api_usage_test.cpp
)

target_link_libraries(tutti_public_api_usage_test PRIVATE
    tutti_api
)

set_target_properties(tutti_public_api_usage_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(NAME tutti_public_api_usage_test COMMAND tutti_public_api_usage_test)

message(STATUS "Configured tutti_public_api_usage_test (public API usage test)")

set_property(TEST tutti_public_api_usage_test APPEND PROPERTY LABELS module_api)
