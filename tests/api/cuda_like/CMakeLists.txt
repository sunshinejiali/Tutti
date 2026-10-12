# tests/api/cuda_like/CMakeLists.txt -- cuda_like contract test
#
# Profile-agnostic contract test that exercises the CUDA-like API surface
# through tutti/cuda_like.h.  Runs in both CUDA and HOST profiles.

cmake_minimum_required(VERSION 3.18)

add_executable(cuda_like_contract_test
    cuda_like_contract.cpp
)

target_link_libraries(cuda_like_contract_test PRIVATE
    tutti_cuda_like
)

set_target_properties(cuda_like_contract_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(NAME cuda_like_contract_test COMMAND cuda_like_contract_test)

message(STATUS "Configured cuda_like_contract_test (profile-agnostic contract test)")

set_property(TEST cuda_like_contract_test APPEND PROPERTY LABELS module_api)
