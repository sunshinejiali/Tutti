# tests/api/spi_consumer/CMakeLists.txt -- SPI usage-requirements consumer test
#
# Proves the tutti_spi INTERFACE target carries every usage requirement needed
# to consume the in-repo SPI headers, with nothing hand-injected by this test:
#   - public include root (tutti/include) -> <tutti/spi/...> resolves
#   - public include root       -> <tutti/status.h> / <tutti/io_types.h> resolve
#                                  (transitively via tutti_api)
#   - accelerator profile macro -> <tutti/cuda_like.h> selects the HOST shim
#                                  (transitively via tutti_api -> tutti_cuda_like)
#
# This file adds no include paths and no compile macros of its own; it links
# ONLY tutti_spi. That is the core build-boundary acceptance point.

cmake_minimum_required(VERSION 3.18)

add_executable(tutti_spi_consumer_test
    spi_consumer_test.cpp
)

target_link_libraries(tutti_spi_consumer_test PRIVATE
    tutti_spi
)

set_target_properties(tutti_spi_consumer_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(NAME tutti_spi_consumer_test COMMAND tutti_spi_consumer_test)

message(STATUS "Configured tutti_spi_consumer_test (SPI usage-requirements consumer test)")

set_property(TEST tutti_spi_consumer_test APPEND PROPERTY LABELS module_api)
