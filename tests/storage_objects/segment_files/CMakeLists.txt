cmake_minimum_required(VERSION 3.18)
find_package(Threads REQUIRED)

add_executable(tutti_storage_object_store_segment_files_test
    storage_object_store_segment_files_test.cpp
)

target_link_libraries(tutti_storage_object_store_segment_files_test PRIVATE
    tutti_storage_objects
    Threads::Threads
)

set_target_properties(tutti_storage_object_store_segment_files_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_options(tutti_storage_object_store_segment_files_test PRIVATE
    -Wall -Wextra
)

target_compile_definitions(tutti_storage_object_store_segment_files_test PRIVATE
    TUTTI_TEST_TMPDIR_DEFAULT="${CMAKE_BINARY_DIR}"
)

add_test(NAME tutti_storage_object_store_segment_files_test
         COMMAND tutti_storage_object_store_segment_files_test)

set_property(TEST tutti_storage_object_store_segment_files_test APPEND PROPERTY LABELS module_storage_objects)
