enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}  --output-on-failure)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")


include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_library(test_interface STATIC config.cpp)
target_link_libraries(test_interface PUBLIC gtest gtest_main parser_lib einsum_lib)
add_compile_definitions(test_interface PUBLIC EINSUM_TACO_TEST_DATADIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")
add_compile_definitions(test_interface PUBLIC EINSUM_TACO_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
add_compile_definitions(test_interface PUBLIC TEST_CXX_COMPILER="${CMAKE_CXX_COMPILER}")
add_compile_definitions(test_interface PUBLIC FINCH_DIR="${FINCH_DIR}")
add_compile_definitions(test_interface PUBLIC JULIA_INCLUDE_DIR="${JULIA_INCLUDE}")
add_compile_definitions(test_interface PUBLIC JULIA_LIB_DIR="${JULIA_LIB}")

function(add_einsum_test test_dir test_name)
    add_executable(${test_name} ${test_dir}/${test_name}.cpp)
    target_link_libraries(${test_name} PUBLIC test_interface)
    add_dependencies(check ${test_name})
#    gtest_discover_tests(${test_name})
    add_test(${test_name} ${test_name})
endfunction()

add_einsum_test(ir dump_test)
add_einsum_test(ir ir_rewriter_test)
add_einsum_test(parser parse_test)
add_einsum_test(codegen gen_test)
add_einsum_test(gstrt tensor_test)