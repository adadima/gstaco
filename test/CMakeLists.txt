enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND}  --output-on-failure)

add_library(test_interface INTERFACE)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_include_directories(test_interface INTERFACE /usr/local/include)
    target_link_directories(test_interface INTERFACE /usr/local/lib)
endif ()
target_link_libraries(test_interface INTERFACE gtest gtest_main einsum_lib)

function(add_einsum_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} test_interface)
    add_test(${test_name} ${test_name})
    add_dependencies(check ${test_name})
endfunction()

add_einsum_test(dump_test)
add_einsum_test(gen_test)