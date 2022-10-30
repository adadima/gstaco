cmake_minimum_required(VERSION 3.17)
project(einsum_taco)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(EINSUM_SHARED_LIBRARY "Build as a shared library" ON)
option(EINSUM_ASSERTS "Build with asserts" ON)
option(EINSUM_SANITIZERS "Build with sanitizers" ON)

set(EINSUM_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(EINSUM_SRC_DIR     ${EINSUM_PROJECT_DIR}/src)
set(EINSUM_TEST_DIR    ${EINSUM_PROJECT_DIR}/test)
set(EINSUM_PARSER_DIR  ${EINSUM_SRC_DIR}/parser)
set(EINSUM_INCLUDE_DIR ${EINSUM_PROJECT_DIR}/include)

file(GLOB EINSUM_SOURCES ${EINSUM_SRC_DIR}/*/*.cpp)
if (EINSUM_ASSERTS AND EINSUM_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
    add_definitions(-DEINSUM_ASSERTS)
endif()

option(GIT_SUBMODULE "Check submodules during build" ON)

include(FetchContent)
FetchContent_Declare(taco GIT_REPOSITORY https://github.com/tensor-compiler/taco.git)
FetchContent_Populate(taco)
add_subdirectory(${taco_SOURCE_DIR} ${taco_BINARY_DIR} EXCLUDE_FROM_ALL)
FetchContent_GetProperties(taco SOURCE_DIR taco_SOURCE_DIR)
set_target_properties(taco taco-test PROPERTIES CXX_STANDARD 14)

if (EINSUM_SHARED_LIBRARY)
    add_library(einsum_lib SHARED ${EINSUM_SOURCES})
else ()
    add_library(einsum_lib STATIC ${EINSUM_SOURCES})
endif()
target_link_libraries(einsum_lib PUBLIC taco)
add_compile_definitions(einsum_lib PUBLIC INCLUDE_GSTACO_RUNTIME="${EINSUM_INCLUDE_DIR}/einsum_taco/gstrt/")
add_compile_definitions(einsum_lib PUBLIC SRC_GSTACO_RUNTIME="${EINSUM_SRC_DIR}/gstrt/")
target_include_directories(einsum_lib PUBLIC ${EINSUM_INCLUDE_DIR} PUBLIC ${taco_SOURCE_DIR}/include)
install(TARGETS einsum_lib DESTINATION lib)

install(DIRECTORY ${EINSUM_INCLUDE_DIR}/ DESTINATION include)

add_subdirectory(test)

find_package(BISON REQUIRED 3.8.2)
find_package(FLEX REQUIRED)

bison_target(parser "${EINSUM_PARSER_DIR}/parser.y"  "${CMAKE_CURRENT_BINARY_DIR}/main.cc"
        DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/tok.h
        COMPILE_FLAGS "--yacc --define=api.pure=full --define=api.push-pull=push -v --define=parse.error=verbose"
        )
flex_target(lexer "${EINSUM_PARSER_DIR}/parser.lex" "${CMAKE_CURRENT_BINARY_DIR}/F.cc")

add_flex_bison_dependency(lexer parser)

add_library(parser_lib STATIC
        ${EINSUM_PARSER_DIR}
        "${CMAKE_CURRENT_BINARY_DIR}/main.cc"
        "${CMAKE_CURRENT_BINARY_DIR}/F.cc"
        )
set_target_properties(parser_lib PROPERTIES CXX_STANDARD 14)
target_include_directories(parser_lib PUBLIC ${EINSUM_INCLUDE_DIR})
target_link_libraries(parser_lib PUBLIC einsum_lib)

add_executable(parser_exec "${EINSUM_PARSER_DIR}/main.cc")
target_link_libraries(parser_exec parser_lib)

#add_executable(page_rank "${EINSUM_TEST_DIR}/tmp/codegen/pagerank.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_pagerank.cpp")
#add_executable(bfs "${EINSUM_TEST_DIR}/tmp/codegen/bfs.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_bfs.cpp")
#add_executable(sssp "${EINSUM_TEST_DIR}/tmp/codegen/sssp.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_sssp.cpp")
add_executable(bc "${EINSUM_TEST_DIR}/tmp/codegen/bc.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_bc.cpp")

add_executable(bfs_taco "${EINSUM_TEST_DIR}/apps/bfs_taco.cpp")
set_target_properties(bfs_taco PROPERTIES CXX_STANDARD 14)
target_link_libraries(bfs_taco PUBLIC einsum_lib)
add_compile_definitions(bfs_taco PUBLIC EINSUM_TACO_TEST_DATADIR="${EINSUM_PROJECT_DIR}/test/data/")

add_executable(pr_taco "${EINSUM_TEST_DIR}/apps/pr_taco.cpp")
set_target_properties(pr_taco PROPERTIES CXX_STANDARD 14)
target_link_libraries(pr_taco PUBLIC einsum_lib)
add_compile_definitions(pr_taco PUBLIC EINSUM_TACO_TEST_DATADIR="${EINSUM_PROJECT_DIR}/test/data/")

add_executable(sssp_taco "${EINSUM_TEST_DIR}/apps/sssp_taco.cpp")
set_target_properties(sssp_taco PROPERTIES CXX_STANDARD 14)
target_link_libraries(sssp_taco PUBLIC einsum_lib)
add_compile_definitions(sssp_taco PUBLIC EINSUM_TACO_TEST_DATADIR="${EINSUM_PROJECT_DIR}/test/data/")

add_executable(bc_taco "${EINSUM_TEST_DIR}/apps/bc_taco.cpp")
set_target_properties(bc_taco PROPERTIES CXX_STANDARD 14)
target_link_libraries(bc_taco PUBLIC einsum_lib)
add_compile_definitions(bc_taco PUBLIC EINSUM_TACO_TEST_DATADIR="${EINSUM_PROJECT_DIR}/test/data/")

add_executable(cc_taco "${EINSUM_TEST_DIR}/apps/cc_taco.cpp")
set_target_properties(cc_taco PROPERTIES CXX_STANDARD 14)
target_link_libraries(cc_taco PUBLIC einsum_lib)
add_compile_definitions(cc_taco PUBLIC EINSUM_TACO_TEST_DATADIR="${EINSUM_PROJECT_DIR}/test/data/")