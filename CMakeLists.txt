cmake_minimum_required(VERSION 3.17)
project(einsum_taco)
set(CMAKE_C_STANDARD 11)
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
SET(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath /usr/local/Cellar/julia/1.8.2/lib -Wl,-rpath /usr/local/Cellar/julia/1.8.2/lib/julia -Wl,-rpath /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/lib/darwin -Wl,-rpath /Users/adadima/mit/commit/einsum_taco/cmake-build-debug/test -Wl,-rpath /Users/adadima/mit/commit/einsum_taco/cmake-build-debug -Wl,-rpath /usr/local/opt/curl/lib/ -Wl,-rpath /usr/local/opt/mbedtls@2/lib/ -Wl,-rpath /usr/local/opt/openblas/lib/ -Wl,-rpath /usr/local/opt/gcc/lib/gcc/current/ -Wl,-rpath /usr/local/lib/ -Wl,-rpath /usr/lib/")
#set(CMAKE_BUILD_RPATH "/usr/local/Cellar/julia/1.8.2/lib/julia${CMAKE_BUILD_RPATH}")

file(GLOB EINSUM_SOURCES ${EINSUM_SRC_DIR}/*/*.cpp ${EINSUM_SRC_DIR}/*/*.c)
if (EINSUM_ASSERTS AND EINSUM_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
    add_definitions(-DEINSUM_ASSERTS)
endif()

option(GIT_SUBMODULE "Check submodules during build" ON)

include(FetchContent)

if (EINSUM_SHARED_LIBRARY)
    add_library(einsum_lib SHARED ${EINSUM_SOURCES})
else ()
    add_library(einsum_lib STATIC ${EINSUM_SOURCES})
endif()
add_compile_definitions(einsum_lib PUBLIC INCLUDE_GSTACO_RUNTIME="${EINSUM_INCLUDE_DIR}/einsum_taco/gstrt/")
add_compile_definitions(einsum_lib PUBLIC SRC_GSTACO_RUNTIME="${EINSUM_SRC_DIR}/gstrt/")

#FetchContent_Declare(build_finch GIT_REPOSITORY https://github.com/adadima/Finch.jl.git GIT_TAG 73188971d8107dfe0d8da31b20d3c52e0e8a3fb3)
#FetchContent_Populate(build_finch)
set(FINCH_DIR           /Users/adadima/mit/commit/Finch.jl/embed)
add_custom_target(build_finch
        COMMAND make clean
        COMMAND make
        WORKING_DIRECTORY ${FINCH_DIR}
        SOURCES ${FINCH_DIR}/finch.c ${FINCH_DIR}/finch.h
        )
add_library(finch STATIC IMPORTED)
set_target_properties(finch
        PROPERTIES IMPORTED_LOCATION ${FINCH_DIR}/libfinch.a
        INTERFACE_INCLUDE_DIRECTORIES ${FINCH_DIR}
        LINKER_LANGUAGE C
        )
add_dependencies(finch build_finch)

target_include_directories(einsum_lib PUBLIC ${EINSUM_INCLUDE_DIR})

add_library(julia SHARED IMPORTED)
set(JULIA_LIB   "/usr/local/Cellar/julia/1.8.2/lib")
set(JULIA_INCLUDE "/usr/local/Cellar/julia/1.8.2/include/julia")
set_target_properties(julia PROPERTIES
        IMPORTED_LOCATION ${JULIA_LIB}/libjulia.1.8.dylib
        INTERFACE_INCLUDE_DIRECTORIES ${JULIA_INCLUDE}
        )
target_link_libraries(einsum_lib PUBLIC finch julia)
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

add_executable(page_rank "${EINSUM_TEST_DIR}/tmp/codegen/pagerank.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_pagerank.cpp")
add_executable(bfs "${EINSUM_TEST_DIR}/tmp/codegen/bfs.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_bfs.cpp")
add_executable(sssp "${EINSUM_TEST_DIR}/tmp/codegen/sssp.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_sssp.cpp")
add_executable(bc "${EINSUM_TEST_DIR}/tmp/codegen/bc.cpp" "${EINSUM_TEST_DIR}/data/codegen/drivers/driver_bc.cpp")
