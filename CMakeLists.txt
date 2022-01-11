cmake_minimum_required(VERSION 3.17)
project(einsum_taco)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(EINSUM_SHARED_LIBRARY "Build as a shared library" ON)
option(EINSUM_ASSERTS "Build with asserts" ON)

set(EINSUM_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(EINSUM_SRC_DIR     ${EINSUM_PROJECT_DIR}/src)
set(EINSUM_TEST_DIR    ${EINSUM_PROJECT_DIR}/test)
set(EINSUM_PARSER_DIR  ${EINSUM_SRC_DIR}/parser)
set(EINSUM_INCLUDE_DIR ${EINSUM_PROJECT_DIR}/include)

file(GLOB EINSUM_SOURCES ${EINSUM_SRC_DIR}/*/*.cpp)
if (EINSUM_ASSERTS)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
    add_definitions(-DEINSUM_ASSERTS)
endif()

if (EINSUM_SHARED_LIBRARY)
    add_library(einsum_lib SHARED ${EINSUM_SOURCES})
else ()
    add_library(einsum_lib STATIC ${EINSUM_SOURCES})
endif()
target_include_directories(einsum_lib PUBLIC ${EINSUM_INCLUDE_DIR})
install(TARGETS einsum_lib DESTINATION lib)

install(DIRECTORY ${EINSUM_INCLUDE_DIR}/ DESTINATION include)

add_subdirectory(test)

find_package(BISON REQUIRED 3.8.2)
find_package(FLEX REQUIRED)

bison_target(parser "${EINSUM_PARSER_DIR}/parser.y"  "${CMAKE_CURRENT_BINARY_DIR}/main.cc"
        DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/tok.h
        COMPILE_FLAGS "--yacc --define=api.pure=full --define=api.push-pull=push -v --define=parse.error=detailed"
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