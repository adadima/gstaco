cmake_minimum_required(VERSION 3.19)
project(einsum_taco)
set(CMAKE_CXX_STANDARD 14)

option(EINSUM_SHARED_LIBRARY "Build as a shared library" ON)

set(EINSUM_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(EINSUM_SRC_DIR     ${EINSUM_PROJECT_DIR}/src)
set(EINSUM_TEST_DIR    ${EINSUM_PROJECT_DIR}/test)
set(EINSUM_INCLUDE_DIR ${EINSUM_PROJECT_DIR}/include)

file(GLOB EINSUM_IR_SOURCES ${EINSUM_SRC_DIR}/ir/*.cpp)

if (EINSUM_SHARED_LIBRARY)
    add_library(einsum_ir SHARED ${EINSUM_IR_SOURCES})
else ()
    add_library(einsum_ir STATIC ${EINSUM_IR_SOURCES})
endif()
target_include_directories(einsum_ir PUBLIC ${EINSUM_INCLUDE_DIR})
install(TARGETS einsum_ir DESTINATION lib)

add_executable(einsum main.cpp include/einsum_taco/type.h src/type.cpp)
target_link_libraries(einsum einsum_ir)
install(TARGETS einsum DESTINATION bin)

install(DIRECTORY ${EINSUM_INCLUDE_DIR}/ DESTINATION include)

add_subdirectory(test)
