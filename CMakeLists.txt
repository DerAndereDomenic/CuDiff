cmake_minimum_required(VERSION 3.27)

project(AutoDiff LANGUAGES C CXX VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

add_library(CuDiff INTERFACE)
target_include_directories(CuDiff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


if (PROJECT_IS_TOP_LEVEL)
    enable_language(CUDA)

    add_executable(Test)
    target_sources(Test PRIVATE samples/test/main.cu)
    target_link_libraries(Test PRIVATE CuDiff)
endif()