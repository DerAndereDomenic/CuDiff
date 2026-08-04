cmake_minimum_required(VERSION 3.27)

project(AutoDiff LANGUAGES C CXX VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(CuDiff INTERFACE)
target_include_directories(CuDiff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


if (PROJECT_IS_TOP_LEVEL)
    enable_language(CUDA)
    include(CTest)
    enable_testing()

    include(FetchContent)
    # Fetch glm
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
        )
    
    # Make glm available
    FetchContent_MakeAvailable(glm)

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(Test)
    target_sources(Test PRIVATE samples/test/main.cu)
    target_link_libraries(Test PRIVATE CuDiff glm)

    add_executable(Benchmark)
    target_sources(Benchmark PRIVATE samples/Benchmark/main.cu)
    target_link_libraries(Benchmark PRIVATE CuDiff glm)

    add_executable(CuDiffTests)
    target_sources(CuDiffTests PRIVATE tests/CuDiffTests.cu)
    target_link_libraries(CuDiffTests PRIVATE CuDiff glm GTest::gtest_main)
    target_include_directories(CuDiffTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    include(GoogleTest)
    gtest_discover_tests(CuDiffTests)

endif()