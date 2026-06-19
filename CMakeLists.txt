cmake_minimum_required(VERSION 3.12)
project(forecaster)

set(CMAKE_CXX_STANDARD 14)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add C++ compiler flags for AVX2 and OpenMP (Multi-threading support)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -fopenmp -O3")
endif()

pybind11_add_module(forecaster kernel.cpp)
