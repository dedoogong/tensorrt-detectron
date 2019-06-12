cmake_minimum_required(VERSION 2.8)
project(cub_sort LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CUDA_STANDARD 11)

find_package(CUB REQUIRED)

set(CUDA_VERBOSE_BUILD ON)


# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#add wrapper
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cub)

MESSAGE(${CMAKE_CURRENT_SOURCE_DIR})
#build
add_executable(run_cub_sort ${CMAKE_CURRENT_SOURCE_DIR}/examples/device/example_device_radix_sort.cu)
