cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
MESSAGE("-------------------------------------------------------------")
MESSAGE("-----------------TensorRT Wrapper Cmake : -------------------")
MESSAGE(${CMAKE_SOURCE_DIR}/cmake/Modules)

project(trtNet LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
#include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# -------------
# CUB
# -------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(CUB REQUIRED)
include_directories(${CUB_INCLUDE_DIR})
include_directories(${CUB_INCLUDE_DIR}/cub)
MESSAGE("CUB FOUND")
MESSAGE(${CUB_INCLUDE_DIR})
#src
set(PLUGIN_SOURCES
  src/EntroyCalibrator.cpp
  src/UpsampleLayer.cpp
  src/UpsampleLayer.cu
  src/YoloLayer.cu
  src/TrtNet.cpp
  src/BatchPermuteLayer.cpp
  src/BoxTransformLayer.cpp
  src/CollectNDistributeFPNLayer.cpp
  src/BoxWithNMSLimitLayer.cpp
  src/RoIAlign.cu
  src/GenerateProposalLayer.cu
)
#

# CUDA Configuration
find_package(CUDA REQUIRED)
set(CUDA_VERBOSE_BUILD ON)
# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

#GLOG
find_package(glog 0.4.0 REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE /usr/local/cuda-10.0/bin)
# if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
#   list(APPEND CUDA_NVCC_FLAGS -std=c++11)
# endif()

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_LIBRARY(TrtNet STATIC ${PLUGIN_SOURCES})

target_include_directories(TrtNet PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(TrtNet ${TENSORRT_LIBRARY})
target_link_libraries(TrtNet glog::glog)