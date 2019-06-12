cmake_minimum_required(VERSION 2.8)

if (POLICY CMP0074)
    cmake_policy (SET CMP0074 NEW)
endif (POLICY CMP0074)


project(demo LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

#add wrapper
set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/caffe2)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/cub)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/Eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

#build runYolov3
add_executable(runYolov3 main.cpp dataReader.cpp include/common_gpu.h)
target_link_libraries(runYolov3 TrtNet ${OpenCV_LIBS})

#build runMRCNN
add_executable(runMRCNN
        main.cpp
        dataReader.cpp
        )
target_link_libraries(runMRCNN TrtNet ${OpenCV_LIBS})
install(TARGETS runYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
install(TARGETS runMRCNN DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)