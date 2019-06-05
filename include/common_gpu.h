// Created by seunghyun lee on 19. 5. 23.
#ifndef COMMON_GPU_H
#define COMMON_GPU_H

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <curand.h>
#include <driver_types.h>
#include <algorithm>

// CUDA: various checks for different function calls.

const int pooled_width=14;
const int pooled_height=14;
const float spatial_scale[4]={0.25f, 0.125f, 0.0625f, 0.03125f}; // ==1/4, 1/8, 1/16, 1/32
const int rois_count=1000;

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_CHECK(condition)                                 \
{                                                             \
    cudaError_t error = condition;                            \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
}

#define CHECK(condition)                                 \
{                                                             \
    cudaError_t error = condition;                            \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
}
// 1D grid
const int CAFFE_CUDA_NUM_THREADS = 128;

#define CUDA_NUM_THREADS 512
// CUDA: number of blocks for threads.
inline int GET_BLOCKS_COUNT_IN_GRID(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// 2D grid
const int CAFFE_CUDA_NUM_THREADS_2D_DIMX = 16;
const int CAFFE_CUDA_NUM_THREADS_2D_DIMY = 16;

// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
// 1D grid
const int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;
// 2D grid
const int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX = 128;
const int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY = 128;

const int kCUDAGridDimMaxX = 2147483647;
const int kCUDAGridDimMaxY = 65535;
const int kCUDAGridDimMaxZ = 65535;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int CAFFE_GET_BLOCKS(const int N) {
    return std::max(
            std::min(
                    (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
                    CAFFE_MAXIMUM_NUM_BLOCKS),
            // Use at least 1 block, since CUDA does not allow empty block
            1);
}

/**
 * @brief Compute the number of blocks needed to run N threads for a 2D grid
 */
inline dim3 CAFFE_GET_BLOCKS_2D(const int N, const int /* M */) {
    dim3 grid;
    // Not calling the 1D version for each dim to keep all constants as literals

    grid.x = std::max(
            std::min(
                    (N + CAFFE_CUDA_NUM_THREADS_2D_DIMX - 1) /
                    CAFFE_CUDA_NUM_THREADS_2D_DIMX,
                    CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX),
            1);

    grid.y = std::max(
            std::min(
                    (N + CAFFE_CUDA_NUM_THREADS_2D_DIMY - 1) /
                    CAFFE_CUDA_NUM_THREADS_2D_DIMY,
                    CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY),
            1);

    return grid;
}


#endif //COMMON_GPU_H
