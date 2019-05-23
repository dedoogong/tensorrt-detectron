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
#define CUDA_ENFORCE(condition, ...) cudaError_t error = condition;
#define CUBLAS_ENFORCE(condition) cudaError_t error = condition;
#define CURAND_ENFORCE(condition) cudaError_t error = condition;

#define CUDA_CHECK(condition)                                 \
  {                                                        \
    cudaError_t error = condition;                            \
    CHECK(error == cudaSuccess) << cudaGetErrorString(error); \
  }

#define CUDA_DRIVERAPI_ENFORCE(condition)                            \
  {                                                               \
    CUresult result = condition;                                     \
    if (result != CUDA_SUCCESS) {                                    \
      const char* msg;                                               \
      cuGetErrorName(result, &msg);                                  \
      CAFFE_THROW("Error at: ", __FILE__, ":", __LINE__, ": ", msg); \
    }                                                                \
  }

#define CUDA_DRIVERAPI_CHECK(condition)                                 \
  {                                                                  \
    CUresult result = condition;                                        \
    if (result != CUDA_SUCCESS) {                                       \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": " \
                 << msg;                                                \
    }                                                                   \
  }

#define CUBLAS_CHECK(condition)                    \
  {                                             \
    cublasStatus_t status = condition;             \
    CHECK(status == CUBLAS_STATUS_SUCCESS)         \
        << ::caffe2::cublasGetErrorString(status); \
  }

#define CURAND_CHECK(condition)                    \
  {                                             \
    curandStatus_t status = condition;             \
    CHECK(status == CURAND_STATUS_SUCCESS)         \
        << ::caffe2::curandGetErrorString(status); \
  }

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

// 1D grid
const int CAFFE_CUDA_NUM_THREADS = 128;
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
