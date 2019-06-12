#include "cub/cub/block/block_reduce.cuh"
#include "cub/cub/cub.cuh"

// Block-sorting CUDA kernel
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
    using namespace cub;

    // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads
    // owning 16 integer items each
    typedef BlockRadixSort<int, 128, 16>                     BlockRadixSort;
    typedef BlockLoad<int, 128, 16, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
    typedef BlockStore<int, 128, 16, BLOCK_STORE_TRANSPOSE> BlockStore;

    // Allocate shared memory
    __shared__ union {
        typename BlockRadixSort::TempStorage  sort;
        typename BlockLoad::TempStorage       load;
        typename BlockStore::TempStorage      store;
    } temp_storage;

    int block_offset = blockIdx.x * (128 * 16);	  // OffsetT for this block's ment

    // Obtain a segment of 2048 consecutive keys that are blocked across threads
    int thread_keys[16];
    BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
    __syncthreads();

    // Collectively sort the keys
    BlockRadixSort(temp_storage.sort).Sort(thread_keys);
    __syncthreads();

    // Store the sorted segment
    BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}

int main(void){

    BlockSortKernel<<<1024,512>>>
    return 0;
}