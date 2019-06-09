//#include "BatchPermutationConfigs.h"
#include "BatchPermutationLayer.h"
#include <cub/cub/cub.cuh> 
#include "common_gpu.h"
#include "Utils.h"
#include <algorithm>
#include <cfloat>
#include <vector> 
using std::max;
using std::min;
using std::floor;
using std::ceil;
using namespace BatchPermutation;
 
	  
//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

namespace nvinfer1{
    BatchPermutationLayerPlugin::BatchPermutationLayerPlugin(const int cudaThread /*= 512*/):
                                                             mThreadCount(cudaThread){
        /*mClassCount = CLASS_NUM;
        mBatchPermutationKernel.clear();
        mBatchPermutationKernel.push_back(yolo1);
        mBatchPermutationKernel.push_back(yolo2);
        mBatchPermutationKernel.push_back(yolo3);

        mKernelCount = mBatchPermutationKernel.size();*/
    }
    BatchPermutationLayerPlugin::~BatchPermutationLayerPlugin(){
        if(mInputBuffer)
            CUDA_CHECK(cudaFreeHost(mInputBuffer));
        if(mOutputBuffer)
            CUDA_CHECK(cudaFreeHost(mOutputBuffer));
    }
    // create the plugin at runtime from a byte stream
    BatchPermutationLayerPlugin::BatchPermutationLayerPlugin(const void* data, size_t length){
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;        
        read(d, mThreadCount);
        //mBatchPermutationKernel.resize(mKernelCount);
        //auto kernelSize = mKernelCount*sizeof(BatchPermutationKernel);
        //memcpy(mBatchPermutationKernel.data(),d,kernelSize);
        //d += kernelSize;

        assert(d == a + length);
    }

    void BatchPermutationLayerPlugin::serialize(void* buffer)
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mThreadCount);
        //auto kernelSize = mKernelCount*sizeof(BatchPermutationKernel);
        //memcpy(d,mBatchPermutationKernel.data(),kernelSize);
        //d += kernelSize; 
        assert(d == a + getSerializationSize());
    }
    
    size_t BatchPermutationLayerPlugin::getSerializationSize()
    {  
        return sizeof(mThreadCount) + sizeof(BatchPermutation::BatchPermutationKernel) *
                                             mBatchPermutationKernel.size();
    }

    int BatchPermutationLayerPlugin::initialize()
    { 
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, m_inputTotalCount * sizeof(float), cudaHostAllocDefault));
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer,m_ouputTotalCount * sizeof(float), cudaHostAllocDefault));		
        return 0;
    }
    
    Dims BatchPermutationLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
		/*
			bottom: "roi_feat_shuffled"      // 1000, 256, 7, 7
			bottom: "rois_idx_restore_int32" // 1000, 1
			top:    "roi_feat"               // 1000, 256, 7, 7
		*/
        assert(nbInputDims==2);
        
		mRoIFeatureShuffledN = inputs[0].d[0];//1000
		mRoIFeatureShuffledC = inputs[0].d[1];//256
		mRoIFeatureShuffledH = inputs[0].d[2];//7
		mRoIFeatureShuffledW = inputs[0].d[3];//7 
		
		m_inputTotalCount = mRoIFeatureShuffledN * mRoIFeatureShuffledC * mRoIFeatureShuffledH * mRoIFeatureShuffledW;
		m_ouputTotalCount = m_inputTotalCount;

        return DimsCHW(mRoIFeatureShuffledN, mRoIFeatureShuffledC, mRoIFeatureShuffledH, mRoIFeatureShuffledW);
    }
	    
	  
    
    template <typename Dtype>
    void BatchPermutationLayerPlugin::forwardCpu(//const float *const * inputs,
                                                 //      float * output,
                                                 const Dtype * roi_feat_shuffled,// 1000, 256, 7, 7
                                                 const int * rois_idx_restore_int32,// 1000, 1
                                                 Dtype* roi_feat,// 1000, 12544(256*7*7)
                                                 cudaStream_t stream){
		CUDA_CHECK(cudaStreamSynchronize(stream));
		int i = 0;
		int size = 0;
		Dtype* inputData = (Dtype*)mInputBuffer;
		Dtype* inputIdx[mRoIFeatureShuffledN];
		
		CUDA_CHECK(cudaMemcpyAsync(mInputBuffer, roi_feat_shuffled, m_inputTotalCount * sizeof(float),
				cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(inputIdx, rois_idx_restore_int32, mRoIFeatureShuffledN * sizeof(float),
				cudaMemcpyDeviceToHost, stream));

		const float* X = (const Dtype*)mInputBuffer;       //roi_feat_shuffled : 1000, 256, 7,7
		const int* indices = (const int*)rois_idx_restore_int32;// 1000, 1

		Dtype* Y = roi_feat;
		const int N = mRoIFeatureShuffledN;
		const int C = mRoIFeatureShuffledC;
		const int H = mRoIFeatureShuffledH;
		const int W = mRoIFeatureShuffledW;

		const float* src = X;
		float* dst = Y;

		for (int i = 0; i < N; i++) {
			int idx = indices[i];
			//if(idx>=1000 || idx<0)
			//    printf("out of index range(0,1000) : %d\n",idx);
			std::memcpy(dst + i * C * H * W, src + idx * C * H * W, sizeof(float) * C * H * W);
		}  
        
		CUDA_CHECK(cudaMemcpyAsync(roi_feat, dst, m_ouputTotalCount * sizeof(float),
				cudaMemcpyDeviceToHost, stream));

    }


    int BatchPermutationLayerPlugin::enqueue(int batchSize,
                                             const void*const * inputs,
                                             void** outputs,
                                             void* workspace,
                                             cudaStream_t stream){
        assert(batchSize == 1); 

        switch (mDataType)
        {
            case DataType::kFLOAT :
                forwardCpu<float>((const float *)inputs[0],
                                  (const float *)inputs[1], 
                                  (float *)outputs[0], 
                                  stream); 
                break;
            case DataType::kHALF:
				forwardCpu<__half>((const __half *)inputs[0],
                                   (const __half *)inputs[1], 
                                   (__half *)outputs[0], 
                                   stream);
                break;
            case DataType::kINT8:
				forwardCpu<u_int8_t>((const u_int8_t *)inputs[0],
                                     (const u_int8_t *)inputs[1], 
                                     (u_int8_t *)outputs[0], 
                                     stream);
                break;
            default:
                std::cerr << "error data type" << std::endl;
        }
        
        return 0;
    };

} 