#include "generate_proposals_op_util_nms_gpu.h"
#include "../../../include/common_gpu.h"
#include "Utils.h"

#include <algorithm>
#include <cfloat>
#include <vector>

#include "ResizeNearest.h"
#include "upsample_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

namespace nvinfer1 {
	ResizeNearestLayerPlugin::ResizeNearestLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
	}
	ResizeNearestLayerPlugin::~ResizeNearestLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	ResizeNearestLayerPlugin::ResizeNearestLayerPlugin(const void* data, size_t length) {

	}

	void ResizeNearestLayerPlugin::serialize(void* buffer)
	{
	}

	size_t ResizeNearestLayerPlugin::getSerializationSize()
	{
        return 0;
	}

	int ResizeNearestLayerPlugin::initialize()
	{

		m_inputTotalCount = mInputDimC*mInputDimH*mInputDimW;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, m_inputTotalCount * sizeof(float), cudaHostAllocDefault));

        m_ouputTotalCount = m_inputTotalCount *mScaleH*mScaleW;
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, m_ouputTotalCount * sizeof(float), cudaHostAllocDefault));

		return 0;
	}

	Dims ResizeNearestLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 1);

		mScaleH=2.0f;
        mScaleW=2.0f;

		mInputDimC = inputs[0].d[0];
		mInputDimH = inputs[0].d[1];
        mInputDimW = inputs[0].d[2];

        return DimsCHW(mScoreC, mScoreH*mScaleH, mScoreW*mScaleW );
	}

	template <typename Dtype>
	void ResizeNearestLayerPlugin::forwardCpu(const Dtype * X,//const float * const * inputs,
                                              Dtype* Y,       //      float * output,
                                              cudaStream_t stream){
        const int batch_size = 1;//
        const int num_channels = mInputDimC;
        const int input_height = mInputDimH;
        const int input_width  = mInputDimW;

        int output_width = input_width * mScaleW;
        int output_height = input_height * mScaleH;

        const Dtype* input = X;
        Dtype* output =(Dtype*) mOutputBuffer;
        int channels = num_channels * batch_size;

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(mInputBuffer, X, m_inputTotalCount * sizeof(Dtype),cudaMemcpyDeviceToHost, stream));

        const float rheight= (output_height> 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
        const float rwidth = (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;

        for (int h2 = 0; h2 < output_height; ++h2) {
            const float h1r = rheight * h2;
            const int h1 = h1r;
            const int h1p = (h1 < input_height - 1) ? 1 : 0;
            const float h1lambda = h1r - h1;
            const float h0lambda = (float)1. - h1lambda;
            for (int w2 = 0; w2 < output_width; ++w2) {
                const float w1r = rwidth * w2;
                const int w1 = w1r;
                const int w1p = (w1 < input_width - 1) ? 1 : 0;
                const float w1lambda = w1r - w1;
                const float w0lambda = (float)1. - w1lambda;
                const Dtype* Xdata = &input[h1 * input_width + w1];
                Dtype* Ydata = &output[h2 * output_width + w2];
                for (int c = 0; c < channels; ++c) {
                    Ydata[0] = h0lambda * (w0lambda * Xdata[0] +
                                                                 w1lambda * Xdata[w1p]) +
                               h1lambda * (w0lambda * Xdata[h1p * input_width] +
                                                                 w1lambda * Xdata[h1p * input_width + w1p]);
                    Xdata += input_width * input_height;
                    Ydata += output_width * output_height;
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(Y, mOutputBuffer, sizeof(Dtype)* m_ouputTotalCount, cudaMemcpyHostToDevice, stream));
    }

    int ResizeNearestLayerPlugin::enqueue(int batchSize,const void*const * inputs,
                                          void**			   outputs,
                                          void* workspace,    cudaStream_t stream){

	assert(batchSize == 1);

    switch (mDataType)
    {
    case DataType::kFLOAT:
        forwardCpu<float>((const float*)inputs[0],
            (float*)outputs[0],
            stream);
        //forwardCpu((const float *const *)inputs,(float *)outputs[0],stream);
        break;
    case DataType::kHALF:
        forwardCpu<__half>((const __half*)inputs[0],
            (__half*)outputs[0],
            stream);
        break;
    case DataType::kINT8:
        forwardCpu<u_int8_t>((const u_int8_t*)inputs[0],
            (u_int8_t*)outputs[0],
            stream);
        break;
    default:
        std::cerr << "error data type" << std::endl;
    }

    return 0;
    };

}