//#include "ResizeNearestConfigs.h"
#include "ResizeNearestLayer.h"
#include <cub/cub/cub.cuh>
#include "generate_proposals_op_util_nms_gpu.h"
#include "common_gpu.h"
#include "Utils.h"

#include <algorithm>
#include <cfloat>
#include <vector>

#include "upsample_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;


using namespace ResizeNearest;

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

namespace nvinfer1 {
	ResizeNearestLayerPlugin::ResizeNearestLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
		/*mClassCount = CLASS_NUM;
		mResizeNearestKernel.clear();
		mResizeNearestKernel.push_back(yolo1);
		mResizeNearestKernel.push_back(yolo2);
		mResizeNearestKernel.push_back(yolo3);

		mKernelCount = mResizeNearestKernel.size();*/
	}
	ResizeNearestLayerPlugin::~ResizeNearestLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	ResizeNearestLayerPlugin::ResizeNearestLayerPlugin(const void* data, size_t length) {
		using namespace Tn;
		const char* d = reinterpret_cast<const char*>(data), * a = d;
		read(d, mThreadCount);
		//mResizeNearestKernel.resize(mKernelCount);
		//auto kernelSize = mKernelCount*sizeof(ResizeNearestKernel);
		//memcpy(mResizeNearestKernel.data(),d,kernelSize);
		//d += kernelSize;

		assert(d == a + length);
	}

	void ResizeNearestLayerPlugin::serialize(void* buffer)
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer), * a = d;
		write(d, mThreadCount);
		//auto kernelSize = mKernelCount*sizeof(ResizeNearestKernel);
		//memcpy(d,mResizeNearestKernel.data(),kernelSize);
		//d += kernelSize; 
		assert(d == a + getSerializationSize());
	}

	size_t ResizeNearestLayerPlugin::getSerializationSize()
	{
		return sizeof(mThreadCount) + sizeof(ResizeNearest::ResizeNearestKernel) *
			mResizeNearestKernel.size();
	}

	int ResizeNearestLayerPlugin::initialize()
	{
		ResizeNearestParameter resize_nearest_param = this->layer_param_.resize_nearest_param();

		height_scale_ = resize_nearest_param.height_scale();
		width_scale_ = resize_nearest_param.width_scale();


		const int batch_size = bottom[0]->shape(0);
		const int num_channels = bottom[0]->shape(1);
		const int input_height = bottom[0]->shape(2);
		const int input_width = bottom[0]->shape(3);

		int output_width = input_width * width_scale_;
		int output_height = input_height * height_scale_;

		/*
		int totalCount = 0;
		for(const auto& yolo : mResizeNearestKernel)
			totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

		totalCount = 0;//detection count
		for(const auto& yolo : mResizeNearestKernel)
			totalCount += yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection),
																			  cudaHostAllocDefault));
		*/

		return 0;
	}

	Dims ResizeNearestLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 4);
		top[0]->Reshape(batch_size, num_channels, output_height, output_width);
		/*
		  bottom: "rpn_cls_probs_fpn2"
		  bottom: "rpn_bbox_pred_fpn2"
		  bottom: "im_info"
		  bottom: "anchor2"
		  top: "rpn_rois_fpn2"
		  top: "rpn_roi_probs_fpn2
		*/
		mScoreC = inputs[0].d[0];
		mScoreH = inputs[0].d[1];
		mScoreW = inputs[0].d[2];

		return index == 0 ? DimsCHW(300, 5) : DimsCHW(300, 1);
	}



	template <typename Dtype>
	void ResizeNearestLayerPlugin::forwardGpu(
}


int ResizeNearestLayerPlugin::enqueue(int batchSize,

	assert(batchSize == 1);

	const float* X = (const float*)bottom[0]->cpu_data();//Input(0);
	float* Y = (float*)(top[0]->mutable_cpu_data());// Output(0); 
	const int batch_size = bottom[0]->shape(0);
	const int num_channels = bottom[0]->shape(1);
	const int input_height = bottom[0]->shape(2);
	const int input_width = bottom[0]->shape(3);

	int output_width = input_width * width_scale_;
	int output_height = input_height * height_scale_;

	//

	const float* input = X;
	float* output = Y;
	int channels = num_channels * batch_size;

	const float rheight = (output_height > 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
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
			const float* Xdata = &input[h1 * input_width + w1];
			float* Ydata = &output[h2 * output_width + w2];
			for (int c = 0; c < channels; ++c) {
				Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
					h1lambda *
					(w0lambda * Xdata[h1p * input_width] +
						w1lambda * Xdata[h1p * input_width + w1p]);
				Xdata += input_width * input_height;
				Ydata += output_width * output_height;
			}
		}
	}

/*
const int channels = mCHW.d[0];
const int64_t in_height = mCHW.d[1];
const int64_t in_width = mCHW.d[2];
const int64_t out_height = mOutputHeight;
const int64_t out_width = mOutputWidth;
int totalElems = batchSize * in_height * in_width * channels;
//int N, C, H, W == mCHW.d[0], mOutputHeight, mOutputWidth

// Handle no-op resizes efficiently.
if (out_height == in_height && out_width == in_width) {
	CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0],
							   totalElems * type2size(mDataType),
							   cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return 0;
}*/
switch (mDataType)
{
case DataType::kFLOAT:
	forwardGpu<float>((const float*)inputs[0],
		(const float*)inputs[1],
		(const float*)inputs[2],
		(const float*)inputs[3],
		(float*)outputs[0],
		(float*)outputs[1],
		stream);
	//forwardGpu((const float *const *)inputs,(float *)outputs[0],stream);
	break;
case DataType::kHALF:
	forwardGpu<__half>((const __half*)inputs[0],
		(const __half*)inputs[1],
		(const __half*)inputs[2],
		(const __half*)inputs[3],
		(__half*)outputs[0],
		(__half*)outputs[1],
		stream);
	break;
case DataType::kINT8:
	forwardGpu<u_int8_t>((const u_int8_t*)inputs[0],
		(const u_int8_t*)inputs[1],
		(const u_int8_t*)inputs[2],
		(const u_int8_t*)inputs[3],
		(u_int8_t*)outputs[0],
		(u_int8_t*)outputs[1],
		stream);
	break;
default:
	std::cerr << "error data type" << std::endl;
}

return 0;
};

	}