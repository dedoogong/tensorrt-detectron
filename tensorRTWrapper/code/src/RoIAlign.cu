#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_pooling_layer.hpp"


using std::max;
using std::min;

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace nvinfer1 {
	RoIAlignLayerPlugin::RoIAlignLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
		/*mClassCount = CLASS_NUM;
		mRoIAlignKernel.clear();
		mRoIAlignKernel.push_back(yolo1);
		mRoIAlignKernel.push_back(yolo2);
		mRoIAlignKernel.push_back(yolo3);

		mKernelCount = mRoIAlignKernel.size();*/
	}
	RoIAlignLayerPlugin::~RoIAlignLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	RoIAlignLayerPlugin::RoIAlignLayerPlugin(const void* data, size_t length) {
		using namespace Tn;
		const char* d = reinterpret_cast<const char*>(data), * a = d;
		read(d, mThreadCount);
		//mRoIAlignKernel.resize(mKernelCount);
		//auto kernelSize = mKernelCount*sizeof(RoIAlignKernel);
		//memcpy(mRoIAlignKernel.data(),d,kernelSize);
		//d += kernelSize;

		assert(d == a + length);
	}

	void RoIAlignLayerPlugin::serialize(void* buffer)
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer), * a = d;
		write(d, mThreadCount);
		//auto kernelSize = mKernelCount*sizeof(RoIAlignKernel);
		//memcpy(d,mRoIAlignKernel.data(),kernelSize);
		//d += kernelSize; 
		assert(d == a + getSerializationSize());
	}

	size_t RoIAlignLayerPlugin::getSerializationSize()
	{
		return sizeof(mThreadCount) + sizeof(RoIAlign::RoIAlignKernel) *
			mRoIAlignKernel.size();
	}

	int RoIAlignLayerPlugin::initialize()
	{
		/*
		int totalCount = 0;
		for(const auto& yolo : mRoIAlignKernel)
			totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

		totalCount = 0;//detection count
		for(const auto& yolo : mRoIAlignKernel)
			totalCount += yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection),
																			  cudaHostAllocDefault));
		*/

		return 0;
	}

	Dims RoIAlignLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 4);
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
	__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
		const Dtype spatial_scale, const int channels, const int height,
		const int width, const int pooled_height, const int pooled_width,
		const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			bottom_rois += n * 5;
			int roi_batch_ind = bottom_rois[0];
			int roi_start_w = round(bottom_rois[1] * spatial_scale);
			int roi_start_h = round(bottom_rois[2] * spatial_scale);
			int roi_end_w = round(bottom_rois[3] * spatial_scale);
			int roi_end_h = round(bottom_rois[4] * spatial_scale);

			// Force malformed ROIs to be 1x1
			int roi_width = max(roi_end_w - roi_start_w + 1, 1);
			int roi_height = max(roi_end_h - roi_start_h + 1, 1);
			Dtype bin_size_h = static_cast<Dtype>(roi_height)
				/ static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = static_cast<Dtype>(roi_width)
				/ static_cast<Dtype>(pooled_width);

			int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
				* bin_size_h));
			int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
				* bin_size_w));
			int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
				* bin_size_h));
			int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
				* bin_size_w));

			// Add roi offsets and clip to input boundaries
			hstart = min(max(hstart + roi_start_h, 0), height);
			hend = min(max(hend + roi_start_h, 0), height);
			wstart = min(max(wstart + roi_start_w, 0), width);
			wend = min(max(wend + roi_start_w, 0), width);
			bool is_empty = (hend <= hstart) || (wend <= wstart);

			// Define an empty pooling region to be zero
			Dtype maxval = is_empty ? 0 : -FLT_MAX;
			// If nothing is pooled, argmax = -1 causes nothing to be backprop'd
			int maxidx = -1;
			bottom_data += (roi_batch_ind * channels + c) * height * width;
			for (int h = hstart; h < hend; ++h) {
				for (int w = wstart; w < wend; ++w) {
					int bottom_index = h * width + w;
					if (bottom_data[bottom_index] > maxval) {
						maxval = bottom_data[bottom_index];
						maxidx = bottom_index;
					}
				}
			}
			top_data[index] = maxval;
			argmax_data[index] = maxidx;
		}
	}
 

//==============================================================================================

	template <typename T>
	__device__ T bilinear_interpolate(const T * bottom_data,
		const int height, const int width,
		T y, T x,
		const int index /* index for debug only*/) {

		// deal with cases that inverse elements are out of feature map boundary
		if (y < -1.0 || y > height || x < -1.0 || x > width) {
			//empty
			return 0;
		}

		if (y <= 0) y = 0;
		if (x <= 0) x = 0;

		int y_low = (int)y;
		int x_low = (int)x;
		int y_high;
		int x_high;

		if (y_low >= height - 1) {
			y_high = y_low = height - 1;
			y = (T)y_low;
		}
		else {
			y_high = y_low + 1;
		}

		if (x_low >= width - 1) {
			x_high = x_low = width - 1;
			x = (T)x_low;
		}
		else {
			x_high = x_low + 1;
		}

		T ly = y - y_low;
		T lx = x - x_low;
		T hy = 1. - ly, hx = 1. - lx;
		// do bilinear interpolation
		T v1 = bottom_data[y_low * width + x_low];
		T v2 = bottom_data[y_low * width + x_high];
		T v3 = bottom_data[y_high * width + x_low];
		T v4 = bottom_data[y_high * width + x_high];
		T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

		T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

		return val;
	}

	template <typename T>
	__global__ void RoIAlignForward(const int nthreads, const T * bottom_data,
		const T spatial_scale, const int channels,
		const int height, const int width,
		const int pooled_height, const int pooled_width,
		const int sampling_ratio,
		const T * bottom_rois, T * top_data) {
		CUDA_1D_KERNEL_LOOP(index, nthreads) {
			// (n, c, ph, pw) is an element in the pooled output
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int c = (index / pooled_width / pooled_height) % channels;
			int n = index / pooled_width / pooled_height / channels;

			const T * offset_bottom_rois = bottom_rois + n * 5;
			int roi_batch_ind = offset_bottom_rois[0];

			// Do not using rounding; this implementation detail is critical
			T roi_start_w = offset_bottom_rois[1] * spatial_scale;
			T roi_start_h = offset_bottom_rois[2] * spatial_scale;
			T roi_end_w = offset_bottom_rois[3] * spatial_scale;
			T roi_end_h = offset_bottom_rois[4] * spatial_scale;
			// T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
			// T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
			// T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
			// T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

			// Force malformed ROIs to be 1x1
			T roi_width = max(roi_end_w - roi_start_w, (T)1.);
			T roi_height = max(roi_end_h - roi_start_h, (T)1.);
			T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
			T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

			const T * offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

			// We use roi_bin_grid to sample the grid and mimic integral
			int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
			int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

			// We do average (integral) pooling inside a bin
			const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

			T output_val = 0.;
			for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
			{
				const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
				for (int ix = 0; ix < roi_bin_grid_w; ix++)
				{
					const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

					T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
					output_val += val;
				}
			}
			output_val /= count;

			top_data[index] = output_val;
		}
	}

	vector<float> RoIAlignLayerPlugin::forwardGpu(const float & input,
		const float & rois, // at::Tensor  to float
		const float spatial_scale,
		const int pooled_height,
		const int pooled_width,
		const int sampling_ratio) {
		//AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
		//AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

		auto num_rois = rois.size(0);
		auto channels = input.size(1);
		auto height = input.size(2);
		auto width = input.size(3);

		auto output = at::empty({ num_rois, channels, pooled_height, pooled_width }, input.options());
		auto output_size = num_rois * pooled_height * pooled_width * channels;
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		//TODO
		//dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
		dim3 block(512);

		if (output.numel() == 0) {
			//TODO
			//CudaCheck(cudaGetLastError());
			return output;
		}

			RoIAlignForward<scalar_t> << <grid, block, 0, stream >> > (
				output_size,
				input.contiguous().data<scalar_t>(),
				spatial_scale,
				channels,
				height,
				width,
				pooled_height,
				pooled_width,
				sampling_ratio,
				rois.contiguous().data<scalar_t>(),
				output.data<scalar_t>());
			});
		CUDA_CHECK(cudaGetLastError());
		return output;
	}

    // original roi pooling
	template <typename Dtype>
	void RoIAlignLayerPlugin::forwardGpuv1(const vector<Dtype>& bottom,
		const vector<Dtype>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_rois = bottom[1]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int* argmax_data = max_idx_.mutable_gpu_data();
		int count = top[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		ROIPoolForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
			count, bottom_data, spatial_scale_, channels_, height_, width_,
			pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
		CUDA_POST_KERNEL_CHECK;
	}
//==============================================================================================

int RoIAlignLayerPlugin::enqueue(int batchSize){
//////,??///
	assert(batchSize == 1);
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

