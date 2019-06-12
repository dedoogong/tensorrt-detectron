#include <algorithm>
#include <cfloat>
#include <vector>

#include "RoIAlign.h"
#include "../../../include/common_gpu.h"
#include "fp16.h"

using std::max;
using std::min;

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

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
	}

	void RoIAlignLayerPlugin::serialize(void* buffer)
	{
	}

	size_t RoIAlignLayerPlugin::getSerializationSize()
	{
        return 0;
	}

	int RoIAlignLayerPlugin::initialize()
	{
		return 0;
	}

	Dims RoIAlignLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 4);

        mFeatureMap_C = inputs[0].d[0];
        mFeatureMap_H = inputs[0].d[1];
        mFeatureMap_W = inputs[0].d[2];

        mRois_H = inputs[0].d[0];
        mRois_W = inputs[0].d[1];


        //X shape : 1, 256, 56, 56 R shape : 841, 5
        //X shape : 1, 256, 56, 56 R shape : 131, 5
        //X shape : 1, 256, 56, 56 R shape : 27, 5
        //X shape : 1, 256, 56, 56 R shape : 1, 5

        // 841, 256, 7,7
        // 131, 256, 7,7
        // 27, 256, 7,7
        // 1, 256, 7,7
        // Concat -> 1000, 256, 7,7?
		return DimsCHW(mRois_H, pooled_height, pooled_width);
	}
    /*
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
 
    */
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
	__global__ void RoIAlignForward(const int nthreads,
                                    const T * bottom_data,
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
                                        int roi_batch_ind = (int)offset_bottom_rois[0];

                                        // Do not using rounding; this implementation detail is critical
                                        T roi_start_w = offset_bottom_rois[1] * spatial_scale;
                                        T roi_start_h = offset_bottom_rois[2] * spatial_scale;
                                        T roi_end_w   = offset_bottom_rois[3] * spatial_scale;
                                        T roi_end_h   = offset_bottom_rois[4] * spatial_scale;
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
                                        int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_height / pooled_height); // e.g., = 2
                                        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

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

    template <typename DType>
	void RoIAlignLayerPlugin::forwardGpu(   const DType* features,
                                            const DType* rois,
                                            const float spatial_scale,
                                            const int pooled_height,
                                            const int pooled_width,
                                            const int sampling_ratio,
                                            cudaStream_t stream,
                                            const int num_rois,
                                            const int channels,
                                            const int height,
                                            const int width,
                                            DType* output) {

		auto output_size = num_rois * pooled_height * pooled_width * channels;
        // 841, 256, 7,7
        // 131, 256, 7,7
        // 27, 256, 7,7
        // 1, 256, 7,7
        // Concat -> 1000, 256, 7,7?

		//cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		dim3 grid(GET_BLOCKS_COUNT_IN_GRID(output_size));//std::min(THCCeilDiv((long)output_size, 512L), 4096L));
		dim3 block(CUDA_NUM_THREADS);

		//if (output.numel() == 0) {
		//    cudaGetLastError();
		//	return output;
		//}

        RoIAlignForward<DType> << <grid, block, 0, stream >> > (output_size,
                                                                features,
                                                                spatial_scale, channels,
                                                                height, width,
                                                                pooled_height, pooled_width,
                                                                sampling_ratio,
                                                                rois,
                                                                output);//     .data<scalar_t>());



		cudaGetLastError();
		return;
	}

    int RoIAlignLayerPlugin::enqueue(int batchSize,
                                     const void*const * inputs,
                                     void** outputs,
                                     void* workspace, cudaStream_t stream){
            assert(batchSize == 1);
            switch (mDataType){
                //  bottom: "fpn_resXf_sum"  <- Featuremap from Conv
                //  bottom: "rois_fpnX"    <- RoIs
                case DataType::kFLOAT:
                    forwardGpu<float>((const float*)inputs[0],
                                      (const float*)inputs[1],
                                        spatial_scale,
                                        pooled_height,
                                        pooled_width,
                                        sampling_ratio,
                                        stream,
                                        mRois_H,
                                        mFeatureMap_C,
                                        mFeatureMap_H,
                                        mFeatureMap_W,
                                        (float*)outputs[0]);
                    break;
                    /*
                case DataType::kHALF:
                    forwardGpu<__half>((const __half*)inputs[0],
                                       (const __half*)inputs[1],
                                       spatial_scale,
                                       pooled_height,
                                       pooled_width,
                                       sampling_ratio,
                                       stream,
                                       mRois_H,
                                       mFeatureMap_C,
                                       mFeatureMap_H,
                                       mFeatureMap_W,
                                       (__half*)outputs[0]);
                    break;*/
                case DataType::kINT8:
                    forwardGpu<u_int8_t>((const u_int8_t*)inputs[0],
                                         (const u_int8_t*)inputs[1],
                                         spatial_scale,
                                         pooled_height,
                                         pooled_width,
                                         sampling_ratio,
                                         stream,
                                         mRois_H,
                                         mFeatureMap_C,
                                         mFeatureMap_H,
                                         mFeatureMap_W,
                                         (u_int8_t*)outputs[0]);
                    /*
                    const DType * features,
                    const DType * rois,
                    const float spatial_scale,
                    const int pooled_height,
                    const int pooled_width,
                    const int sampling_ratio,
                    cudaStream_t stream,
                    const int num_rois,
                    const int channels,
                    const int height,
                    const int width,
                    DType* output */
                    break;
                default:
                    std::cerr << "error data type" << std::endl;
            }

            return 0;
        };

}  

