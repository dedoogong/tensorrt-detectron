//#include "GenerateProposalConfigs.h"
#include "GenerateProposalLayer.h"
#include <cub/cub/cub.cuh>
#include "generate_proposals_op_util_nms_gpu.h"
#include "common_gpu.h"
#include "Utils.h"
using namespace GenerateProposal;

#include <algorithm>
#include <cfloat>
#include <vector> 
#include "bbox_transform_layer.hpp"
#include "generate_proposals_op_util_boxes.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;
   

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

namespace nvinfer1 {
	BoxTransformLayerPlugin::BoxTransformLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
		/*mClassCount = CLASS_NUM;
		mBoxTransformKernel.clear();
		mBoxTransformKernel.push_back(yolo1);
		mBoxTransformKernel.push_back(yolo2);
		mBoxTransformKernel.push_back(yolo3);

		mKernelCount = mBoxTransformKernel.size();*/
	}
	BoxTransformLayerPlugin::~BoxTransformLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	BoxTransformLayerPlugin::BoxTransformLayerPlugin(const void* data, size_t length) {
		using namespace Tn;
		const char* d = reinterpret_cast<const char*>(data), * a = d;
		read(d, mThreadCount);
		//mBoxTransformKernel.resize(mKernelCount);
		//auto kernelSize = mKernelCount*sizeof(BoxTransformKernel);
		//memcpy(mBoxTransformKernel.data(),d,kernelSize);
		//d += kernelSize;

		assert(d == a + length);
	}

	void BoxTransformLayerPlugin::serialize(void* buffer)
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer), * a = d;
		write(d, mThreadCount);
		//auto kernelSize = mKernelCount*sizeof(BoxTransformKernel);
		//memcpy(d,mBoxTransformKernel.data(),kernelSize);
		//d += kernelSize; 
		assert(d == a + getSerializationSize());
	}

	size_t BoxTransformLayerPlugin::getSerializationSize()
	{
		return sizeof(mThreadCount) + sizeof(BoxTransform::BoxTransformKernel) *
			mBoxTransformKernel.size();
	}

	int BoxTransformLayerPlugin::initialize()
	{
		/*
		int totalCount = 0;
		for(const auto& yolo : mBoxTransformKernel)
			totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

		totalCount = 0;//detection count
		for(const auto& yolo : mBoxTransformKernel)
			totalCount += yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection),
																			  cudaHostAllocDefault));
		*/

		return 0;
	}

	Dims BoxTransformLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
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
	void BoxTransformLayerPlugin::forwardGpu( 
	}


	int BoxTransformLayerPlugin::enqueue(int batchSize,
		const void* const* inputs,
		void** outputs,
		void* workspace,
		cudaStream_t stream) {
		assert(batchSize == 1);




		using std::max;
		using std::min;
		using std::floor;
		using std::ceil;

		template <typename Dtype>
		void BBoxTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*> & top) {

			weights_1_ = bboxtransform_param.weights_1();
			weights_2_ = bboxtransform_param.weights_2();
			weights_3_ = bboxtransform_param.weights_3();
			weights_4_ = bboxtransform_param.weights_4();
			apply_scale_ = bboxtransform_param.apply_scale();
			correct_transform_coords_ = bboxtransform_param.correct_transform_coords();
			top[0]->Reshape(bottom[1]->shape());


			const float* roi_in = (const float*)bottom[0]->cpu_data();//(1000,4)
			const float* delta_in = (const float*)bottom[1]->cpu_data();//(1000,8) bg x1 y1 x2 y2, human x1 y1 x2 y2
			const float* iminfo_in = (const float*)bottom[2]->cpu_data();
			float* box_out = (float*)(top[0]->mutable_cpu_data());

			//rpn_rois 2D
			//bbox_pred shape : 2D 
			 //im_info shape : 2D  

			for (int i = 0; i < min(bottom[0]->count(), 40); i += 4) {
				printf("roi_in[%d] : %.2f %.2f %.2f %.2f \n", i / 4, roi_in[i], roi_in[i + 1], roi_in[i + 2], roi_in[i + 3]);
			}
			const int box_dim = 4;
			const int N = bottom[0]->shape(0);//roi_in.dim32(0) == 1000

			CHECK_EQ(bottom[0]->shape().size(), 2);//roi_in.ndim()
			CHECK(bottom[0]->shape(1) == box_dim || bottom[0]->shape(1) == box_dim + 1);//roi_in.dim32(1)  

			CHECK_EQ(bottom[1]->shape().size(), 2);//delta_in.ndim()
			CHECK_EQ(bottom[1]->shape(0), N);//delta_in.dim32(0)
			CHECK_EQ(bottom[1]->shape(1) % box_dim, 0);//delta_in.dim32(1) 

			const int num_classes = (bottom[1]->shape(1)) / box_dim; ;//delta_in.dim32(1) / box_dim;

			CHECK_EQ(bottom[2]->shape().size(), 2);//iminfo_in.ndim()
			CHECK_EQ(bottom[2]->shape(1), 3); //iminfo_in.dim32(1)
			const int batch_size = (bottom[2]->shape(0));//iminfo_in.dim32(0);

			//CHECK_EQ(weights_.size(), 4);

			Eigen::Map<const caffe2::ERArrXXf> boxes0(roi_in, bottom[0]->shape(0), bottom[0]->shape(1));//(1000,4)---------------------------------
			Eigen::Map<const caffe2::ERArrXXf> deltas0(delta_in, bottom[1]->shape(0), bottom[1]->shape(1));//(1000,8)

			// Count the number of RoIs per batch
			vector<int> num_rois_per_batch(batch_size, 0);
			if (bottom[0]->shape(1) == box_dim) {//roi_in.dim32(1) 
				CHECK_EQ(batch_size, 1);
				num_rois_per_batch[0] = N;//1000
			}
			else {
				const auto& roi_batch_ids = boxes0.col(0);
				for (int i = 0; i < roi_batch_ids.size(); ++i) {
					const int roi_batch_id = roi_batch_ids(i);
					CHECK_LT(roi_batch_id, batch_size);
					num_rois_per_batch[roi_batch_id]++;
				}
			}

			CHECK_EQ(bottom[2]->shape(0), batch_size);
			CHECK_EQ(bottom[2]->shape(1), 3);//1,3,256,256

			Eigen::Map<const caffe2::ERArrXXf> iminfo(iminfo_in, bottom[2]->shape(0), bottom[2]->shape(1));

			printf("iminfo_in : %f %f %f\n", iminfo_in[0], iminfo_in[1], iminfo_in[2]);

			// We assume roi_in and delta_in over multiple batches are grouped
			// together in increasing order as generated by GenerateProposalsOp
			int offset = 0;
			vector<float> weights_;
			weights_.push_back(weights_1_);
			weights_.push_back(weights_2_);
			weights_.push_back(weights_3_);
			weights_.push_back(weights_4_);

			const int   num_rois = num_rois_per_batch[i];
			const auto& cur_iminfo = iminfo.row(i);
			const float scale_before = cur_iminfo(2);
			const float scale_after = apply_scale_ ? cur_iminfo(2) : 1.0;

			int img_h = int(cur_iminfo(0) / scale_before + 0.5);
			int img_w = int(cur_iminfo(1) / scale_before + 0.5);
			printf("\n[origin H,W] : [%f, %f] [After H,W] : [%d, %d] scale_before : %f scale_after : %f \n", cur_iminfo(0), cur_iminfo(1), img_h, img_w, scale_before, scale_after);

			caffe2::EArrXXf cur_boxes = boxes0.rightCols(box_dim).block(offset, 0, num_rois, box_dim);// boxes0.block(0, 0, 1000, 4) -> (1000, 0, 1000, 4) -> (2000, 0, 1000, 4) ....
			//const auto& cur_boxes =  boxes0.block(offset, 0, num_rois, box_dim); 

			//Do not apply scale for angle in rotated boxes
			cur_boxes.leftCols(4) /= scale_before;

			for (int k = 0; k < num_classes; k++) {//class == human, bg
				const auto& cur_deltas = deltas0.block(offset, k * box_dim, num_rois, box_dim);
				// deltas0.block(0, 0, 1000, 4) -> (0, 0, 1000, 4) -> (0, 0, 1000, 4) -> 
				//              (1000, 0, 1000, 4) -> (1000, 4, 1000, 4) -> (1000, 8, 1000, 4) ....

				if (0) {
					for (int i = 0; i < cur_boxes.rows() * cur_boxes.cols() / 100; i += 4) {
						printf("cur_boxes[%d] : %.2f %.2f %.2f %.2f \n", i / 4, cur_boxes.data()[i], cur_boxes.data()[i + 1], cur_boxes.data()[i + 2], cur_boxes.data()[i + 3]);
					}
					for (int i = 0; i < deltas0.rows() / 100; i++) {
						printf("deltas0[%d] : %.2f %.2f %.2f %.2f \n", i, deltas0.col(k * box_dim)[i], deltas0.col(k * box_dim + 1)[i], deltas0.col(k * box_dim + 2)[i], deltas0.col(k * box_dim + 3)[i]);
					}

					for (int i = 0; i < cur_deltas.rows() / 100; i++) {
						printf("cur_deltas[%d] : %.2f %.2f %.2f %.2f \n", i, cur_deltas.col(0)[i], cur_deltas.col(1)[i], cur_deltas.col(2)[i], cur_deltas.col(3)[i]);
					}
				}

				caffe2::EArrXXf clip_boxes;
				//printf("cur_boxes shape : %ld, %ld \n", cur_boxes.rows(), cur_boxes.cols()); //1000,4
				//printf("cur_deltas shape : %ld, %ld \n", cur_deltas.rows(), cur_deltas.cols()); //1000,4

				const auto& trans_boxes = caffe2::utils::bbox_transform(cur_boxes,
					cur_deltas,
					weights_,
					caffe2::utils::BBOX_XFORM_CLIP_DEFAULT,
					correct_transform_coords_,
					angle_bound_on_,
					angle_bound_lo_,
					angle_bound_hi_);
				clip_boxes = caffe2::utils::clip_boxes(trans_boxes, img_h, img_w, clip_angle_thresh_);

				if (0) {
					printf("trans_boxes shape : %ld, %ld \n", trans_boxes.rows(), trans_boxes.cols()); //1000,4

					for (int i = 0; i < trans_boxes.rows() / 100; i++) {
						printf("trans_boxes[%d] : %.2f %.2f %.2f %.2f \n", i, trans_boxes.col(0)[i],
							trans_boxes.col(1)[i],
							trans_boxes.col(2)[i],
							trans_boxes.col(3)[i]);
					}

					// Do not apply scale for angle in rotated boxes
					// printf("clip_boxes shape : %ld, %ld \n", clip_boxes.rows(), clip_boxes.cols()); //1000,4
					for (int i = 0; i < clip_boxes.rows() / 100; i++) {// .data()[i] -> column major order!!!
						printf("clip_boxes[%d] : %.2f %.2f %.2f %.2f \n", i / 4, clip_boxes.col(0)[i],
							clip_boxes.col(1)[i],
							clip_boxes.col(2)[i],
							clip_boxes.col(3)[i]);
					}
				}

				// TODO : assign clip boxes to top blob instead of eigen mat! / check clip_boxes function / check normalization is valid or not / check 800 x 800 input!
				clip_boxes.leftCols(4) *= scale_after;
				//new_boxes.block(offset, k * box_dim, num_rois, box_dim) = clip_boxes;
				printf("clip_boxes.rows(),clip_boxes.cols() : %ld %ld \n", clip_boxes.rows(), clip_boxes.cols());
				for (int j = 0; j < clip_boxes.rows(); j++) {// bg x1 y1 x2 y2, fg x1 y1 x2 y2

					box_out[num_classes * box_dim * j + (k)* box_dim] = clip_boxes.col(0)[j];
					box_out[num_classes * box_dim * j + 1 + (k)* box_dim] = clip_boxes.col(1)[j];
					box_out[num_classes * box_dim * j + 2 + (k)* box_dim] = clip_boxes.col(2)[j];
					box_out[num_classes * box_dim * j + 3 + (k)* box_dim] = clip_boxes.col(3)[j];
					/*
					0123 4567 <= 0
					8901 2345 <= 1
					*/
				}
			}

			offset += num_rois;
			/*printf("Top shape : %d %d\n", top[0]->shape(0), top[0]->shape(1));
			for(int j=0;  j < top[0]->shape(0)*top[0]->shape(1)/100;  j++){// bg x1 y1 x2 y2, fg x1 y1 x2 y2
			if(j%8==0)
				printf("\n");
			printf("%.2f ",box_out[j]);
			}
			printf("\n==============================================BBoxTransformLayer Done=======================================\n");
			*/
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