//#include "BoxNMSConfigs.h"
#include "BoxNMSLayer.h"
#include <cub/cub/cub.cuh>
#include "generate_proposals_op_util_nms_gpu.h"
#include "common_gpu.h"
#include "Utils.h"
using namespace BoxNMS;

#include "bbox_with_nms_limit_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

template <class Derived, class Func>
vector<int> filter_with_indices(
	const Eigen::ArrayBase<Derived>& array,
	const vector<int>& indices,
	const Func& func) {
	vector<int> ret;
	for (auto& cur : indices) {
		if (func(array[cur])) {
			ret.push_back(cur);
		}
	}
	return ret;
}
  
namespace nvinfer1 {
	BoxNMSLayerPlugin::BoxNMSLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
		/*mClassCount = CLASS_NUM;
		mBoxNMSKernel.clear();
		mBoxNMSKernel.push_back(yolo1);
		mBoxNMSKernel.push_back(yolo2);
		mBoxNMSKernel.push_back(yolo3);

		mKernelCount = mBoxNMSKernel.size();*/
	}
	BoxNMSLayerPlugin::~BoxNMSLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	BoxNMSLayerPlugin::BoxNMSLayerPlugin(const void* data, size_t length) {
		using namespace Tn;
		const char* d = reinterpret_cast<const char*>(data), * a = d;
		read(d, mThreadCount);
		//mBoxNMSKernel.resize(mKernelCount);
		//auto kernelSize = mKernelCount*sizeof(BoxNMSKernel);
		//memcpy(mBoxNMSKernel.data(),d,kernelSize);
		//d += kernelSize;

		assert(d == a + length);
	}

	void BoxNMSLayerPlugin::serialize(void* buffer)
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer), * a = d;
		write(d, mThreadCount);
		//auto kernelSize = mKernelCount*sizeof(BoxNMSKernel);
		//memcpy(d,mBoxNMSKernel.data(),kernelSize);
		//d += kernelSize; 
		assert(d == a + getSerializationSize());
	}

	size_t BoxNMSLayerPlugin::getSerializationSize()
	{
		return sizeof(mThreadCount) + sizeof(BoxNMS::BoxNMSKernel) *
			mBoxNMSKernel.size();
	}

	int BoxNMSLayerPlugin::initialize()
	{
		BoxWithNMSLimitParameter box_nms_param = this->layer_param_.box_nms_param();

		score_thresh_ = box_nms_param.score_thresh();
		nms_thresh_ = box_nms_param.nms_thresh();
		detections_per_im_ = box_nms_param.detections_per_im();
		soft_nms_enabled_ = box_nms_param.soft_nms_enabled();
		soft_nms_method_ = box_nms_param.soft_nms_method_();
		soft_nms_sigma_ = box_nms_param.soft_nms_sigma();
		soft_nms_min_score_thresh_ = box_nms_param.soft_nms_min_score_thresh();
		rotated_ = box_nms_param.rotated();
		/*
		int totalCount = 0;
		for(const auto& yolo : mBoxNMSKernel)
			totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

		totalCount = 0;//detection count
		for(const auto& yolo : mBoxNMSKernel)
			totalCount += yolo.width*yolo.height * CHECK_COUNT;
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection),
																			  cudaHostAllocDefault));
		*/

		return 0;
	}

	Dims BoxNMSLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 4);



		int nms_max_count = 300;// tscores.dim(1);
		vector<int> out_scores_shape;
		vector<int> out_boxes_shape;
		vector<int> out_classes_shape;

		out_scores_shape.push_back(nms_max_count);
		out_scores_shape.push_back(1);
		top[0]->Reshape(out_scores_shape);

		out_boxes_shape.push_back(nms_max_count);
		out_boxes_shape.push_back(5);
		top[1]->Reshape(out_boxes_shape);

		out_classes_shape.push_back(nms_max_count);
		out_classes_shape.push_back(1);
		top[2]->Reshape(out_classes_shape);



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
	void BoxNMSLayerPlugin::forwardGpu(
}


int BoxNMSLayerPlugin::enqueue(int batchSize,
	 
	assert(batchSize == 1);


const float* tscores = (const float*)bottom[0]->cpu_data();//cls_prob  blob (1000,2)
const float* tboxes = (const float*)bottom[1]->cpu_data(); //pred_box  blob (1000,8) from bbox_transform layer
float* out_scores = (float*)(top[0]->mutable_cpu_data()); //score_nms blob
float* out_boxes = (float*)(top[1]->mutable_cpu_data());   //bbox_nms  blob
float* out_classes = (float*)(top[2]->mutable_cpu_data());//class_nms blob

//printf("Top shape : %d %d\n", top[0]->shape(0), top[0]->shape(1));
printf("==============================================BoxWithNMSLimitLayer start====================================\n");
/*
for(int j=0;  j < bottom[1]->shape(0)*bottom[1]->shape(1)/100;  j++){// bg x1 y1 x2 y2, fg x1 y1 x2 y2
	if(j%8==0)
		printf("\n");
	printf("%.2f ",tboxes[j]);
}
*/
//printf("cls_prob: [%d, %d]\n",bottom[0]->shape(0),bottom[0]->shape(1)); 
//printf("pred_bbox: [%d, %d]\n",bottom[1]->shape(0),bottom[1]->shape(1)); 

const int box_dim = 4;// rotated_ ? 5 :
const int N = bottom[0]->shape(0);
// tscores: (num_boxes, num_classes), 0 for background

CHECK_EQ(bottom[0]->shape().size(), 2);//tscores.ndim(), 2
CHECK_EQ(bottom[1]->shape().size(), 2);//tboxes.ndim(),  2

int num_classes = bottom[0]->shape(1);// tscores.dim(1);

CHECK_EQ(N, bottom[1]->shape(0));//tboxes.dim(0));
CHECK_EQ(num_classes* box_dim, bottom[1]->shape(1));// tboxes.dim(1));

int batch_size = 1;
vector<float> batch_splits_default(1, bottom[0]->shape(0));//tscores.dim(0)
const float* batch_splits_data = batch_splits_default.data();

Eigen::Map<const caffe2::EArrXf> batch_splits(batch_splits_data, batch_size);
CHECK_EQ(batch_splits.sum(), N);

//vector<int> total_keep_per_batch(batch_size);
int offset = 0;
int final_nms_count = 0;
for (int b = 0; b < batch_splits.size(); ++b) {// size == 1
	int num_boxes = batch_splits(b);// == 1000

	Eigen::Map<const caffe2::ERArrXXf> scores(
		tscores + offset * bottom[0]->shape(1),//tscores.dim(1),
		num_boxes,
		bottom[0]->shape(1));//tscores.dim(1));
	Eigen::Map<const caffe2::ERArrXXf> boxes(
		tboxes + offset * bottom[1]->shape(1),// tboxes.dim(1),
		num_boxes,
		bottom[1]->shape(1));//tboxes.dim(1));


// To store updated scores if SoftNMS is used
	caffe2::ERArrXXf soft_nms_scores(num_boxes, bottom[0]->shape(1));//tscores.dim(1));
	vector<vector<int>> keeps(num_classes);

	///////////////////////// 1. Perform nms to each class /////////////////////////////////
	// skip j = 0, because it's the background class

	int total_keep_count = 0;

	for (int j = 1; j < num_classes; j++) {
		auto cur_scores = scores.col(j);

		auto inds = caffe2::utils::GetArrayIndices(cur_scores > score_thresh_);
		auto cur_boxes = boxes.block(0, j * box_dim, boxes.rows(), box_dim);

		if (soft_nms_enabled_) {
			auto cur_soft_nms_scores = soft_nms_scores.col(j);
			keeps[j] = caffe2::utils::soft_nms_cpu(
				&cur_soft_nms_scores,
				cur_boxes,
				cur_scores,
				inds,
				soft_nms_sigma_,
				nms_thresh_,
				soft_nms_min_score_thresh_,
				soft_nms_method_);
		}
		else {
			std::sort(
				inds.data(),
				inds.data() + inds.size(),
				[&cur_scores](int lhs, int rhs) {
					return cur_scores(lhs) > cur_scores(rhs);
				});
			keeps[j] = caffe2::utils::nms_cpu(cur_boxes, cur_scores, inds, nms_thresh_);
		}
		total_keep_count += keeps[j].size();
		//vector<int> cur_keeps=keeps[j];
		//for(int i=0; i<cur_keeps.size();i++)
		//    printf("cur_keeps[i] : %d\n",cur_keeps[i]);
		//printf("cur_keeps.size() : %d\n",cur_keeps.size());
	}
	printf("\ntotal_keep_count after step 1: %d\n", total_keep_count);//996
	if (soft_nms_enabled_) {
		// Re-map scores to the updated SoftNMS scores
		new (&scores) Eigen::Map<const caffe2::ERArrXXf>(
			soft_nms_scores.data(),
			soft_nms_scores.rows(),
			soft_nms_scores.cols());
	}

	/////////////////////// 2. Limit to max_per_image detections *over all classes* ///////////////////////////
	if (detections_per_im_ > 0 && total_keep_count > detections_per_im_) { // detections_per_im_ == 100
		// merge all scores together and sort
		auto get_all_scores_sorted = [&scores, &keeps, total_keep_count]() {
			caffe2::EArrXf ret(total_keep_count);

			int ret_idx = 0;

			for (int i = 1; i < keeps.size(); i++) {
				auto& cur_keep = keeps[i];
				auto cur_scores = scores.col(i);
				auto cur_ret = ret.segment(ret_idx, cur_keep.size());
				caffe2::utils::GetSubArray(cur_scores, caffe2::utils::AsEArrXt(keeps[i]), &cur_ret);
				ret_idx += cur_keep.size();
			}
			std::sort(ret.data(), ret.data() + ret.size());
			return ret;
		};
		// Compute image thres based on all classes
		auto all_scores_sorted = get_all_scores_sorted();
		CHECK_GT(all_scores_sorted.size(), detections_per_im_);
		auto image_thresh = all_scores_sorted[all_scores_sorted.size() - detections_per_im_];

		total_keep_count = 0;
		// filter results with image_thresh
		for (int j = 1; j < num_classes; j++) {
			auto& cur_keep = keeps[j];
			auto cur_scores = scores.col(j);
			keeps[j] = filter_with_indices(
				cur_scores, cur_keep, [&image_thresh](float sc) {
					return sc >= image_thresh;
				});
			total_keep_count += keeps[j].size();
		}
	}
	printf("\ntotal_keep_count after step 2: %d\n", total_keep_count);//3

	//total_keep_per_batch[b] = total_keep_count;
	// Write results
	int cur_start_idx = top[0]->shape(0);
	int cur_out_idx = 0;
	float max_score = 0.0f;
	int   max_idx = 0;
	for (int j = 1; j < num_classes; j++) {
		auto  cur_scores = scores.col(j);
		auto  cur_boxes = boxes.block(0, j * box_dim, boxes.rows(), box_dim);
		auto& cur_keep = keeps[j]; // vector<vector<int>> keeps(num_classes);

		Eigen::Map<caffe2::EArrXf>  cur_out_scores(out_scores + cur_start_idx + cur_out_idx, cur_keep.size());
		Eigen::Map<caffe2::ERArrXXf> cur_out_boxes(out_boxes + (cur_start_idx + cur_out_idx) * box_dim, cur_keep.size(), box_dim);
		Eigen::Map<caffe2::EArrXf> cur_out_classes(out_classes + cur_start_idx + cur_out_idx, cur_keep.size());

		caffe2::utils::GetSubArray(cur_scores, caffe2::utils::AsEArrXt(cur_keep), &cur_out_scores);
		caffe2::utils::GetSubArrayRows(cur_boxes, caffe2::utils::AsEArrXt(cur_keep), &cur_out_boxes);
		if (0) {
			printf("\n");
			for (int i = 0; i < scores.rows(); i++) {
				if (scores.col(1)[i] > 0.6f)
					printf("scores[%d] : %.2f box : [%.2f %.2f %.2f %.2f]\n", i, scores.col(1)[i], boxes.col(4)[i], boxes.col(5)[i], boxes.col(6)[i], boxes.col(7)[i]);

			}
			for (int k = 0; k < cur_out_scores.size(); k++) {
				if (cur_out_scores.data()[k] > max_score) {
					max_score = cur_out_scores.data()[k];
					max_idx = k;
				}
			}

			printf("max_score :%.3f max_idx: %d\n", max_score, max_idx);
			for (int k = 0; k < cur_out_scores.size(); k++) {
				out_scores[k] = cur_out_scores.data()[k];
			}
			for (int i = 0; i < cur_boxes.rows(); i++) {
				printf("cur_boxes[%d]: [%.2f %.2f %.2f %.2f]\n", i, cur_boxes.col(0)[i], cur_boxes.col(1)[i], cur_boxes.col(2)[i], cur_boxes.col(3)[i]);
			}
			printf("max box : [%.2f %.2f %.2f %.2f] \n", out_boxes[4 * max_idx], out_boxes[4 * max_idx + 1], out_boxes[4 * max_idx + 2], out_boxes[4 * max_idx + 3]);
		}


		for (int i = 0; i < cur_keep.size(); i++) {
			printf("cur_keep[%d]: %d\n", i, cur_keep[i]);
			printf("cur_scores[%d]: %.2f\n", i, cur_scores[cur_keep[i]]);
			out_boxes[5 * i] = b;
			out_boxes[5 * i + 1] = cur_boxes.col(0)[cur_keep[i]];
			out_boxes[5 * i + 2] = cur_boxes.col(1)[cur_keep[i]];
			out_boxes[5 * i + 3] = cur_boxes.col(2)[cur_keep[i]];
			out_boxes[5 * i + 4] = cur_boxes.col(3)[cur_keep[i]];
		}
		max_score = 0.0f;
		max_idx = 0;
	}// end for (int j = 1; j < num_classes; j++) 
	offset += num_boxes;

	vector<int> out_scores_shape;
	vector<int> out_boxes_shape;
	vector<int> out_classes_shape;

	out_scores_shape.push_back(total_keep_count);
	out_scores_shape.push_back(1);
	top[0]->Reshape(out_scores_shape);

	out_boxes_shape.push_back(total_keep_count);
	out_boxes_shape.push_back(5);
	top[1]->Reshape(out_boxes_shape);

	out_classes_shape.push_back(total_keep_count);
	out_classes_shape.push_back(1);
	top[2]->Reshape(out_classes_shape);
	for (int i = 0; i < total_keep_count; i++)
		printf("\nout_boxes: [%.2f, %.2f, %.2f, %.2f, %.2f]\n", out_boxes[5 * i], out_boxes[5 * i + 1], out_boxes[5 * i + 2], out_boxes[5 * i + 3], out_boxes[5 * i + 4]);

}// end for (int b = 0; b < batch_splits.size(); ++b)

printf("\n==============================================BoxWithNMSLimitLayer Done=====================================\n");
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