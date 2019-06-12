#include "generate_proposals_op_util_nms_gpu.h"
#include "../../../include/common_gpu.h"
#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>
#include "BoxWithNMSLimitLayer.h"
#include "cuda_runtime.h"
//#include "bbox_with_nms_limit_layer.hpp"
#include "../../../include/caffe2/utils/eigen_utils.h"
#include "generate_proposals_op_util_boxes.hpp"
#include "../../../include/caffe2/utils/generate_proposals_op_util_nms.h"
#include <fp16.h>
using std::max;
using std::min;
using std::floor;
using std::ceil;
using namespace std;

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
	BoxWithNMSLimitLayerPlugin::BoxWithNMSLimitLayerPlugin(const int cudaThread /*= 512*/) :
		mThreadCount(cudaThread) {
	}
	BoxWithNMSLimitLayerPlugin::~BoxWithNMSLimitLayerPlugin() {
		if (mInputBuffer)
			CUDA_CHECK(cudaFreeHost(mInputBuffer));
		if (mOutputBuffer)
			CUDA_CHECK(cudaFreeHost(mOutputBuffer));
	}
	// create the plugin at runtime from a byte stream
	BoxWithNMSLimitLayerPlugin::BoxWithNMSLimitLayerPlugin(const void* data, size_t length) {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d; 
        read(d, mDataType);
        read(d, mThreadCount);

        read(d, mClsProbH);
        read(d, mClsProbW);
        read(d, mPredBoxH);
        read(d, mPredBoxW);

        read(d, m_inputTotalCount);
        read(d, m_ouputTotalCount);

        //std::cout << "read:" << a << " " << mOutputWidth<< " " <<mOutputHeight<<std::endl;
        assert(d == a + length);
	}

   
	void BoxWithNMSLimitLayerPlugin::serialize(void* buffer)
	{ 
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mDataType);
        write(d, mThreadCount);

        write(d, mClsProbH);
        write(d, mClsProbW);
        write(d, mPredBoxH);
        write(d, mPredBoxW);

        write(d, m_inputTotalCount);
        write(d, m_ouputTotalCount);

        //std::cout << "write:" << a << " " << mOutputHeight<< " " <<mOutputWidth<<std::endl;
        assert(d == a + getSerializationSize());
	}

    void BoxWithNMSLimitLayerPlugin::configureWithFormat( const Dims* inputDims, int nbInputs,
                                                          const Dims* outputDims, int nbOutputs,
                                                          DataType type,
                                                          PluginFormat format, int maxBatchSize){
        //std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
        assert((type == DataType::kFLOAT || type == DataType::kHALF ||
                type == DataType::kINT8) && format == PluginFormat::kNCHW);
        mDataType = type;
        //std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " "
        //<<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
    }

	size_t BoxWithNMSLimitLayerPlugin::getSerializationSize()
	{
        return 0;
	}

	int BoxWithNMSLimitLayerPlugin::initialize(){
		int totalCount = 0;

		totalCount += mClsProbH*mClsProbW; // bottom: "cls_prob", "pred_bbox"
        totalCount += mPredBoxH*mPredBoxW;
        CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

		totalCount = 300*(1+5+1);//score_nms+bbox_nms+class_nms ==  (300x1+300x5+300x1) count
        CUDA_CHECK(cudaHostAlloc(&mOutputBuffer,totalCount * sizeof(float), cudaHostAllocDefault));

		return 0;
	}

	Dims BoxWithNMSLimitLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 2);// bottom: "cls_prob", "pred_bbox"

        m_nms_max_count = 300; // tscores.dim(1);

		mClsProbH = inputs[0].d[0];//1000
		mClsProbW = inputs[0].d[1];//2

        mPredBoxH = inputs[1].d[0];//1000
        mPredBoxW = inputs[1].d[1];//8

        if (index == 0){
		    return DimsHW(m_nms_max_count, 1);}//score_nms shape
        else if (index == 1){
            return DimsHW(m_nms_max_count, 5);}//bbox_nms shape
        else if (index == 2){
            return DimsHW(m_nms_max_count, 1);}//class_nms shape
	}

	template <typename Dtype>
	void BoxWithNMSLimitLayerPlugin::forwardCpu(//const float *const * inputs,
                                       //     float * output,
                                       const Dtype * tscores,//cls_prob, 1000, 2
                                       const Dtype * tboxes,// pred_bbox,1000, 8 <- from bbox_transform layer
                                       Dtype* score_nms,// 300,1
                                       Dtype* bbox_nms,//  300,5
                                       Dtype* class_nms,// 300,1
                                       cudaStream_t stream){

        CUDA_CHECK(cudaStreamSynchronize(stream));
        int size = 0;
        Dtype* inputData = (Dtype*)mInputBuffer;
        size=mClsProbH*mClsProbW;//1000*2
        CUDA_CHECK(cudaMemcpyAsync(inputData, (const void*)tscores, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        inputData += size;

        size=mPredBoxH*mPredBoxW;//1000*8
        CUDA_CHECK(cudaMemcpyAsync(inputData, (const void*)tboxes, size * sizeof(float), cudaMemcpyDeviceToHost, stream));

        Dtype* out_scores  = (Dtype*)mOutputBuffer;          //(top[0]->mutable_cpu_data()); //score_nms blob
        Dtype* out_boxes   = out_scores +   m_nms_max_count; //(top[1]->mutable_cpu_data()); //bbox_nms  blob
        Dtype* out_classes = out_boxes  + 5*m_nms_max_count; //(top[2]->mutable_cpu_data()); //class_nms blob

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
        const int batch_size = 1;
        const int box_dim = 4;// rotated_ ? 5 :
        const int N = mClsProbH;//bottom[0]->shape(0);
        const int num_classes = mClsProbW;// bottom[0]->shape(1);// tscores.dim(1);
        // tscores: (num_boxes, num_classes), 0 for background

        if(mClsProbH != mPredBoxH) printf("mClsProbH != mPredBoxH, mClsProbH: %d , mPredBoxH: %d \n", mClsProbH, mPredBoxH);
        if(num_classes* box_dim != mPredBoxW) printf("num_classes* box_dim != mPredBoxW, num_classes: %d box_dim: %d mPredBoxW: %d \n",num_classes, box_dim, mPredBoxW);

        vector<float> batch_splits_default(1, mClsProbH);//tscores.dim(0) == 1,1000
        const float* batch_splits_data = batch_splits_default.data();

        Eigen::Map<const caffe2::EArrXf> batch_splits(batch_splits_data, batch_size);
        if(batch_splits.sum() != N) printf("batch_splits.sum() != N, batch_splits.sum(): %d , N: %d \n", batch_splits.sum(), N);
        //vector<int> total_keep_per_batch(batch_size);
        int offset = 0;
        int final_nms_count = 0;
        for (int b = 0; b < batch_splits.size(); ++b) {// size == 1
            int num_boxes = batch_splits(b);// == 1000

            Eigen::Map<const caffe2::ERArrXXf> scores((float*)tscores + offset * mClsProbW ,//tscores.dim(1),
                                                      num_boxes,
                                                      mClsProbW);//tscores.dim(1));

            Eigen::Map<const caffe2::ERArrXXf> boxes( (float*)tboxes + offset * mPredBoxW,// tboxes.dim(1),
                                                      num_boxes,
                                                      mPredBoxW);//tboxes.dim(1));

            // To store updated scores if SoftNMS is used
            caffe2::ERArrXXf soft_nms_scores(num_boxes, mClsProbW);//tscores.dim(1));
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
                    std::sort( inds.data(),
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
                new (&scores) Eigen::Map<const caffe2::ERArrXXf>( soft_nms_scores.data(),
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

            int cur_start_idx =m_nms_max_count;//top[0]->shape(0);
            int cur_out_idx = 0;
            float max_score = 0.0f;
            int   max_idx = 0;
            for (int j = 1; j < num_classes; j++) {
                auto  cur_scores = scores.col(j);
                auto  cur_boxes = boxes.block(0, j * box_dim, boxes.rows(), box_dim);
                auto& cur_keep = keeps[j]; // vector<vector<int>> keeps(num_classes);

                Eigen::Map<caffe2::EArrXf>  cur_out_scores((float*)out_scores + cur_start_idx + cur_out_idx, cur_keep.size());
                Eigen::Map<caffe2::ERArrXXf> cur_out_boxes((float*)out_boxes + (cur_start_idx + cur_out_idx) * box_dim, cur_keep.size(), box_dim);
                Eigen::Map<caffe2::EArrXf> cur_out_classes((float*)out_classes + cur_start_idx + cur_out_idx, cur_keep.size());

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

                float onef{1.0f}, zerof{0.0f};
                __half oneh = fp16::__float2half(1.0f),
                       zeroh = fp16::__float2half(0.0f);

                for (int i = 0; i < cur_keep.size(); i++) {
                    printf("cur_keep[%d]: %d\n", i, cur_keep[i]);
                    printf("cur_scores[%d]: %.2f\n", i, cur_scores[cur_keep[i]]);
                    if(mDataType==DataType::kFLOAT)
                        out_boxes[5 * i] = 0.0f;//batch index 'b'==0
                    else if(mDataType==DataType::kHALF)
                        out_boxes[5 * i] = __float2half(0);//batch index 'b'==0
                    else if(mDataType==DataType::kINT8)
                        out_boxes[5 * i] = __half2float(zeroh);//batch index 'b'==0 //TODO
                    out_boxes[5 * i + 1] = (Dtype)cur_boxes.col(0)[cur_keep[i]];
                    out_boxes[5 * i + 2] = (Dtype)cur_boxes.col(1)[cur_keep[i]];
                    out_boxes[5 * i + 3] = (Dtype)cur_boxes.col(2)[cur_keep[i]];
                    out_boxes[5 * i + 4] = (Dtype)cur_boxes.col(3)[cur_keep[i]];

                }
                max_score = 0.0f;
                max_idx = 0;
            }// end for (int j = 1; j < num_classes; j++)
            offset += num_boxes;

            for (int i = 0; i < total_keep_count; i++)
                printf("\nout_boxes: [%.2f, %.2f, %.2f, %.2f, %.2f]\n", out_boxes[5 * i], out_boxes[5 * i + 1], out_boxes[5 * i + 2], out_boxes[5 * i + 3], out_boxes[5 * i + 4]);
        }// end for (int b = 0; b < batch_splits.size(); ++b)
        printf("\n==============================================BoxWithNMSLimitLayer Done=====================================\n");

        //CUDA_CHECK(cudaMemcpyAsync(bbox_nms, mOutputBuffer+m_nms_max_count, sizeof(Dtype)* m_nms_max_count * 5, cudaMemcpyHostToDevice, stream));
    }


    int BoxWithNMSLimitLayerPlugin::enqueue(int batchSize,
                                   const void* const* inputs,
                                   void** outputs,
                                   void* workspace,
                                   cudaStream_t stream) {

        assert(batchSize == 1);

        switch (mDataType)
        {
        case DataType::kFLOAT:
            forwardCpu<float>((const float*)inputs[0],
                (const float*)inputs[1],
                (float*)outputs[0],
                (float*)outputs[1],
                (float*)outputs[2],
                stream);
            //forwardCpu((const float *const *)inputs,(float *)outputs[0],stream);
            break;
        case DataType::kHALF:
            forwardCpu<__half>((const __half*)inputs[0],
                (const __half*)inputs[1],
                (__half*)outputs[0],
                (__half*)outputs[1],
                (__half*)outputs[2],
                stream);
            break;
        case DataType::kINT8:
            forwardCpu<u_int8_t>((const u_int8_t*)inputs[0],
                (const u_int8_t*)inputs[1],
                (u_int8_t*)outputs[0],
                (u_int8_t*)outputs[1],
                (u_int8_t*)outputs[2],
                stream);
            break;
        default:
            std::cerr << "error data type" << std::endl;
        }

        return 0;
    };

}