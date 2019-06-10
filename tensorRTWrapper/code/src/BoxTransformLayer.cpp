#include <cub/cub/cub.cuh>
#include "generate_proposals_op_util_nms_gpu.h"
#include "common_gpu.h"
#include "Utils.h"

#include <algorithm>
#include <cfloat>
#include <vector> 
#include "bbox_transform_layer.hpp"
#include "BoxTransformLayer.h"
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

    template <typename Dtype>
	int BoxTransformLayerPlugin::initialize()
	{

        m_inputTotalCount = mRpnRoisH*mRpnRoisW+mBoxPredH*mBoxPredW+mIminfoH*mIminfoW;
		CUDA_CHECK(cudaHostAlloc(&mInputBuffer, m_inputTotalCount * sizeof(float), cudaHostAllocDefault));

        m_ouputTotalCount = mBoxPredH*mBoxPredW;
		CUDA_CHECK(cudaHostAlloc(&mOutputBuffer,m_ouputTotalCount * sizeof(float), cudaHostAllocDefault));


		return 0;
	}

	Dims BoxTransformLayerPlugin::getOutputDimensions(int index, const Dims * inputs, int nbInputDims)
	{
		assert(nbInputDims == 3);
		mRpnRoisH = inputs[0].d[0];//1000
        mRpnRoisW = inputs[0].d[1];//4

        mBoxPredH = inputs[1].d[0];//1000
        mBoxPredW = inputs[1].d[1];//8

        mIminfoH = inputs[2].d[0];//batch_size(1)
        mIminfoW = inputs[2].d[1];//3

        apply_scale_ = 0;
        correct_transform_coords_ = 1;

		return DimsCHW(mBoxPredH, mBoxPredW);//top[0]->Reshape(bottom[1]->shape());
	}

	template <typename Dtype>
	void BoxTransformLayerPlugin::forwardCpu( //const float *const * inputs,
                                              //      float * output,
                                              const Dtype * roi_in,//rpn_rois:    (1000,4)
                                              const Dtype * delta_in,//bbox_pred: (1000,8) bg x1 y1 x2 y2, human x1 y1 x2 y2
                                              const Dtype * iminfo_in,//batch_size(1),3
                                              Dtype* box_out_,// pred_bbox: (1000, 8)
                                              cudaStream_t stream){


        CUDA_CHECK(cudaStreamSynchronize(stream));
        int size = 0;
        float* inputData = (float*)mInputBuffer;
        int roisCount= 1000;
        size=mRpnRoisH*mRpnRoisW;//1000*4
        CUDA_CHECK(cudaMemcpyAsync(inputData, roi_in, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        inputData += size;

        size=mBoxPredH*mBoxPredW;//1000*8
        CUDA_CHECK(cudaMemcpyAsync(inputData, delta_in, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        inputData += size;

        size=mIminfoH*mIminfoW;//3
        CUDA_CHECK(cudaMemcpyAsync(inputData, iminfo_in, size * sizeof(float), cudaMemcpyDeviceToHost, stream));

        Dtype* box_out=mOutputBuffer;


        //for (int i = 0; i < min(bottom[0]->count(), 40); i += 4) {
        //    printf("roi_in[%d] : %.2f %.2f %.2f %.2f \n", i / 4, roi_in[i], roi_in[i + 1], roi_in[i + 2], roi_in[i + 3]);
        //}

        const int box_dim = 4;
        const int N = mRpnRoisH;//roisCount
        const int num_classes = mBoxPredW / box_dim; //delta_in.dim32(1) / box_dim;
        const int batch_size = 1;// (bottom[2]->shape(0));//iminfo_in.dim32(0);

        //CHECK_EQ(weights_.size(), 4);

        Eigen::Map<const caffe2::ERArrXXf> boxes0(roi_in, bottom[0]->shape(0), bottom[0]->shape(1));//(1000,4)---------------------------------
        Eigen::Map<const caffe2::ERArrXXf> deltas0(delta_in, bottom[1]->shape(0), bottom[1]->shape(1));//(1000,8)

        // Count the number of RoIs per batch
        vector<int> num_rois_per_batch(batch_size, 0);

        if (mRpnRoisW == box_dim) {//bottom[0]->shape(1) , roi_in.dim32(1)
            num_rois_per_batch[0] = N;//1000
        }
        else {
            const auto& roi_batch_ids = boxes0.col(0);
            for (int i = 0; i < roi_batch_ids.size(); ++i) {
                const int roi_batch_id = roi_batch_ids(i);
                if(roi_batch_id >= batch_size){break;}
                num_rois_per_batch[roi_batch_id]++;
            }
        }

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

            // TODO : assign clip boxes to top blob instead of eigen mat!
            //  / check clip_boxes function / check normalization is valid or not / check 800 x 800 input!
            clip_boxes.leftCols(4) *= scale_after;
            //new_boxes.block(offset, k * box_dim, num_rois, box_dim) = clip_boxes;
            printf("clip_boxes.rows(),clip_boxes.cols() : %ld %ld \n", clip_boxes.rows(), clip_boxes.cols());
            for (int j = 0; j < clip_boxes.rows(); j++) {// bg x1 y1 x2 y2, fg x1 y1 x2 y2

                box_out[num_classes * box_dim * j + 0 + (k)* box_dim] = clip_boxes.col(0)[j];
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
        CUDA_CHECK(cudaMemcpyAsync(box_out_, mOutputBuffer, sizeof(Dtype)* m_ouputTotalCount, cudaMemcpyHostToDevice, stream));
    }

	int BoxTransformLayerPlugin::enqueue(int batchSize,
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
					(const float*)inputs[2],
					(float*)outputs[0],
					stream);
				//forwardCpu((const float *const *)inputs,(float *)outputs[0],stream);
				break;
			case DataType::kHALF:
				forwardCpu<__half>((const __half*)inputs[0],
					(const __half*)inputs[1],
					(const __half*)inputs[2],
					(__half*)outputs[0],
					stream);
				break;
			case DataType::kINT8:
				forwardCpu<u_int8_t>((const u_int8_t*)inputs[0],
					(const u_int8_t*)inputs[1],
					(const u_int8_t*)inputs[2],
					(u_int8_t*)outputs[0],
					stream);
				break;
			default:
				std::cerr << "error data type" << std::endl;
			}

			return 0;
		};

	} 