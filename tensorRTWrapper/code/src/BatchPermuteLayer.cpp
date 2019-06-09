//#include "BatchPermutationConfigs.h"
#include "BatchPermutationLayer.h"
#include <cub/cub/cub.cuh>
#include "generate_proposals_op_util_nms_gpu.h"
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
        /*
        int totalCount = 0;
        for(const auto& yolo : mBatchPermutationKernel)
            totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
        CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

        totalCount = 0;//detection count
        for(const auto& yolo : mBatchPermutationKernel)
            totalCount += yolo.width*yolo.height * CHECK_COUNT;
        CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection),
                                                                              cudaHostAllocDefault));
        */

        return 0;
    }
    
    Dims BatchPermutationLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims==4);
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

        return index==0 ? DimsCHW(300, 5):DimsCHW(300, 1);
    }
	    
	  
    
    template <typename Dtype>
    void BatchPermutationLayerPlugin::forwardGpu(//const float *const * inputs,
                                                 //      float * output,
                                                 const Dtype * scores,
                                                 const Dtype * bbox_deltas,
                                                 const Dtype * im_info_tensor,
                                                 const Dtype * anchors,
                                                 Dtype* out_rois,
                                                 Dtype* out_rois_probs,
                                                 cudaStream_t stream){
        /*
        int numElem;
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

        //first detect count init 0
        CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
        for (unsigned int i = 0;i< mBatchPermutationKernel.size();++i)
        {
            const auto& yolo = mBatchPermutationKernel[i];
            numElem = yolo.width*yolo.height;

            //copy anchor to device
	        CUDA_CHECK(cudaMemcpy(devAnchor,yolo.anchors,AnchorLen,cudaMemcpyHostToDevice));

            CalDetection<<< (yolo.width*yolo.height + mThreadCount - 1) / mThreadCount,
                             mThreadCount>>>
                                            (inputs[i],output, numElem, yolo.width, yolo.height,
                                            (float *)devAnchor, mClassCount);
        }
        CUDA_CHECK(cudaFree(devAnchor));*/
        /*
          assert(batchSize == 1);
          const int channels = mCHW.d[0];
          const int64_t in_height = mCHW.d[1];
          const int64_t in_width = mCHW.d[2];
          const int64_t out_height = mOutputHeight;
          const int64_t out_width = mOutputWidth;
          int totalElems = batchSize * in_height * in_width * channels;
          
          // Handle no-op resizes efficiently.
          if (out_height == in_height && out_width == in_width) {
              CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], totalElems * type2size(mDataType),
              cudaMemcpyDeviceToDevice, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
              return 0;
          }
        */

        /*
        spatial_scale: 1/4 1/8 1/16 1/32 1/64
        nms_thresh: 0.699999988079071
        pre_nms_topn: 1000
        min_size: 16.0
        post_nms_topn: 1000
        correct_transform_coords: 1*/

        //CAFFE_ENFORCE_EQ(scores.ndim(), 4, scores.ndim());
        //CAFFE_ENFORCE(scores.template IsType<float>(), scores.meta().name());

        //const auto num_images = scores.dim(0);
        //const auto A = scores.dim(1);
        //const auto H = scores.dim(2);
        //const auto W = scores.dim(3);
        //const auto box_dim = anchors.dim(1);

        //CAFFE_ENFORCE(box_dim == 4 || box_dim == 5);

        int A=mScoreC;
        int H=mScoreH;
        int W=mScoreW;
        const int K = H * W;
        const int conv_layer_nboxes = K * A;
        // Getting data members ready

        // We'll sort the scores
        // we want to remember their original indexes,
        // ie their indexes in the tensor of shape (num_images,A,K)
        // from the conv layer
        // each row of d_conv_layer_indexes is at first initialized to 1..A*K
        dev_conv_layer_indexes_.Resize(num_images, conv_layer_nboxes);
        int* d_conv_layer_indexes = dev_conv_layer_indexes_.template mutable_data<int>();

        // d_image_offset[i] = i*K*A for i from 1 to num_images+1
        // Used by the segmented sort to only sort scores within one image
        dev_image_offset_.Resize(num_images + 1);
        int* d_image_offset = dev_image_offset_.template mutable_data<int>();

        // The following calls to CUB primitives do nothing
        // (because the first arg is nullptr)
        // except setting cub_*_temp_storage_bytes
        size_t cub_sort_temp_storage_bytes = 0;
        float* flt_ptr = nullptr;
        int* int_ptr = nullptr;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr,
                cub_sort_temp_storage_bytes,
                flt_ptr,
                flt_ptr,
                int_ptr,
                int_ptr,
                num_images * conv_layer_nboxes,
                num_images,
                int_ptr,
                int_ptr,
                0,
                8 * sizeof(float), // sort all bits
                stream);

        // Allocate temporary storage for CUB
        dev_cub_sort_buffer_.Resize(cub_sort_temp_storage_bytes);
        void* d_cub_sort_temp_storage =
                dev_cub_sort_buffer_.template mutable_data<char>();

        size_t cub_select_temp_storage_bytes = 0;
        char* char_ptr = nullptr;
        cub::DeviceSelect::Flagged(
                nullptr,
                cub_select_temp_storage_bytes,
                flt_ptr,
                char_ptr,
                flt_ptr,
                int_ptr,
                K * A,
                stream);

        // Allocate temporary storage for CUB
        dev_cub_select_buffer_.Resize(cub_select_temp_storage_bytes);
        void* d_cub_select_temp_storage =
                dev_cub_select_buffer_.template mutable_data<char>();

        // Initialize :
        // - each row of dev_conv_layer_indexes to 1..K*A
        // - each d_nboxes to 0
        // - d_image_offset[i] = K*A*i for i 1..num_images+1
        // 2D grid
        InitializeDataKernel<<<(CAFFE_GET_BLOCKS(A * K), num_images),
                                CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
                                0,
                                stream>>>(num_images, 
                                          conv_layer_nboxes, 
                                          d_image_offset, 
                                          d_conv_layer_indexes);

        // Sorting input scores
        dev_sorted_conv_layer_indexes_.Resize(num_images, conv_layer_nboxes);
        dev_sorted_scores_.Resize(num_images, conv_layer_nboxes);
        const float* d_in_scores = scores.data<float>();
        int* d_sorted_conv_layer_indexes =
                dev_sorted_conv_layer_indexes_.template mutable_data<int>();
        float* d_sorted_scores = dev_sorted_scores_.template mutable_data<float>();
        ;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
                d_cub_sort_temp_storage,
                cub_sort_temp_storage_bytes,
                d_in_scores,
                d_sorted_scores,
                d_conv_layer_indexes,
                d_sorted_conv_layer_indexes,
                num_images * conv_layer_nboxes,
                num_images,
                d_image_offset,
                d_image_offset + 1,
                0,
                8 * sizeof(float), // sort all bits
                stream);

        // Keeping only the topN pre_nms
        const int nboxes_to_generate = std::min(conv_layer_nboxes, rpn_pre_nms_topN_);

        // Generating the boxes associated to the topN pre_nms scores
        dev_boxes_.Resize(num_images, box_dim * nboxes_to_generate);
        dev_boxes_keep_flags_.Resize(num_images, nboxes_to_generate);
        const float* d_bbox_deltas = bbox_deltas.data<float>();
        const float* d_anchors = anchors.data<float>();
        const float* d_im_info_vec = im_info_tensor.data<float>();
        float* d_boxes = dev_boxes_.template mutable_data<float>();
        ;
        char* d_boxes_keep_flags =
                dev_boxes_keep_flags_.template mutable_data<char>();

        GeneratePreNMSUprightBoxesKernel<<< (CAFFE_GET_BLOCKS(nboxes_to_generate), num_images),
                CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
                0,
                stream>>>(  d_sorted_conv_layer_indexes,
                            nboxes_to_generate,
                            d_bbox_deltas,
                            reinterpret_cast<const float4*>(d_anchors),
                            H,
                            W,
                            A,
                            feat_stride_,
                            rpn_min_size_,
                            d_im_info_vec,
                            num_images,
                            utils::BBOX_XFORM_CLIP_DEFAULT,
                            reinterpret_cast<float4*>(d_boxes),
                            nboxes_to_generate,
                            d_sorted_scores,
                            d_boxes_keep_flags);

        const int nboxes_generated = nboxes_to_generate;

        float* d_image_prenms_boxes  = dev_image_prenms_boxes_.template mutable_data<float>();
        float* d_image_prenms_scores = dev_image_prenms_scores_.template mutable_data<float>();
        int* d_image_boxes_keep_list = dev_image_boxes_keep_list_.template mutable_data<int>();

        dev_image_prenms_boxes_.Resize(box_dim * nboxes_generated);
        dev_image_prenms_scores_.Resize(nboxes_generated);
        dev_image_boxes_keep_list_.Resize(nboxes_generated);

        const int roi_cols = box_dim + 1;
        const int max_postnms_nboxes = std::min(nboxes_generated, rpn_post_nms_topN_);

        dev_postnms_rois_.Resize(roi_cols * num_images * max_postnms_nboxes);
        dev_postnms_rois_probs_.Resize(num_images * max_postnms_nboxes);

        float* d_postnms_rois = dev_postnms_rois_.template mutable_data<float>();
        float* d_postnms_rois_probs =
                dev_postnms_rois_probs_.template mutable_data<float>();

        dev_prenms_nboxes_.Resize(num_images);
        host_prenms_nboxes_.Resize(num_images);

        int* d_prenms_nboxes = dev_prenms_nboxes_.template mutable_data<int>();
        int* h_prenms_nboxes = host_prenms_nboxes_.template mutable_data<int>();

        int nrois_in_output = 0;
        for (int image_index = 0; image_index < num_images; ++image_index) {
            // Sub matrices for current image
            const float* d_image_boxes =
                    &d_boxes[image_index * nboxes_generated * box_dim];
            const float* d_image_sorted_scores = &d_sorted_scores[image_index * K * A];
            char* d_image_boxes_keep_flags =
                    &d_boxes_keep_flags[image_index * nboxes_generated];

            float* d_image_postnms_rois = &d_postnms_rois[roi_cols * nrois_in_output];
            float* d_image_postnms_rois_probs = &d_postnms_rois_probs[nrois_in_output];

            // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
            // to the output tensors

            cub::DeviceSelect::Flagged(
                    d_cub_select_temp_storage,
                    cub_select_temp_storage_bytes,
                    reinterpret_cast<const float4*>(d_image_boxes),
                    d_image_boxes_keep_flags,
                    reinterpret_cast<float4*>(d_image_prenms_boxes),
                    d_prenms_nboxes,
                    nboxes_generated,
                    stream);
            cub::DeviceSelect::Flagged(
                    d_cub_select_temp_storage,
                    cub_select_temp_storage_bytes,
                    d_image_sorted_scores,
                    d_image_boxes_keep_flags,
                    d_image_prenms_scores,
                    d_prenms_nboxes,
                    nboxes_generated,
                    stream);

            host_prenms_nboxes_.CopyFrom(dev_prenms_nboxes_);

            // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
            // topN_prenms. Calling NMS on the generated boxes
            const int prenms_nboxes = *h_prenms_nboxes;
            int nkeep;
            utils::nms_gpu( d_image_prenms_boxes,
                            prenms_nboxes,
                            rpn_nms_thresh_,
                            d_image_boxes_keep_list,
                            &nkeep,
                            dev_nms_mask_,
                            host_nms_mask_,
                            &context_,
                            box_dim);
            }
            // All operations done after previous sort were keeping the relative order
            // of the elements the elements are still sorted keep topN <=> truncate the
            // array
            const int postnms_nboxes = std::min(nkeep, rpn_post_nms_topN_);

            // Moving the out boxes to the output tensors,
            // adding the image_index dimension on the fly
            WriteUprightBoxesOutput<<<
            CAFFE_GET_BLOCKS(postnms_nboxes),
                    CAFFE_CUDA_NUM_THREADS,
                    0,
                    stream>>>(
                            reinterpret_cast<const float4*>(d_image_prenms_boxes),
                                    d_image_prenms_scores,
                                    d_image_boxes_keep_list,
                                    postnms_nboxes,
                                    image_index,
                                    d_image_postnms_rois,
                                    d_image_postnms_rois_probs);


            nrois_in_output += postnms_nboxes;
        

        // Using a buffer because we cannot call ShrinkTo
        out_rois->Resize(nrois_in_output, roi_cols);
        out_rois_probs->Resize(nrois_in_output);
        float* d_out_rois = out_rois->template mutable_data<float>();
        float* d_out_rois_probs = out_rois_probs->template mutable_data<float>();

        CUDA_CHECK(cudaMemcpyAsync( d_out_rois,
                                    d_postnms_rois,
                                    nrois_in_output * roi_cols * sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    stream));

        CUDA_CHECK(cudaMemcpyAsync( d_out_rois_probs,
                                    d_postnms_rois_probs,
                                    nrois_in_output * sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    }


    int BatchPermutationLayerPlugin::enqueue(int batchSize,
                                             const void*const * inputs,
                                             void** outputs,
                                             void* workspace,
                                             cudaStream_t stream){
        assert(batchSize == 1);




		//TOP SHAPE == BOTTOM SHAPE (top[0]->Reshape(bottom[0]->shape())
		template <typename Dtype>
		void BatchPermutationLayer::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
			const vector<Blob<Dtype>*> & top) {
			const float* X = (const float*)bottom[0]->cpu_data();//Input(0); 4D // ex 1000, 256, 7,7
			const int* indices = (const int*)bottom[1]->cpu_data();//Input(0); 2D

			float* Y = (float*)(top[0]->mutable_cpu_data());// Output(0); 
			const int N = bottom[0]->shape(0);
			const int C = bottom[0]->shape(1);
			const int H = bottom[0]->shape(2);
			const int W = bottom[0]->shape(3);

			const float* src = X;
			float* dst = Y;

			for (int i = 0; i < N; i++) {
				int idx = indices[i];
				//if(idx>=1000 || idx<0)
				//    printf("out of index range(0,1000) : %d\n",idx);
				std::memcpy(dst + i * C * H * W, src + idx * C * H * W, sizeof(float) * C * H * W);
			}
			//"Y shape : %d, %d, %d, %d \n", top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
			// 1000, 256, 7,7



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
            case DataType::kFLOAT :
                forwardGpu<float>((const float *)inputs[0],
                                  (const float *)inputs[1],
                                  (const float *)inputs[2],
                                  (const float *)inputs[3],
                                  (float *)outputs[0],
                                  (float *)outputs[1],
                                  stream);
                //forwardGpu((const float *const *)inputs,(float *)outputs[0],stream);
                break;
            case DataType::kHALF:
                forwardGpu<__half>((const __half *)inputs[0],
                                   (const __half *)inputs[1],
                                   (const __half *)inputs[2],
                                   (const __half *)inputs[3],
                                   (__half *)outputs[0],
                                   (__half *)outputs[1],
                                   stream);
                break;
            case DataType::kINT8:
                forwardGpu<u_int8_t>((const u_int8_t *)inputs[0],
                                     (const u_int8_t *)inputs[1],
                                     (const u_int8_t *)inputs[2],
                                     (const u_int8_t *)inputs[3],
                                     (u_int8_t *)outputs[0],
                                     (u_int8_t *)outputs[1],
                                     stream);
                break;
            default:
                std::cerr << "error data type" << std::endl;
        }
        
        return 0;
    };

}
/*

TEST(BatchPermutationsTest, TestRealDownSampledGPU) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("BatchPermutations");
  def.add_input("scores");
  def.add_input("bbox_deltas");
  def.add_input("im_info");
  def.add_input("anchors");
  def.add_output("rois");
  def.add_output("rois_probs");
  def.mutable_device_option()->set_device_type(PROTO_CUDA);
  const int img_count = 2;
  const int A = 2;
  const int H = 4;
  const int W = 5;

  vector<float> scores{
      5.44218998e-03f, 1.19207997e-03f, 1.12379994e-03f, 1.17181998e-03f,
      1.20544003e-03f, 6.17993006e-04f, 1.05261997e-05f, 8.91025957e-06f,
      9.29536981e-09f, 6.09605013e-05f, 4.72735002e-04f, 1.13482002e-10f,
      1.50015003e-05f, 4.45032993e-06f, 3.21612994e-08f, 8.02662980e-04f,
      1.40488002e-04f, 3.12508007e-07f, 3.02616991e-06f, 1.97759000e-08f,
      2.66913995e-02f, 5.26766013e-03f, 5.05053019e-03f, 5.62100019e-03f,
      5.37420018e-03f, 5.26280981e-03f, 2.48894998e-04f, 1.06842002e-04f,
      3.92931997e-06f, 1.79388002e-03f, 4.79440019e-03f, 3.41609990e-07f,
      5.20430971e-04f, 3.34090000e-05f, 2.19159006e-07f, 2.28786003e-03f,
      5.16703985e-05f, 4.04523007e-06f, 1.79227004e-06f, 5.32449000e-08f};
  vector<float> bbx{
      -1.65040009e-02f, -1.84051003e-02f, -1.85930002e-02f, -2.08263006e-02f,
      -1.83814000e-02f, -2.89172009e-02f, -3.89706008e-02f, -7.52277970e-02f,
      -1.54091999e-01f, -2.55433004e-02f, -1.77490003e-02f, -1.10340998e-01f,
      -4.20190990e-02f, -2.71421000e-02f, 6.89801015e-03f,  5.71171008e-02f,
      -1.75665006e-01f, 2.30021998e-02f,  3.08554992e-02f,  -1.39333997e-02f,
      3.40579003e-01f,  3.91070992e-01f,  3.91624004e-01f,  3.92527014e-01f,
      3.91445011e-01f,  3.79328012e-01f,  4.26631987e-01f,  3.64892989e-01f,
      2.76894987e-01f,  5.13985991e-01f,  3.79999995e-01f,  1.80457994e-01f,
      4.37402993e-01f,  4.18545991e-01f,  2.51549989e-01f,  4.48318988e-01f,
      1.68564007e-01f,  4.65440989e-01f,  4.21891987e-01f,  4.45928007e-01f,
      3.27155995e-03f,  3.71480011e-03f,  3.60032008e-03f,  4.27092984e-03f,
      3.74579988e-03f,  5.95752988e-03f,  -3.14473989e-03f, 3.52022005e-03f,
      -1.88564006e-02f, 1.65188999e-03f,  1.73791999e-03f,  -3.56074013e-02f,
      -1.66615995e-04f, 3.14146001e-03f,  -1.11830998e-02f, -5.35363983e-03f,
      6.49790000e-03f,  -9.27671045e-03f, -2.83346009e-02f, -1.61233004e-02f,
      -2.15505004e-01f, -2.19910994e-01f, -2.20872998e-01f, -2.12831005e-01f,
      -2.19145000e-01f, -2.27687001e-01f, -3.43973994e-01f, -2.75869995e-01f,
      -3.19516987e-01f, -2.50418007e-01f, -2.48537004e-01f, -5.08224010e-01f,
      -2.28724003e-01f, -2.82402009e-01f, -3.75815988e-01f, -2.86352992e-01f,
      -5.28333001e-02f, -4.43836004e-01f, -4.55134988e-01f, -4.34897989e-01f,
      -5.65053988e-03f, -9.25739005e-04f, -1.06790999e-03f, -2.37016007e-03f,
      -9.71166010e-04f, -8.90910998e-03f, -1.17592998e-02f, -2.08992008e-02f,
      -4.94231991e-02f, 6.63906988e-03f,  3.20469006e-03f,  -6.44695014e-02f,
      -3.11607006e-03f, 2.02738005e-03f,  1.48096997e-02f,  4.39785011e-02f,
      -8.28424022e-02f, 3.62076014e-02f,  2.71668993e-02f,  1.38250999e-02f,
      6.76669031e-02f,  1.03252999e-01f,  1.03255004e-01f,  9.89722982e-02f,
      1.03646003e-01f,  4.79663983e-02f,  1.11014001e-01f,  9.31736007e-02f,
      1.15768999e-01f,  1.04014002e-01f,  -8.90677981e-03f, 1.13103002e-01f,
      1.33085996e-01f,  1.25405997e-01f,  1.50051996e-01f,  -1.13038003e-01f,
      7.01059997e-02f,  1.79651007e-01f,  1.41055003e-01f,  1.62841007e-01f,
      -1.00247003e-02f, -8.17587040e-03f, -8.32176022e-03f, -8.90108012e-03f,
      -8.13035015e-03f, -1.77263003e-02f, -3.69572006e-02f, -3.51580009e-02f,
      -5.92143014e-02f, -1.80795006e-02f, -5.46086021e-03f, -4.10550982e-02f,
      -1.83081999e-02f, -2.15411000e-02f, -1.17953997e-02f, 3.33894007e-02f,
      -5.29635996e-02f, -6.97528012e-03f, -3.15250992e-03f, -3.27355005e-02f,
      1.29676998e-01f,  1.16080999e-01f,  1.15947001e-01f,  1.21797003e-01f,
      1.16089001e-01f,  1.44875005e-01f,  1.15617000e-01f,  1.31586999e-01f,
      1.74735002e-02f,  1.21973999e-01f,  1.31596997e-01f,  2.48907991e-02f,
      6.18605018e-02f,  1.12855002e-01f,  -6.99798986e-02f, 9.58312973e-02f,
      1.53593004e-01f,  -8.75087008e-02f, -4.92327996e-02f, -3.32239009e-02f};
  vector<float> im_info{60, 80, 0.166667f};
  vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};

  // Doubling everything related to images, to simulate
  // num_images = 2
  scores.insert(scores.begin(), scores.begin(), scores.end());
  bbx.insert(bbx.begin(), bbx.begin(), bbx.end());
  im_info.insert(im_info.begin(), im_info.begin(), im_info.end());

  ERMatXf rois_gt(18, 5);
  rois_gt << 0, 0, 0, 79, 59, 0, 0, 5.0005703f, 51.6324f, 42.6950f, 0,
      24.13628387f, 7.51243401f, 79, 45.0663f, 0, 0, 7.50924301f, 67.4779f,
      45.0336, 0, 0, 23.09477997f, 50.61448669f, 59, 0, 0, 39.52141571f,
      51.44710541f, 59, 0, 23.57396317f, 29.98791885f, 79, 59, 0, 0,
      41.90219116f, 79, 59, 0, 0, 23.30098343f, 78.2413f, 58.7287f, 1, 0, 0, 79,
      59, 1, 0, 5.0005703f, 51.6324f, 42.6950f, 1, 24.13628387f, 7.51243401f,
      79, 45.0663f, 1, 0, 7.50924301f, 67.4779f, 45.0336, 1, 0, 23.09477997f,
      50.61448669f, 59, 1, 0, 39.52141571f, 51.44710541f, 59, 1, 23.57396317f,
      29.98791885f, 79, 59, 1, 0, 41.90219116f, 79, 59, 1, 0, 23.30098343f,
      78.2413f, 58.7287f;

  vector<float> rois_probs_gt{2.66913995e-02f,
                              5.44218998e-03f,
                              1.20544003e-03f,
                              1.19207997e-03f,
                              6.17993006e-04f,
                              4.72735002e-04f,
                              6.09605013e-05f,
                              1.50015003e-05f,
                              8.91025957e-06f};

  // Doubling everything related to images, to simulate
  // num_images = 2
  rois_probs_gt.insert(
      rois_probs_gt.begin(), rois_probs_gt.begin(), rois_probs_gt.end());

  AddInput<CUDAContext>(
      vector<int64_t>{img_count, A, H, W}, scores, "scores", &ws);
  AddInput<CUDAContext>(
      vector<int64_t>{img_count, 4 * A, H, W}, bbx, "bbox_deltas", &ws);
  AddInput<CUDAContext>(vector<int64_t>{img_count, 3}, im_info, "im_info", &ws);
  AddInput<CUDAContext>(vector<int64_t>{A, 4}, anchors, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("pre_nms_topN", 6000));
  def.add_arg()->CopyFrom(MakeArgument("post_nms_topN", 300));
  def.add_arg()->CopyFrom(MakeArgument("nms_thresh", 0.7f));
  def.add_arg()->CopyFrom(MakeArgument("min_size", 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("correct_transform_coords", true));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  // test rois
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois_gpu = rois_blob->Get<TensorCUDA>();
  Tensor rois{CPU};
  rois.CopyFrom(rois_gpu);

  EXPECT_EQ(rois.sizes(), (vector<int64_t>{rois_gt.rows(), rois_gt.cols()}));
  auto rois_data =
      Eigen::Map<const ERMatXf>(rois.data<float>(), rois.dim(0), rois.dim(1));
  EXPECT_NEAR((rois_data.matrix() - rois_gt).cwiseAbs().maxCoeff(), 0, 1e-4);

  // test rois_probs
  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs_gpu = rois_probs_blob->Get<TensorCUDA>();
  Tensor rois_probs{CPU};
  rois_probs.CopyFrom(rois_probs_gpu);
  EXPECT_EQ(
      rois_probs.sizes(), (vector<int64_t>{int64_t(rois_probs_gt.size())}));
  auto rois_probs_data =
      ConstEigenVectorArrayMap<float>(rois_probs.data<float>(), rois.dim(0));
  EXPECT_NEAR(
      (rois_probs_data.matrix() - utils::AsEArrXt(rois_probs_gt).matrix())
          .cwiseAbs()
          .maxCoeff(),
      0,
      1e-4);
}  

*/
