#define CUB_STDERR
#include "GenerateProposalLayer.h"
//#include <../../../include/cub/cub/cub.cuh>
#include "../../../include/cub/cub/device/dispatch/dispatch_radix_sort.cuh"
#include "../../../include/cub/cub/util_arch.cuh"
#include "../../../include/cub/cub/util_namespace.cuh"
#include ""
//#include "generate_proposals_op_util_nms_gpu.h"
#include "../../../include/common_gpu.h"
#include "Utils.h"

/*
static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// stuff we know about the network and the caffe input/output blobs

static const int INPUT_C = 3;
static const int INPUT_H = 375;
static const int INPUT_W = 500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
static int gUseDLACore{-1};

const std::string CLASSES[OUTPUT_CLS_SIZE]{"background", "aeroplane"};


static const int INPUT_C = 3;
static const int INPUT_H = 1080;//?? 1080
static const int INPUT_W = 1920;// 1920??
*/
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 5+1;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
const std::string CLASSES[OUTPUT_CLS_SIZE]{"background", "person", "catcher", "pitcher", "simpan", "hitter" };

//CUDA_1D_KERNEL_LOOP(i, nboxes) {  
//CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
//CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {

const float BBOX_XFORM_CLIP_DEFAULT= log(1000.0 / 16.0);


namespace nvinfer1{
    GenerateProposalLayerPlugin::GenerateProposalLayerPlugin(const int cudaThread /*= 512*/):
                                                                               mThreadCount(cudaThread){
        /*mClassCount = CLASS_NUM;
        mGenerateProposalKernel.clear();
        mGenerateProposalKernel.push_back(yolo1);
        mGenerateProposalKernel.push_back(yolo2);
        mGenerateProposalKernel.push_back(yolo3);

        mKernelCount = mGenerateProposalKernel.size();*/
    }
    GenerateProposalLayerPlugin::~GenerateProposalLayerPlugin(){
        if(mInputBuffer)
            CUDA_CHECK(cudaFreeHost(mInputBuffer));
        if(mOutputBuffer)
            CUDA_CHECK(cudaFreeHost(mOutputBuffer));
    }
    // create the plugin at runtime from a byte stream
    GenerateProposalLayerPlugin::GenerateProposalLayerPlugin(const void* data, size_t length){

    }

    void GenerateProposalLayerPlugin::serialize(void* buffer)
    {
    }
    
    size_t GenerateProposalLayerPlugin::getSerializationSize()
    {  
        return 0;
    }

    int GenerateProposalLayerPlugin::initialize()
    {
        return 0;
    }
    
    Dims GenerateProposalLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
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

        mBoxDeltaC = inputs[1].d[0];
        mBoxDeltaH = inputs[1].d[1];
        mBoxDeltaW = inputs[1].d[2];

        return index==0 ? DimsHW(300, 5):DimsHW(300, 1);
    }


    __global__ void InitializeDataKernel(
            const int num_images,
            const int KA,
            int* d_image_offsets,
            int* d_boxes_keys_iota) {
        CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {
            d_boxes_keys_iota[img_idx * KA + box_idx] = box_idx;

            // One 1D line sets the 1D data
            if (box_idx == 0) {
                d_image_offsets[img_idx] = KA * img_idx;
                // One thread sets the last+1 offset
                if (img_idx == 0)
                    d_image_offsets[num_images] = KA * num_images;
            }
        }
    }
    template <typename Dtype>
    __global__ void GeneratePreNMSUprightBoxesKernel(
            const int* d_sorted_scores_keys,
            const int nboxes_to_generate,
            const Dtype* d_bbox_deltas,
            const float4* d_anchors,
            const int H,
            const int W,
            const int A,
            const float feat_stride,
            const int min_size,
            const Dtype* d_img_info_vec,
            const int num_images,
            const float bbox_xform_clip,
            float4* d_out_boxes,
            const int prenms_nboxes, // leading dimension of out_boxes
            Dtype * d_inout_scores,
            char* d_boxes_keep_flags) {
        const int K = H * W;
        const int KA = K * A;
        CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
            // box_conv_index : # of the same box, but indexed in
            // the scores from the conv layer, of shape (A,H,W)
            // the num_images dimension was already removed
            // box_conv_index = a*K + h*W + w
            const int box_conv_index = d_sorted_scores_keys[image_index * KA + ibox];

            // We want to decompose box_conv_index in (a,h,w)
            // such as box_conv_index = a*K + h*W + w
            // (avoiding modulos in the process)
            int remaining = box_conv_index;
            const int dA = K; // stride of A
            const int a = remaining / dA;
            remaining -= a * dA;
            const int dH = W; // stride of H
            const int h = remaining / dH;
            remaining -= h * dH;
            const int w = remaining; // dW = 1

            // Loading the anchor a
            // float4 is a struct with float x,y,z,w
            const float4 anchor = d_anchors[a];
            // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
            const float shift_w = feat_stride * w;
            float x1 = shift_w + anchor.x;
            float x2 = shift_w + anchor.z;
            const float shift_h = feat_stride * h;
            float y1 = shift_h + anchor.y;
            float y2 = shift_h + anchor.w;

            // TODO use fast math when possible

            // Deltas for that box
            // Deltas of shape (num_images,4*A,K)
            // We're going to compute 4 scattered reads
            // better than the alternative, ie transposing the complete deltas
            // array first
            int deltas_idx = image_index * (KA * 4) + a * 4 * K + h * W + w;
            const float dx = d_bbox_deltas[deltas_idx];
            // Stride of K between each dimension
            deltas_idx += K;
            const float dy = d_bbox_deltas[deltas_idx];
            deltas_idx += K;
            float dw = d_bbox_deltas[deltas_idx];
            deltas_idx += K;
            float dh = d_bbox_deltas[deltas_idx];

            // Upper bound on dw,dh
            dw = fmin(dw, bbox_xform_clip);
            dh = fmin(dh, bbox_xform_clip);

            // Applying the deltas
            float width = x2 - x1 + 1.0f;
            const float ctr_x = x1 + 0.5f * width;
            const float pred_ctr_x = ctr_x + width * dx; // TODO fuse madd
            const float pred_w = width * expf(dw);
            x1 = pred_ctr_x - 0.5f * pred_w;
            x2 = pred_ctr_x + 0.5f * pred_w - 1.0f;

            float height = y2 - y1 + 1.0f;
            const float ctr_y = y1 + 0.5f * height;
            const float pred_ctr_y = ctr_y + height * dy;
            const float pred_h = height * expf(dh);
            y1 = pred_ctr_y - 0.5f * pred_h;
            y2 = pred_ctr_y + 0.5f * pred_h - 1.0f;

            // Clipping box to image
            const float img_height = d_img_info_vec[3 * image_index + 0];
            const float img_width = d_img_info_vec[3 * image_index + 1];
            const Dtype min_size_scaled = (Dtype)1;//(Dtype)min_size * d_img_info_vec[3 * image_index + 2];
            x1 = fmax(fmin(x1, img_width - 1.0f), 0.0f);
            y1 = fmax(fmin(y1, img_height - 1.0f), 0.0f);
            x2 = fmax(fmin(x2, img_width - 1.0f), 0.0f);
            y2 = fmax(fmin(y2, img_height - 1.0f), 0.0f);

            // Filter boxes
            // Removing boxes with one dim < min_size
            // (center of box is in image, because of previous step)
            width = x2 - x1 + 1.0f; // may have changed
            height = y2 - y1 + 1.0f;
            bool keep_box = true;//(bool)(((const Dtype)fmin(width, height)) >= min_size_scaled);

            // We are not deleting the box right now even if !keep_box
            // we want to keep the relative order of the elements stable
            // we'll do it in such a way later
            // d_boxes_keep_flags size: (num_images,prenms_nboxes)
            // d_out_boxes size: (num_images,prenms_nboxes)
            const int out_index = image_index * prenms_nboxes + ibox;
            d_boxes_keep_flags[out_index] = keep_box;
            d_out_boxes[out_index] = {x1, y1, x2, y2};

            // d_inout_scores size: (num_images,KA)
            if (!keep_box)
                d_inout_scores[image_index * KA + ibox] = FLT_MIN; // for NMS
        }
    }
    template <typename Dtype>
    __global__ void WriteUprightBoxesOutput(
            const float4* d_image_boxes,
            const Dtype* d_image_scores,
            const int* d_image_boxes_keep_list,
            const int nboxes,
            const int image_index,
            Dtype* d_image_out_rois,
            Dtype* d_image_out_rois_probs) {
        CUDA_1D_KERNEL_LOOP(i, nboxes) {
            const int ibox = d_image_boxes_keep_list[i];
            const float4 box = d_image_boxes[ibox];
            const float score = d_image_scores[ibox];
            // Scattered memory accesses
            // postnms_nboxes is small anyway
            d_image_out_rois_probs[i] = static_cast<Dtype>(score);
            const int base_idx = 5 * i;
            d_image_out_rois[base_idx + 0] = static_cast<Dtype>(image_index);
            d_image_out_rois[base_idx + 1] = static_cast<Dtype>(box.x);
            d_image_out_rois[base_idx + 2] = static_cast<Dtype>(box.y);
            d_image_out_rois[base_idx + 3] = static_cast<Dtype>(box.z);
            d_image_out_rois[base_idx + 4] = static_cast<Dtype>(box.w);
        }
    }

    #define BOXES_PER_THREAD (8 * sizeof(int))
    #define CHUNK_SIZE 2000

    const dim3 CAFFE_CUDA_NUM_THREADS_2D = {
            static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMX),
            static_cast<unsigned int>(CAFFE_CUDA_NUM_THREADS_2D_DIMY),
            1u};
    struct
    #ifndef __HIP_PLATFORM_HCC__
                __align__(16)
    #endif
        Box {
            float x1, y1, x2, y2;
        };

    __launch_bounds__(CAFFE_CUDA_NUM_THREADS_2D_DIMX * CAFFE_CUDA_NUM_THREADS_2D_DIMY, 4)
    __global__ void NMSKernel(  const Box* d_desc_sorted_boxes,
                                const int nboxes,
                                const float thresh,
                                const bool legacy_plus_one,
                                const int mask_ld,
                                int* d_delete_mask) {
        // Storing boxes used by this CUDA block in the shared memory
        __shared__ Box shared_i_boxes[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
        // Same thing with areas
        __shared__ float shared_i_areas[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
        // The condition of the for loop is common to all threads in the block
        // This is necessary to be able to call __syncthreads() inside of the loop
        for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < nboxes;
             i_block_offset += blockDim.x * gridDim.x) {
            const int i_to_load = i_block_offset + threadIdx.x;
            if (i_to_load < nboxes) {
                // One 1D line load the boxes for x-dimension
                if (threadIdx.y == 0) {
                    const Box box = d_desc_sorted_boxes[i_to_load];
                    shared_i_areas[threadIdx.x] =
                            (box.x2 - box.x1 + float(int(legacy_plus_one))) *
                            (box.y2 - box.y1 + float(int(legacy_plus_one)));
                    shared_i_boxes[threadIdx.x] = box;
                }
            }
            __syncthreads();
            const int i = i_block_offset + threadIdx.x;
            for (int j_thread_offset =
                    BOXES_PER_THREAD * (blockIdx.y * blockDim.y + threadIdx.y);
                 j_thread_offset < nboxes;
                 j_thread_offset += BOXES_PER_THREAD * blockDim.y * gridDim.y) {
                // Note : We can do everything using multiplication,
                // and use fp16 - we are comparing against a low precision
                // threshold
                int above_thresh = 0;
                bool valid = false;
                for (int ib = 0; ib < BOXES_PER_THREAD; ++ib) {
                    // This thread will compare Box i and Box j
                    const int j = j_thread_offset + ib;
                    if (i < j && i < nboxes && j < nboxes) {
                        valid = true;
                        const Box j_box = d_desc_sorted_boxes[j];
                        const Box i_box = shared_i_boxes[threadIdx.x];
                        const float j_area =
                                (j_box.x2 - j_box.x1 + float(int(legacy_plus_one))) *
                                (j_box.y2 - j_box.y1 + float(int(legacy_plus_one)));
                        const float i_area = shared_i_areas[threadIdx.x];
                        // The following code will not be valid with empty boxes
                        if (i_area == 0.0f || j_area == 0.0f)
                            continue;
                        const float xx1 = fmaxf(i_box.x1, j_box.x1);
                        const float yy1 = fmaxf(i_box.y1, j_box.y1);
                        const float xx2 = fminf(i_box.x2, j_box.x2);
                        const float yy2 = fminf(i_box.y2, j_box.y2);

                        // fdimf computes the positive difference between xx2+1 and xx1
                        const float w = fdimf(xx2 + float(int(legacy_plus_one)), xx1);
                        const float h = fdimf(yy2 + float(int(legacy_plus_one)), yy1);
                        const float intersection = w * h;

                        // Testing for a/b > t
                        // eq with a > b*t (b is !=0)
                        // avoiding divisions
                        const float a = intersection;
                        const float b = i_area + j_area - intersection;
                        const float bt = b * thresh;
                        // eq. to if ovr > thresh
                        if (a > bt) {
                            // we have score[j] <= score[i]
                            above_thresh |= (1U << ib);
                        }
                    }
                }
                if (valid)
                    d_delete_mask[i * mask_ld + j_thread_offset / BOXES_PER_THREAD] =
                            above_thresh;
            }
            __syncthreads(); // making sure everyone is done reading smem
        }
    }
    template <typename Dtype>
    void nms_gpu_upright(
            const Dtype* d_desc_sorted_boxes_float_ptr,
            const int N,
            const float thresh,
            const bool legacy_plus_one,
            int* d_keep_sorted_list,
            int* h_nkeep,
            float* dev_delete_mask,
            float* host_delete_mask,
            cudaStream_t stream) {
        // Making sure we respect the __align(16)__ we promised to the compiler
        //auto iptr = reinterpret_cast<std::uintptr_t>(d_desc_sorted_boxes_float_ptr);
        //CAFFE_ENFORCE_EQ(iptr % 16, 0);

        // The next kernel expects squares
        //CAFFE_ENFORCE_EQ(
        //        CAFFE_CUDA_NUM_THREADS_2D_DIMX, CAFFE_CUDA_NUM_THREADS_2D_DIMY);

        const int mask_ld = (N + BOXES_PER_THREAD - 1) / BOXES_PER_THREAD;
        const Box* d_desc_sorted_boxes =
                reinterpret_cast<const Box*>(d_desc_sorted_boxes_float_ptr);

        //dev_delete_mask.Resize(N * mask_ld);
        int* d_delete_mask = (int*)(dev_delete_mask);//.template mutable_data<int>();

        NMSKernel<<<CAFFE_GET_BLOCKS_2D(N, mask_ld),
                    CAFFE_CUDA_NUM_THREADS_2D,    0,
                    stream>>>(
                        d_desc_sorted_boxes, N, thresh, legacy_plus_one, mask_ld, d_delete_mask);

        //host_delete_mask.Resize(N * mask_ld);
        int* h_delete_mask = (int*)host_delete_mask;//.template mutable_data<int>();

        // Overlapping CPU computes and D2H memcpy
        // both take about the same time
        cudaEvent_t copy_done;
        cudaEventCreate(&copy_done);
        int nto_copy = std::min(CHUNK_SIZE, N);
        CUDA_CHECK(cudaMemcpyAsync(
                &h_delete_mask[0],
                &d_delete_mask[0],
                nto_copy * mask_ld * sizeof(int),
                cudaMemcpyDeviceToHost,
                stream));
        CUDA_CHECK(cudaEventRecord(copy_done, stream));
        int offset = 0;
        std::vector<int> h_keep_sorted_list;
        std::vector<int> rmv(mask_ld, 0);
        while (offset < N) {
            const int ncopied = nto_copy;
            int next_offset = offset + ncopied;
            nto_copy = std::min(CHUNK_SIZE, N - next_offset);
            if (nto_copy > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                        &h_delete_mask[next_offset * mask_ld],
                        &d_delete_mask[next_offset * mask_ld],
                        nto_copy * mask_ld * sizeof(int),
                        cudaMemcpyDeviceToHost,
                        stream));
            }
            // Waiting for previous copy
            CUDA_CHECK(cudaEventSynchronize(copy_done));
            if (nto_copy > 0)
                cudaEventRecord(copy_done, stream);
            for (int i = offset; i < next_offset; ++i) {
                int iblock = i / BOXES_PER_THREAD;
                int inblock = i % BOXES_PER_THREAD;
                if (!(rmv[iblock] & (1 << inblock))) {
                    h_keep_sorted_list.push_back(i);
                    int* p = &h_delete_mask[i * mask_ld];
                    for (int ib = 0; ib < mask_ld; ++ib) {
                        rmv[ib] |= p[ib];
                    }
                }
            }
            offset = next_offset;
        }
        cudaEventDestroy(copy_done);

        const int nkeep = h_keep_sorted_list.size();
        cudaMemcpyAsync(d_keep_sorted_list,
                        &h_keep_sorted_list[0],
                        nkeep * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream);

        *h_nkeep = nkeep;
    }
    template <typename Dtype>
    void nms_gpu(
            const Dtype* d_desc_sorted_boxes,
            const int N,
            const float thresh,
            const bool legacy_plus_one,
            int* d_keep_sorted_list,
            int* h_nkeep,
            float*  dev_delete_mask,
            float* host_delete_mask,
            cudaStream_t stream) {
        nms_gpu_upright(
                d_desc_sorted_boxes,
                N,
                thresh,
                legacy_plus_one,
                d_keep_sorted_list,
                h_nkeep,
                dev_delete_mask,
                host_delete_mask,
                stream);
    }
    
    template <typename Dtype>
    void GenerateProposalLayerPlugin::forwardGpu(//const float *const * inputs,
                                                 //      float * output,
                                                 const Dtype * scores,
                                                 const Dtype * bbox_deltas,
                                                 const Dtype * im_info_tensor,
                                                 const Dtype * anchors,
                                                 Dtype* out_rois,
                                                 Dtype* out_rois_probs,
                                                 cudaStream_t stream) {
        /*
        int numElem;
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

        //first detect count init 0
        CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
        for (unsigned int i = 0;i< mGenerateProposalKernel.size();++i)
        {
            const auto& yolo = mGenerateProposalKernel[i];
            numElem = yolo.width*yolo.height;

            //copy anchor to device
	        CUDA_CHECK(cudaMemcpy(devAnchor,yolo.anchors,AnchorLen,cudaMemcpyHostToDevice));

            CalDetection<<< (yolo.width*yolo.height + mThreadCount - 1) / mThreadCount,
                             mThreadCount>>>
                                            (inputs[i],output, numElem, yolo.width, yolo.height,
                                            (float *)devAnchor, mClassCount);
        }
        CUDA_CHECK(cudaFree(devAnchor));

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


        spatial_scale: 1/4 1/8 1/16 1/32 1/64
        nms_thresh: 0.699999988079071
        pre_nms_topn: 1000
        min_size: 16.0
        post_nms_topn: 1000
        correct_transform_coords: 1

        //CAFFE_ENFORCE_EQ(scores.ndim(), 4, scores.ndim());
        //CAFFE_ENFORCE(scores.template IsType<float>(), scores.meta().name());

        //const auto num_images = scores.dim(0);
        //const auto A = scores.dim(1);
        //const auto H = scores.dim(2);
        //const auto W = scores.dim(3);
        //const auto box_dim = anchors.dim(1);
        */
        int A = mScoreC;
        int H = mScoreH;
        int W = mScoreW;
        int num_images = 1;
        int box_dim = 4;
        const int K = H * W;
        const int conv_layer_nboxes = K * A;

        // sort the scores while remember their original indexes,
        // ie their indexes in the tensor of shape (num_images,A,K) from the conv layer
        // each row of d_conv_layer_indexes is at first initialized to 1..A*K

        // dev_conv_layer_indexes_.Resize(num_images, conv_layer_nboxes);
        int *d_conv_layer_indexes = (int *)dev_conv_layer_indexes_;//.template mutable_data<int>();

        // d_image_offset[i] = i*K*A for i from 1 to num_images+1
        // Used by the segmented sort to only sort scores within one image

        //dev_image_offset_.Resize(num_images + 1);
        int *d_image_offset = (int *)dev_image_offset_;//.template mutable_data<int>();

        // The following calls to CUB primitives do nothing
        // (because the first arg is nullptr)
        // except setting cub_*_temp_storage_bytes
        size_t cub_sort_temp_storage_bytes = 0;
        float *flt_ptr = nullptr;
        int *int_ptr = nullptr;
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
                stream,
                false);

        // Allocate temporary storage for CUB
        //dev_cub_sort_buffer_.Resize(cub_sort_temp_storage_bytes);
        void *d_cub_sort_temp_storage = (char*)dev_cub_sort_buffer_;//.template mutable_data<char>();

        size_t cub_select_temp_storage_bytes = 0;
        char *char_ptr = nullptr;
        cub::DeviceSelect::Flagged( nullptr,
                                    cub_select_temp_storage_bytes,
                                    flt_ptr,
                                    char_ptr,
                                    flt_ptr,
                                    int_ptr,
                                    K * A,
                                    stream);

        // Allocate temporary storage for CUB
        //dev_cub_select_buffer_.Resize(cub_select_temp_storage_bytes);
        void *d_cub_select_temp_storage =(char*) dev_cub_select_buffer_;//.template mutable_data<char>();

        // Initialize :
        // - each row of dev_conv_layer_indexes to 1..K*A
        // - each d_nboxes to 0
        // - d_image_offset[i] = K*A*i for i 1..num_images+1
        // 2D grid
        InitializeDataKernel << < (CAFFE_GET_BLOCKS(A * K), num_images),
                CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
                0,
                stream >> >(num_images,
                            conv_layer_nboxes,
                            d_image_offset,
                            d_conv_layer_indexes);

        // Sorting input scores
        // dev_sorted_conv_layer_indexes_.Resize(num_images, conv_layer_nboxes);
        // dev_sorted_scores_.Resize(num_images, conv_layer_nboxes);

        const float *d_in_scores = (const float *)scores;// const Dtype * scores
        int *d_sorted_conv_layer_indexes = dev_sorted_conv_layer_indexes_;
        float *d_sorted_scores;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
                d_cub_sort_temp_storage,
                cub_sort_temp_storage_bytes,
                d_in_scores,//flt_ptr
                d_sorted_scores,//flt_ptr
                d_conv_layer_indexes,//int_ptr
                d_sorted_conv_layer_indexes,//int_ptr
                num_images * conv_layer_nboxes,//int
                num_images,//int
                d_image_offset,//int_ptr
                d_image_offset + 1,//int_ptr
                0,
                8 * sizeof(float), // sort all bits
                stream,
                false);

        // Keeping only the topN pre_nms
        const int nboxes_to_generate = std::min(conv_layer_nboxes, rpn_pre_nms_topN_);

        // Generating the boxes associated to the topN pre_nms scores
        //dev_boxes_.Resize(num_images, box_dim * nboxes_to_generate);
        //dev_boxes_keep_flags_.Resize(num_images, nboxes_to_generate);

        const Dtype *d_bbox_deltas = bbox_deltas;//Dtype
        const Dtype *d_anchors = anchors;//Dtype
        const Dtype *d_im_info_vec = im_info_tensor;//Dtype
        float *d_boxes;//Dtype

        char *d_boxes_keep_flags = dev_boxes_keep_flags_;//char

        GeneratePreNMSUprightBoxesKernel << < (CAFFE_GET_BLOCKS(nboxes_to_generate), num_images),
                CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
                0,
                stream >> > (d_sorted_conv_layer_indexes,//const int*
                            nboxes_to_generate,          //const int
                            (const Dtype*)d_bbox_deltas,
                            reinterpret_cast<const float4 *>(d_anchors),
                            H,//int
                            W,//int
                            A,//int
                            feat_stride_, //float
                            rpn_min_size_,//float
                            d_im_info_vec,//const Dtype *
                            num_images,
                            BBOX_XFORM_CLIP_DEFAULT,
                            reinterpret_cast<float4 *>(d_boxes),
                            nboxes_to_generate,
                            (Dtype *)d_sorted_scores,
                            d_boxes_keep_flags);

        const int nboxes_generated = nboxes_to_generate;

        Dtype *d_image_prenms_boxes = (Dtype *)dev_image_prenms_boxes_;//.template mutable_data<float>();
        Dtype *d_image_prenms_scores = (Dtype *)dev_image_prenms_scores_;//.template mutable_data<float>();
        int *d_image_boxes_keep_list = (int*)dev_image_boxes_keep_list_;//.template mutable_data<int>();

        //dev_image_prenms_boxes_.Resize(box_dim * nboxes_generated);
        //dev_image_prenms_scores_.Resize(nboxes_generated);
        //dev_image_boxes_keep_list_.Resize(nboxes_generated);

        const int roi_cols = box_dim + 1;
        const int max_postnms_nboxes = std::min(nboxes_generated, rpn_post_nms_topN_);

        //dev_postnms_rois_.Resize(roi_cols * num_images * max_postnms_nboxes);
        //dev_postnms_rois_probs_.Resize(num_images * max_postnms_nboxes);

        //dev_prenms_nboxes_.Resize(num_images);
        //host_prenms_nboxes_.Resize(num_images);

        int *d_prenms_nboxes = (int*)dev_prenms_nboxes_;//int
        int *h_prenms_nboxes = (int*)host_prenms_nboxes_;//int

        int nrois_in_output = 0;
        for (int image_index = 0; image_index < num_images; ++image_index) {
            // Sub matrices for current image
            const float *d_image_boxes = &d_boxes[image_index * nboxes_generated * box_dim];
            const float *d_image_sorted_scores = &d_sorted_scores[image_index * K * A];
            char *d_image_boxes_keep_flags =
                    &d_boxes_keep_flags[image_index * nboxes_generated];

            Dtype *d_image_postnms_rois = (Dtype *)&d_postnms_rois[roi_cols * nrois_in_output];
            Dtype *d_image_postnms_rois_probs = (Dtype *)&d_postnms_rois_probs[nrois_in_output];

            // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
            // to the output tensors

            cub::DeviceSelect::Flagged(
                    d_cub_select_temp_storage,
                    cub_select_temp_storage_bytes,
                    reinterpret_cast<const float4 *>(d_image_boxes),
                    d_image_boxes_keep_flags,
                    reinterpret_cast<float4 *>(d_image_prenms_boxes),
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
            //////////////////////////// TODO
            //host_prenms_nboxes_.CopyFrom(dev_prenms_nboxes_);

            // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
            // topN_prenms. Calling NMS on the generated boxes
            const int prenms_nboxes = *h_prenms_nboxes;
            int nkeep;
            nms_gpu(d_image_prenms_boxes,//const float* d_desc_sorted_boxes,
                           prenms_nboxes,//const int N,
                           rpn_nms_thresh_,//const float thresh,
                           true,//const bool legacy_plus_one,
                           d_image_boxes_keep_list,//int* d_keep_sorted_list,
                           &nkeep,//int* h_nkeep,
                           dev_nms_mask_,//float*  dev_delete_mask,
                           host_nms_mask_,//float* host_delete_mask,
                           stream//cudaStream_t stream,
                           );

            // All operations done after previous sort were keeping the relative order
            // of the elements the elements are still sorted keep topN <=> truncate the
            // array
            const int postnms_nboxes = std::min(nkeep, rpn_post_nms_topN_);

            // Moving the out boxes to the output tensors,
            // adding the image_index dimension on the fly
            WriteUprightBoxesOutput << < CAFFE_GET_BLOCKS(postnms_nboxes),
                                         CAFFE_CUDA_NUM_THREADS,
                                         0, stream >> > (
                    reinterpret_cast<const float4 *>(d_image_prenms_boxes),
                                                     d_image_prenms_scores,
                                                     d_image_boxes_keep_list,
                                                     postnms_nboxes,
                                                     image_index,
                                                     d_image_postnms_rois,
                                                     d_image_postnms_rois_probs);

            nrois_in_output += postnms_nboxes;

        }
        // Using a buffer because we cannot call ShrinkTo
        //out_rois->Resize(      nrois_in_output, roi_cols);
        //out_rois_probs->Resize(nrois_in_output);
        Dtype* d_out_rois = out_rois;//->template mutable_data<float>();
        Dtype* d_out_rois_probs = out_rois_probs;//->template mutable_data<float>();

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


    int GenerateProposalLayerPlugin::enqueue(int batchSize,
                                             const void*const * inputs,
                                             void** outputs,
                                             void* workspace,
                                             cudaStream_t stream){
        assert(batchSize == 1);
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

TEST(GenerateProposalsTest, TestRealDownSampledGPU) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("GenerateProposals");
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
