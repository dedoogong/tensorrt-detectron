#ifndef CAFFE2_OPERATORS_UTILS_NMS_GPU_H_
#define CAFFE2_OPERATORS_UTILS_NMS_GPU_H_

#include <vector>

//#include "../../../include/caffe2/core/context_gpu.h"
#include <math.h>
#include <cuda_runtime.h>
namespace utils {

    const float BBOX_XFORM_CLIP_DEFAULT= log(1000.0 / 16.0);

// Computes Non-Maximum Suppression on the GPU
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
//
// d_desc_sorted_boxes : pixel coordinates of proposed bounding boxes
//    size: (N,4), format: [x1; y1; x2; y2]
//    the boxes are sorted by scores in descending order
// N : number of boxes
// d_keep_sorted_list : row indices of the selected proposals, sorted by score
// h_nkeep  : number of selected proposals
// dev_delete_mask, host_delete_mask : Tensors that will be used as temp storage
// by NMS
//    Those tensors will be resized to the necessary size
// context : current CUDA context
void  nms_gpu_upright(
            const float* d_desc_sorted_boxes_float_ptr,
            const int N,
            const float thresh,
            const bool legacy_plus_one,
            int* d_keep_sorted_list,
            int* h_nkeep,
            float* dev_delete_mask,
            float* host_delete_mask,
            cudaStream_t stream);

struct RotatedBox {
  float x_ctr, y_ctr, w, h, a;
};

// Same as nms_gpu_upright, but for rotated boxes with angle info.
// d_desc_sorted_boxes : pixel coordinates of proposed bounding boxes
//    size: (N,5), format: [x_ct; y_ctr; width; height; angle]
//    the boxes are sorted by scores in descending order
/*void nms_gpu_rotated(
    const float* d_desc_sorted_boxes,
    const int N,
    const float thresh,
    int* d_keep_sorted_list,
    int* h_nkeep,
    TensorCUDA& dev_delete_mask,
    TensorCPU& host_delete_mask,
    CUDAContext* context);
*/
void nms_gpu(
            const float* d_desc_sorted_boxes,
            const int N,
            const float thresh,
            const bool legacy_plus_one,
            int* d_keep_sorted_list,
            int* h_nkeep,
            float*  dev_delete_mask,
            float* host_delete_mask,
            cudaStream_t stream,
            const int box_dim);

} // namespace utils

#endif // CAFFE2_OPERATORS_UTILS_NMS_GPU_H_
