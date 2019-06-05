// Created by seunghyun lee on 19. 5. 23.

#ifndef GENERATEPROPOSALLAYER_H
#define GENERATEPROPOSALLAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>
#include "caffe2/utils/eigen_utils.h"
#include "cub/cub/cub.cuh"

namespace GenerateProposal
{
    struct GenerateProposalKernel;

    static constexpr int LOCATIONS = 4;
    struct Detection{
        //x y w h
        float bbox[LOCATIONS];
        //float objectness;
        int classId;
        float prob;
    };
}


namespace nvinfer1
{
    class GenerateProposalLayerPlugin: public IPluginExt{
    public:
        explicit GenerateProposalLayerPlugin(const int cudaThread = 512);
        GenerateProposalLayerPlugin(const void* data, size_t length);

        ~GenerateProposalLayerPlugin();

        int getNbOutputs() const override{ return 2; }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs,
                                 const Dims* outputDims,
                                 int nbOutputs,
                                 DataType type, PluginFormat format,
                                 int maxBatchSize) override {};

        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize,
                            const void*const * inputs,
                            void** outputs, void* workspace,
                            cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        template <typename Dtype>
        void forwardGpu(const Dtype * scores,
                        const Dtype * bbox_deltas,
                        const Dtype * im_info_tensor,
                        const Dtype * anchors,
                              Dtype* out_rois,
                              Dtype* out_rois_probs,
                              cudaStream_t stream);
                       //(const float *const * inputs,
                       //float * output, cudaStream_t stream);

    private:
        std::vector<GenerateProposal::GenerateProposalKernel> mGenerateProposalKernel;
        int mThreadCount;

        int mScoreC{0};
        int mScoreH{0};
        int mScoreW{0};

        int mBoxDeltaC{0};
        int mBoxDeltaH{0};
        int mBoxDeltaW{0};

        DataType mDataType{DataType::kFLOAT};
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};

        // spatial_scale_ must be declared before feat_stride_
        float spatial_scale_{1.0};
        float feat_stride_{1.0};

        // RPN_PRE_NMS_TOP_N
        int rpn_pre_nms_topN_{6000};
        // RPN_POST_NMS_TOP_N
        int rpn_post_nms_topN_{300};
        // RPN_NMS_THRESH
        float rpn_nms_thresh_{0.7};
        // RPN_MIN_SIZE
        float rpn_min_size_{16};
        // If set, for rotated boxes in RRPN, output angles are normalized to be
        // within [angle_bound_lo, angle_bound_hi].
        bool angle_bound_on_{true};
        int angle_bound_lo_{-90};
        int angle_bound_hi_{90};
        // For RRPN, clip almost horizontal boxes within this threshold of
        // tolerance for backward compatibility. Set to negative value for
        // no clipping.
        float clip_angle_thresh_{1.0};

        // Scratch space required by the CUDA version
        // CUB buffers
        float* dev_cub_sort_buffer_{0};
        float* dev_cub_select_buffer_{0};
        float* dev_image_offset_{0};
        float* dev_conv_layer_indexes_{0};
        int* dev_sorted_conv_layer_indexes_{0};
        float * dev_sorted_scores_{0};
        float* dev_boxes_{0};
        char* dev_boxes_keep_flags_{0};

        // prenms proposals (raw proposals minus empty boxes)
        float* dev_image_prenms_boxes_{0};
        float* dev_image_prenms_scores_{0};
        float* dev_prenms_nboxes_{0};
        float* host_prenms_nboxes_{0};

        float* dev_image_boxes_keep_list_{0};

        // Tensors used by NMS
        int* dev_nms_mask_{0};
        int* host_nms_mask_{0};

        // Buffer for output
        float* dev_postnms_rois_{0};
        float* dev_postnms_rois_probs_{0};


    };
};


#endif //GENERATEPROPOSALLAYER_H
