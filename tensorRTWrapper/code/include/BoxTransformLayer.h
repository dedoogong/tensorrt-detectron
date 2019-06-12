// Created by seunghyun lee on 19. 5. 23.

#ifndef BOXTRANSFORMLAYER_H
#define BOXTRANSFORMLAYER_H


#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>


namespace nvinfer1
{
    class BoxTransformLayerPlugin: public IPluginExt
    {
    public:
        explicit BoxTransformLayerPlugin(const int cudaThread = 512);
        BoxTransformLayerPlugin(const void* data, size_t length);

        ~BoxTransformLayerPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            (type == DataType::kFLOAT || type == DataType::kHALF ||
             type == DataType::kINT8 ) && format == PluginFormat::kNCHW;
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                 DataType type, PluginFormat format, int maxBatchSize) override {};

        //template <typename Dtype>
        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize,   const void*const * inputs, void** outputs,
                            void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        template <typename Dtype>
        void forwardCpu( //const float *const * inputs,
                //      float * output,
                const Dtype * roi_in,//rpn_rois:    (1000,4)
                const Dtype * delta_in,//bbox_pred: (1000,8) bg x1 y1 x2 y2, human x1 y1 x2 y2
                const Dtype * iminfo_in,//batch_size(1),3
                Dtype* box_out_,// pred_bbox: (1000, 8)
                cudaStream_t stream);

    private:
        int mThreadCount;

        DataType mDataType{ DataType::kFLOAT };
        float weights_1_= 10.0;
        float weights_2_= 10.0;
        float weights_3_= 5.0;
        float weights_4_= 5.0;
        int correct_transform_coords_= 1;
        int apply_scale_= 0;

        int mRpnRoisH=0;
        int mRpnRoisW=0;
        int mBoxPredH=0;
        int mBoxPredW=0;
        int mIminfoH=0;
        int mIminfoW=0;

        bool angle_bound_on_=true;
        int angle_bound_lo_=-90;
        int angle_bound_hi_=90;
        float clip_angle_thresh_=1.0;

        int m_inputTotalCount = 0;
        int m_ouputTotalCount = 0;
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BOXTRANSFORMLAYER_H
