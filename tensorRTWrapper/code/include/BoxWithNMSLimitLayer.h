// Created by seunghyun lee on 19. 5. 23.

#ifndef BOXWITHNMSLIMITLAYER_H
#define BOXWITHNMSLIMITLAYER_H


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
    class BoxWithNMSLimitLayerPlugin: public IPluginExt
    {
    public:
        explicit BoxWithNMSLimitLayerPlugin(const int cudaThread = 512);
        BoxWithNMSLimitLayerPlugin(const void* data, size_t length);

        ~BoxWithNMSLimitLayerPlugin();

        int getNbOutputs() const override
        {
            return 3;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {};

        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        template <typename Dtype>
        void forwardCpu(const Dtype *inputs, Dtype* outputs, cudaStream_t stream);

    private:
        int mThreadCount;

        int mClsProbH;
        int mClsProbW;

        int mPredBoxH;
        int mPredBoxW;

        DataType mDataType{ DataType::kFLOAT };
        float score_thresh_=0.5f;
        float nms_thresh_= 0.5f;
        int detections_per_im_= 100;
        int soft_nms_enabled_= 1;
        int soft_nms_method_= 1;//#"linear"
        float soft_nms_sigma_= 0.5f;
        float soft_nms_min_score_thresh_= 0.1f;
        bool rotated_=false;

        int m_inputTotalCount = 0;
        int m_ouputTotalCount = 0;

        int m_nms_max_count=300;
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BATCHPERMUTELAYER_H
