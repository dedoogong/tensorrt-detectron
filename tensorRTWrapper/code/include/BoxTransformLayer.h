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
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                 DataType type, PluginFormat format, int maxBatchSize) override {};

        template <typename Dtype>
        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize,   const void*const * inputs, void** outputs,
                            void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        template <typename Dtype>
        void forwardCpu(const Dtype *inputs, Dtype* outputs, cudaStream_t stream);

    private:
        int mThreadCount;

        DataType mDataType{ DataType::kFLOAT };
        float weights_1_= 10.0;
        float weights_2_= 10.0;
        float weights_3_= 5.0;
        float weights_4_= 5.0;
        int correct_transform_coords_= 1;
        int apply_scale_= 0;

        int m_inputTotalCount = 0;
        int m_ouputTotalCount = 0;
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BOXTRANSFORMLAYER_H
