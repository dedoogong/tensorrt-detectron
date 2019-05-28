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

namespace BoxTransform
{
    struct BoxTransformKernel;

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

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {};

        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream);

        void forwardCpu(const float *const * inputs,float * output, cudaStream_t stream);

    private:
        int mClassCount;
        int mKernelCount;
        std::vector<BoxTransform::BoxTransformKernel> mBoxTransformKernel;
        int mThreadCount;

        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BOXTRANSFORMLAYER_H
