// Created by seunghyun lee on 19. 5. 23.

#ifndef BATCHPERMUTELAYER_H
#define BATCHPERMUTELAYER_H


#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>

namespace BatchPermute
{
    struct BatchPermuteKernel;

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
    class BatchPermuteLayerPlugin: public IPluginExt
    {
    public:
        explicit BatchPermuteLayerPlugin(const int cudaThread = 512);
        BatchPermuteLayerPlugin(const void* data, size_t length);

        ~BatchPermuteLayerPlugin();

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
        std::vector<BatchPermute::BatchPermuteKernel> mBatchPermuteKernel;
        int mThreadCount;

		int m_inputTotalCount = 0;
		int m_ouputTotalCount = 0;
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BATCHPERMUTELAYER_H
