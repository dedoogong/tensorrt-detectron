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
    class GenerateProposalLayerPlugin: public IPluginExt
    {
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
        void forwardGpu(const Dtype* input,Dtype * outputint);
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
    };
};


#endif //GENERATEPROPOSALLAYER_H
