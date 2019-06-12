// Created by seunghyun lee on 19. 5. 23.

#ifndef ROIALIGNLAYER_H
#define ROIALIGNLAYER_H


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
    class RoIAlignLayerPlugin: public IPluginExt
    {
    public:
        explicit RoIAlignLayerPlugin(const int cudaThread = 512);
        RoIAlignLayerPlugin(const void* data, size_t length);
        DataType mDataType{DataType::kFLOAT};
        ~RoIAlignLayerPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            (type == DataType::kFLOAT || type == DataType::kHALF ||
             type == DataType::kINT8 ) && format == PluginFormat::kNCHW;
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {};

        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        template <typename DType>
        void forwardGpu(const DType* features,
                        const DType* rois,
                        const float spatial_scale,
                        const int pooled_height,
                        const int pooled_width,
                        const int sampling_ratio,
                        cudaStream_t stream,
                        const int num_rois,
                        const int channels,
                        const int height,
                        const int width,
                        DType* output);
    private:
        int mThreadCount;

        int mFeatureMap_C;
        int mFeatureMap_H;
        int mFeatureMap_W;

        int mRois_H;
        int mRois_W;

        int pooled_height=14;
        int pooled_width=14;
        int sampling_ratio=2;

        const float spatial_scale=0.25f;
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BATCHPERMUTELAYER_H
