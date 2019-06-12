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

        template <typename Dtype>
        void forwardCpu(const Dtype * roi_feat_shuffled,     // 1000, 256, 7, 7
                        const int   * rois_idx_restore_int32,// 1000, 1
                              Dtype * roi_feat,              // 1000, 256, 7, 7
                        cudaStream_t  stream);

    private:
        int mThreadCount;

        DataType mDataType{ DataType::kFLOAT };

		int m_inputTotalCount = 0;
		int m_ouputTotalCount = 0;

        int mRoIFeatureShuffledN;
        int mRoIFeatureShuffledC;
        int mRoIFeatureShuffledH;
        int mRoIFeatureShuffledW;
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //BATCHPERMUTELAYER_H
