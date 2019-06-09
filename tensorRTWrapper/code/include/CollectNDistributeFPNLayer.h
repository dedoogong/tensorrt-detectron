// Created by seunghyun lee on 19. 5. 23.

#ifndef COLLECTANDDISTRIBUTEFPNLAYER_H
#define COLLECTANDDISTRIBUTEFPNLAYER_H


#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>

namespace CollectNDistributeFPN
{
    struct CollectNDistributeFPNKernel;

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
    class CollectNDistributeFPNLayerPlugin: public IPluginExt
    {
    public:
        explicit CollectNDistributeFPNLayerPlugin(const int cudaThread = 512);
        CollectNDistributeFPNLayerPlugin(const void* data, size_t length);

		~CollectNDistributeFPNLayerPlugin() {
		
			if (mInputBuffer)
				CUDA_CHECK(cudaFreeHost(mInputBuffer));

			if (mOutputBuffer)
				CUDA_CHECK(cudaFreeHost(mOutputBuffer));
		
		};

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
		template <typename Dtype>
		void forwardCpu(const Dtype *inputs, Dtype* outputs, cudaStream_t stream);

    private:

		DataType mDataType{ DataType::kFLOAT };

		int roiCountPerFPN_ = 250;
		int rpn_min_level_ = 2;
		int rpn_max_level_ = 6;

		int rpn_post_nms_topN_ = 1000;

		int roi_min_level_ = 2;
		int roi_max_level_ = 5;

		int roi_canonical_level_ = 4
		int	roi_canonical_scale_ = 224;

		int m_rpn_rois_fpnH[5];
		int m_rpn_rois_fpnW[5];

		int m_rpn_rois_probs_fpn_H[5];
		int m_rpn_rois_probs_fpn_W[5];

		int m_inputTotalCount = 0;
		int m_ouputTotalCount = 0;
		int m_proposal_num;
		DataType mDataType{DataType::kFLOAT};
        //cpu
        void* mInputBuffer  {nullptr};
        void* mOutputBuffer {nullptr};
    };
};


#endif //COLLECTANDDISTRIBUTEFPNLAYER_H
