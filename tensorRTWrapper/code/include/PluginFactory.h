#ifndef __PLUGIN_FACTORY_H_
#define __PLUGIN_FACTORY_H_

#include <vector>
#include <memory>
#include <regex>
#include "UpsampleLayer.h"
#include "YoloLayer.h"
#include "GenerateProposalLayer.h"
#include "RoIAlign.h"
#include "BatchPermuteLayer.h"
#include "BoxTransformLayer.h"
#include "CollectNDistributeFPNLayer.h"
#include "BoxWithNMSLimitLayer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
namespace Tn
{
    static constexpr int CUDA_THREAD_NUM = 512;

    // Integration for serialization.
    using nvinfer1::plugin::INvPlugin;
    using nvinfer1::plugin::createPReLUPlugin;
    //using nvinfer1::UpsampleLayerPlugin;
    //using nvinfer1::YoloLayerPlugin;

    // Added by SH Lee
    using nvinfer1::GenerateProposalLayerPlugin;
    using nvinfer1::BatchPermuteLayerPlugin;
    using nvinfer1::CollectNDistributeFPNLayerPlugin;
    using nvinfer1::BoxTransformLayerPlugin;
    using nvinfer1::BoxWithNMSLimitLayerPlugin;
    using nvinfer1::RoIAlignLayerPlugin;

    class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
    {
        public:
			/*
            inline bool isLeakyRelu(const char* layerName){
                return std::regex_match(layerName , std::regex(R"(layer(\d*)-act)"));
            }

            inline bool isUpsample(const char* layerName){
                return std::regex_match(layerName , std::regex(R"(layer(\d*)-upsample)"));
            }

            inline bool isYolo(const char* layerName){
                return strcmp(layerName,"yolo-det") == 0;
            }
			*/
			inline bool isGenerateProposal1(const char* layerName) {
				return strcmp(layerName, "rpn_roi_probs_fpn2") == 0;
			}

			inline bool isGenerateProposal2(const char* layerName) {
				return strcmp(layerName, "rpn_roi_probs_fpn3") == 0;
			}

			inline bool isGenerateProposal3(const char* layerName) {
				return strcmp(layerName, "rpn_roi_probs_fpn4") == 0;
			}

			inline bool isGenerateProposal4(const char* layerName) {
				return strcmp(layerName, "rpn_roi_probs_fpn5") == 0;
			}

			inline bool isGenerateProposal5(const char* layerName) {
				return strcmp(layerName, "rpn_roi_probs_fpn6") == 0;
			}
			/*
					spatial_scale: 1/4 1/8 1/16 1/32 1/64
					nms_thresh: 0.699999988079071
					pre_nms_topn: 1000
					min_size: 16.0
					post_nms_topn: 1000
					correct_transform_coords: 1*/
			inline bool isCollectAndDistributeFpnRpnProposals(const char* layerName) {
				return strcmp(layerName, "CollectAndDistributeFpnRpnProposals") == 0;
			}
			
            //roi_feat_fpn2 ps roi align??
            inline bool isRoIAlign1(const char* layerName){
				return strcmp(layerName, "roi_feat_fpn2") == 0;                
            }
			inline bool isRoIAlign2(const char* layerName) {
				return strcmp(layerName, "roi_feat_fpn3") == 0;
			}
			inline bool isRoIAlign3(const char* layerName) {
				return strcmp(layerName, "roi_feat_fpn4") == 0;
			}
			inline bool isRoIAlign4(const char* layerName) {
				return strcmp(layerName, "roi_feat_fpn5") == 0;
			}
			/*  sampling_ratio: 2
					pooled_w: 7
					pooled_h: 7
					spatial_scale: 0.25 0.125 0.0625  0.03125 */
            inline bool isBatchPermute(const char* layerName){
                return strcmp(layerName,"roi_feat") == 0;
            }
            inline bool isCollectNDistributeFPN(const char* layerName){
                return strcmp(layerName,"CollectAndDistributeFpnRpnProposals") == 0;
            }
            inline bool isBoxTransform(const char* layerName){
                return strcmp(layerName,"pred_bbox") == 0;
            }
            inline bool isBoxNMS(const char* layerName){
                return strcmp(layerName,"class_nms") == 0;
            }
            virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override{
                assert(isPlugin(layerName));

                /*if(isLeakyRelu(layerName)){
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginLeakyRelu.emplace_back(std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createPReLUPlugin(NEG_SLOPE), nvPluginDeleter));
                    return mPluginLeakyRelu.back().get();
                }
                else if (isUpsample(layerName)){
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginUpsample.emplace_back(std::unique_ptr<UpsampleLayerPlugin>(new UpsampleLayerPlugin(UPSAMPLE_SCALE,CUDA_THREAD_NUM)));
                    return mPluginUpsample.back().get();
                }  */
                if (isGenerateProposal1(layerName)){ 
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginGenerateProposal1.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))  );
                    return mPluginGenerateProposal1.back().get();
                }
				else if (isGenerateProposal2(layerName)) { 
					assert(nbWeights == 0 && weights == nullptr);
					mPluginGenerateProposal2.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginGenerateProposal2.back().get();
				}
				else if (isGenerateProposal3(layerName)) { 
					assert(nbWeights == 0 && weights == nullptr);
					mPluginGenerateProposal3.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginGenerateProposal3.back().get();
				}
				else if (isGenerateProposal4(layerName)) { 
					assert(nbWeights == 0 && weights == nullptr);
					mPluginGenerateProposal4.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginGenerateProposal4.back().get();
				}
				else if (isGenerateProposal5(layerName)) { 
					assert(nbWeights == 0 && weights == nullptr);
					mPluginGenerateProposal5.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginGenerateProposal5.back().get();
				}

                else if (isRoIAlign1(layerName)){ 
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginRoIAlign1.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))  ));
                    return mPluginRoIAlign1.back().get();
                }
				else if (isRoIAlign2(layerName)) {
					assert(nbWeights == 0 && weights == nullptr);
					mPluginRoIAlign2.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginRoIAlign2.back().get();
				}
				else if (isRoIAlign3(layerName)) {
					assert(nbWeights == 0 && weights == nullptr);
					mPluginRoIAlign3.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginRoIAlign3.back().get();
				}
				else if (isRoIAlign4(layerName)) {
					assert(nbWeights == 0 && weights == nullptr);
					mPluginRoIAlign4.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))));
					return mPluginRoIAlign4.back().get();
				}

                else if (isBatchPermute(layerName)){
                    assert(nbWeights == 0 && weights == nullptr && mPluginBatchPermute .get() ==  nullptr);
                    mPluginBatchPermute .reset(new BatchPermuteLayerPlugin(CUDA_THREAD_NUM));
                    return mPluginBatchPermute .get();
                }
                else if (isCollectNDistributeFPN(layerName)){
                    assert(nbWeights == 0 && weights == nullptr && mPluginCollectNDistributeFPN.get() ==  nullptr);
                    mPluginCollectNDistributeFPN.reset(new CollectNDistributeFPNLayerPlugin(CUDA_THREAD_NUM));
                    return mPluginCollectNDistributeFPN.get();
                }
                else if (isBoxTransform(layerName)){
                    assert(nbWeights == 0 && weights == nullptr && mPluginBoxTransform.get() ==  nullptr);
                    mPluginBoxTransform.reset(new BoxTransformLayerPlugin(CUDA_THREAD_NUM));
                    return mPluginBoxTransform.get();
                }
                else if (isBoxNMS(layerName)){
                    assert(nbWeights == 0 && weights == nullptr && mPluginBoxWithNMSLimit.get() ==  nullptr);
                    mPluginBoxWithNMSLimit.reset(new BoxWithNMSLimitLayerPlugin(CUDA_THREAD_NUM));
                    return mPluginBoxWithNMSLimit.get();
                }
                else{
                    assert(0);
                    return nullptr;
                }
            }

        nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override{
            assert(isPlugin(layerName));
            /*
            if (isLeakyRelu(layerName)){
                mPluginLeakyRelu.emplace_back(std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createPReLUPlugin(serialData, serialLength), nvPluginDeleter));
                return mPluginLeakyRelu.back().get();}
            else if (isUpsample(layerName)){
                mPluginUpsample.emplace_back(std::unique_ptr<UpsampleLayerPlugin>(new UpsampleLayerPlugin(serialData, serialLength)));
                return mPluginUpsample.back().get();x\}
            else if (isYolo(layerName)){
                assert(mPluginYolo.get() ==  nullptr);
                mPluginYolo.reset(new YoloLayerPlugin(serialData, serialLength));
                return mPluginYolo.get();}*/
            if (isGenerateProposal1(layerName)){ 
                assert(nbWeights == 0 && weights == nullptr);
                mPluginGenerateProposal1.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))  );
                return mPluginGenerateProposal1.back().get();}
			else if (isGenerateProposal2(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginGenerateProposal2.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM)));
				return mPluginGenerateProposal2.back().get();}
			else if (isGenerateProposal3(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginGenerateProposal3.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM)));
				return mPluginGenerateProposal3.back().get();}
			else if (isGenerateProposal4(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginGenerateProposal4.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM)));
				return mPluginGenerateProposal4.back().get();}
			else if (isGenerateProposal5(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginGenerateProposal5.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM)));
				return mPluginGenerateProposal5.back().get();}
            else if (isRoIAlign1(layerName)){ 
                assert(nbWeights == 0 && weights == nullptr);
                mPluginRoIAlign1.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))  ));
                return mPluginRoIAlign1.back().get();}
			else if (isRoIAlign2(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginRoIAlign2.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))));
				return mPluginRoIAlign2.back().get();}
			else if (isRoIAlign3(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginRoIAlign3.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))));
				return mPluginRoIAlign3.back().get();}
			else if (isRoIAlign4(layerName)) {
				assert(nbWeights == 0 && weights == nullptr);
				mPluginRoIAlign4.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))));
				return mPluginRoIAlign4.back().get();}
            else if (isBatchPermute(layerName)){
                assert(nbWeights == 0 && weights == nullptr && mPluginBatchPermute .get() ==  nullptr);
                mPluginBatchPermute .reset(new BatchPermuteLayerPlugin(CUDA_THREAD_NUM));
                return mPluginBatchPermute .get();}
            else if (isCollectNDistributeFPN(layerName)){
                assert(nbWeights == 0 && weights == nullptr && mPluginCollectNDistributeFPN.get() ==  nullptr);
                mPluginCollectNDistributeFPN.reset(new CollectNDistributeFPNLayerPlugin(CUDA_THREAD_NUM));
                return mPluginCollectNDistributeFPN.get();}
            else if (isBoxTransform(layerName)){
                assert(nbWeights == 0 && weights == nullptr && mPluginBoxTransform.get() ==  nullptr);
                mPluginBoxTransform.reset(new BoxTransformLayerPlugin(CUDA_THREAD_NUM));
                return mPluginBoxTransform.get();}
            else if (isBoxNMS(layerName)){
                assert(nbWeights == 0 && weights == nullptr && mPluginBoxWithNMSLimit.get() ==  nullptr);
                mPluginBoxWithNMSLimit.reset(new BoxWithNMSLimitLayerPlugin(CUDA_THREAD_NUM));
                return mPluginBoxWithNMSLimit.get();}
            else{assert(0); return nullptr;}
        }

        bool isPlugin(const char* name) override{ return isPluginExt(name);}

        bool isPluginExt(const char* name) override{
            //std::cout << "check plugin " << name  << isYolo(name)<< std::endl;
            return isGenerateProposal1(name) || 
				isGenerateProposal2(name) || 
				isGenerateProposal3(name) || 
				isGenerateProposal4(name) || 
				isGenerateProposal5(name) || 
				isRoIAlign1(name) || 
				isRoIAlign2(name) || 
				isRoIAlign3(name) || 
				isRoIAlign4(name) || 
				isBatchPermute(name) || isCollectNDistributeFPN(name) || isBoxTransform(name) || isBoxNMS(name);}

        // The application has to destroy the plugin when it knows it's safe to do so.
        void destroyPlugin(){
            for (auto& item : mPluginGenerateProposal1)
                item.reset();
			for (auto& item : mPluginGenerateProposal2)
				item.reset();
			for (auto& item : mPluginGenerateProposal3)
				item.reset();
			for (auto& item : mPluginGenerateProposal4)
				item.reset();
			for (auto& item : mPluginGenerateProposal5)
				item.reset();

            for (auto& item : mPluginRoIAlign1)
                item.reset();
			for (auto& item : mPluginRoIAlign2)
				item.reset();
			for (auto& item : mPluginRoIAlign3)
				item.reset();
			for (auto& item : mPluginRoIAlign4)
				item.reset();
			
            mPluginCollectNDistributeFPN.reset();

			mPluginBatchPermute.reset();

            mPluginBoxTransform.reset();
            mPluginBoxWithNMSLimit.reset();
			//mPluginYolo.reset();
        }
        void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { if(ptr) ptr->destroy(); }};        
        std::vector<std::unique_ptr<GenerateProposalLayerPlugin>> mPluginGenerateProposal1{};
		std::vector<std::unique_ptr<GenerateProposalLayerPlugin>> mPluginGenerateProposal2{};
		std::vector<std::unique_ptr<GenerateProposalLayerPlugin>> mPluginGenerateProposal3{};
		std::vector<std::unique_ptr<GenerateProposalLayerPlugin>> mPluginGenerateProposal4{};
		std::vector<std::unique_ptr<GenerateProposalLayerPlugin>> mPluginGenerateProposal5{};
        std::vector<std::unique_ptr<RoIAlignLayerPlugin>>         mPluginRoIAlign1{};
		std::vector<std::unique_ptr<RoIAlignLayerPlugin>>         mPluginRoIAlign2{};
		std::vector<std::unique_ptr<RoIAlignLayerPlugin>>         mPluginRoIAlign3{};
		std::vector<std::unique_ptr<RoIAlignLayerPlugin>>         mPluginRoIAlign4{};

		std::unique_ptr<CollectNDistributeFPNLayerPlugin> mPluginCollectNDistributeFPN{ nullptr };

        std::unique_ptr<BatchPermuteLayerPlugin>          mPluginBatchPermute {nullptr};
        
        std::unique_ptr<BoxTransformLayerPlugin>          mPluginBoxTransform {nullptr};
        std::unique_ptr<BoxWithNMSLimitLayerPlugin>       mPluginBoxWithNMSLimit {nullptr};

		//std::vector<std::unique_ptr<INvPlugin,void (*)(INvPlugin*)>> mPluginLeakyRelu{};
		//std::vector<std::unique_ptr<UpsampleLayerPlugin>>         mPluginUpsample{};
		//std::unique_ptr<YoloLayerPlugin>                  mPluginYolo {nullptr};

    };
}

#endif