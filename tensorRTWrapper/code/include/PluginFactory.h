#ifndef __PLUGIN_FACTORY_H_
#define __PLUGIN_FACTORY_H_

#include <vector>
#include <memory>
#include <regex>
#include "UpsampleLayer.h"
#include "YoloLayer.h"


#include "BatchPermuteLayer.h"
#include "BoxTransformLayer.h"
#include "CollectNDistributeFPNLayer.h"
#include "BoxWithNMSLimitLayer.h"

//#define CUSTOM
#ifdef CUSTOM
#include "GenerateProposalLayer.h"
#include "RoIAlign.h"

#endif
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
#ifdef CUSTOM
    // Added by SH Lee
    using nvinfer1::GenerateProposalLayerPlugin;
    using nvinfer1::BatchPermuteLayerPlugin;
    using nvinfer1::CollectNDistributeFPNLayerPlugin;
    using nvinfer1::BoxTransformLayerPlugin;
    using nvinfer1::BoxWithNMSLimitLayerPlugin;
    using nvinfer1::RoIAlignLayerPlugin;
#endif
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
#ifdef CUSTOM
			inline bool isGenerateProposal(const char* layerName) {
			    if ((strcmp(layerName, "rpn_roi_probs_fpn2") == 0)||(strcmp(layerName, "rpn_roi_probs_fpn3") == 0)||
                (strcmp(layerName, "rpn_roi_probs_fpn4") == 0)||
                (strcmp(layerName, "rpn_roi_probs_fpn5") == 0)||
                (strcmp(layerName, "rpn_roi_probs_fpn6") == 0)){return true;}
                else{return false;}

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
            inline bool isRoIAlign(const char* layerName){
                if ((strcmp(layerName, "roi_feat_fpn2") == 0)||(strcmp(layerName, "roi_feat_fpn3") == 0)||
                    (strcmp(layerName, "roi_feat_fpn4") == 0)||(strcmp(layerName, "roi_feat_fpn5") == 0)){return true;}
                else{return false;}

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

#endif
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
#ifdef CUSTOM
                if (isGenerateProposal(layerName)){
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginGenerateProposal.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))  );
                    return mPluginGenerateProposal.back().get();
                }
                else if (isRoIAlign(layerName)){
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginRoIAlign.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))  ));
                    return mPluginRoIAlign.back().get();
                }
                else if (isBatchPermute(layerName)){
                    assert(nbWeights == 0 && weights == nullptr && mPluginBatchPermute .get() ==  nullptr);
                    mPluginBatchPermute.reset(new BatchPermuteLayerPlugin(CUDA_THREAD_NUM));
                    return mPluginBatchPermute .get();
                }
                else if (isCollectAndDistributeFpnRpnProposals(layerName)){
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
#else
                assert(0); return nullptr;
#endif
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
#ifdef CUSTOM
            if (isGenerateProposal(layerName)){
                assert(nbWeights == 0 && weights == nullptr);
                mPluginGenerateProposal.emplace_back(std::unique_ptr<GenerateProposalLayerPlugin>(new GenerateProposalLayerPlugin(CUDA_THREAD_NUM))  );
                return mPluginGenerateProposal.back().get();}

            else if (isRoIAlign(layerName)){
                assert(nbWeights == 0 && weights == nullptr);
                mPluginRoIAlign.emplace_back(std::unique_ptr<RoIAlignLayerPlugin>(new RoIAlignLayerPlugin(CUDA_THREAD_NUM))  ));
                return mPluginRoIAlign.back().get();}

            else if (isBatchPermute(layerName)){
                assert(nbWeights == 0 && weights == nullptr && mPluginBatchPermute .get() ==  nullptr);
                mPluginBatchPermute.reset(new BatchPermuteLayerPlugin(CUDA_THREAD_NUM));
                return mPluginBatchPermute .get();}

            else if (isCollectAndDistributeFpnRpnProposals(layerName)){
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
#else
            assert(0); return nullptr;
#endif

        }

        bool isPlugin(const char* name) override{ return isPluginExt(name);}

        bool isPluginExt(const char* name) override{
            //std::cout << "check plugin " << name  << isYolo(name)<< std::endl;
#ifdef CUSTOM
            return isGenerateProposal(name) || isRoIAlign(name) || isBatchPermute(name) || isCollectNDistributeFPN(name) || isBoxTransform(name) || isBoxNMS(name);}
#else
            return false;}
#endif
        // The application has to destroy the plugin when it knows it's safe to do so.
        void destroyPlugin(){
#ifdef CUSTOM
            for (auto& item : mPluginGenerateProposal)
                item.reset();

            for (auto& item : mPluginRoIAlign)
                item.reset();

            mPluginCollectNDistributeFPN.reset();

			mPluginBatchPermute.reset();

            mPluginBoxTransform.reset();
            mPluginBoxWithNMSLimit.reset();
#endif
			//mPluginYolo.reset();
        }
        void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { if(ptr) ptr->destroy(); }};
#ifdef CUSTOM
        std::vector<std::unique_ptr<GenerateProposalLayerPlugin>> mPluginGenerateProposal{};
        std::vector<std::unique_ptr<RoIAlignLayerPlugin>>         mPluginRoIAlign{};
		std::unique_ptr<CollectNDistributeFPNLayerPlugin> mPluginCollectNDistributeFPN{ nullptr };

        std::unique_ptr<BatchPermuteLayerPlugin>          mPluginBatchPermute {nullptr};
        
        std::unique_ptr<BoxTransformLayerPlugin>          mPluginBoxTransform {nullptr};
        std::unique_ptr<BoxWithNMSLimitLayerPlugin>       mPluginBoxWithNMSLimit {nullptr};
#endif
        //std::vector<std::unique_ptr<INvPlugin,void (*)(INvPlugin*)>> mPluginLeakyRelu{};
		//std::vector<std::unique_ptr<UpsampleLayerPlugin>>         mPluginUpsample{};
		//std::unique_ptr<YoloLayerPlugin>                  mPluginYolo {nullptr};

    };
}

#endif