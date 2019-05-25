#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "PluginFactory.h"
#include "Utils.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;


namespace Tn
{
    enum class RUN_MODE
    {
        FLOAT32 = 0,
        FLOAT16 = 1,    
        INT8 = 2
    };

    class trtNet 
    {
        public:
            //Load from caffe model
            trtNet(const std::string& prototxt,const std::string& caffeModel,const std::vector<std::string>& outputNodesName,
                    const std::vector<std::vector<float>>& calibratorData, RUN_MODE mode = RUN_MODE::FLOAT32);
        
            //Load from engine file
            explicit trtNet(const std::string& engineFile);

            ~trtNet()
            {
                // Release the stream and the buffers
                cudaStreamSynchronize(mTrtCudaStream);
                cudaStreamDestroy(mTrtCudaStream);
                for(auto& item : mTrtCudaBuffer)
                    cudaFree(item);

                mTrtPluginFactory.destroyPlugin();

                if(!mTrtRunTime)
                    mTrtRunTime->destroy();
                if(!mTrtContext)
                    mTrtContext->destroy();
                if(!mTrtEngine)
                    mTrtEngine->destroy();

                ///////////////// free part for generate proposal plugin ///////////////
                // release the stream and the buffers
                //cudaStreamDestroy(stream);
                //CHECK(cudaFree(buffers[inputIndex0]));
                //CHECK(cudaFree(buffers[inputIndex1]));
                //CHECK(cudaFree(buffers[outputIndex0]));
                //CHECK(cudaFree(buffers[outputIndex1]));
                //CHECK(cudaFree(buffers[outputIndex2]));


            };

            void saveEngine(std::string fileName)
            {
                if(mTrtEngine)
                {
                    nvinfer1::IHostMemory* data = mTrtEngine->serialize();
                    std::ofstream file;
                    file.open(fileName,std::ios::binary | std::ios::out);
                    if(!file.is_open())
                    {
                        std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                        return;
                    }

                    file.write((const char*)data->data(), data->size());
                    file.close();
                }
            };

            void doInference(IExecutionContext& context, const void* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float* outputRois, int batchSize);

            inline size_t getInputSize() {
                return std::accumulate(mTrtBindBufferSize.begin(), mTrtBindBufferSize.begin() + mTrtInputCount,0);
            };

            inline size_t getOutputSize() {
                return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount, mTrtBindBufferSize.end(),0);
            };
            
            void printTime()
            {
                mTrtProfiler.printLayerTimes(mTrtIterationTime);
            }

        private:
                nvinfer1::ICudaEngine* loadModelAndCreateEngine(const char* deployFile, const char* modelFile,int maxBatchSize,
                                        nvcaffeparser1::ICaffeParser* parser, nvcaffeparser1::IPluginFactory* pluginFactory,
                                        nvinfer1::IInt8Calibrator* calibrator, nvinfer1::IHostMemory*& trtModelStream,const std::vector<std::string>& outputNodesName);

                void InitEngine();

                const int poolingH = 7;
                const int poolingW = 7;
                const int featureStride = 16;
                const int preNmsTop = 6000;
                const int nmsMaxOut = 300;
                const int anchorsRatioCount = 3;
                const int anchorsScaleCount = 3;
                const float iouThreshold = 0.7f;
                const float minBoxSize = 16;
                const float spatialScale = 0.0625f;
                const float anchorsRatios[3] = {0.5f, 1.0f, 2.0f};
                const float anchorsScales[3] = {8.0f, 16.0f, 32.0f};
                void* buffers[5];
                nvinfer1::IExecutionContext* mTrtContext;
                nvinfer1::ICudaEngine* mTrtEngine;
                nvinfer1::IRuntime* mTrtRunTime;
                PluginFactory mTrtPluginFactory;    
                cudaStream_t mTrtCudaStream;
                Profiler mTrtProfiler;
                RUN_MODE mTrtRunMode;

                std::vector<void*> mTrtCudaBuffer;
                std::vector<int64_t> mTrtBindBufferSize;
                int mTrtInputCount;
                int mTrtIterationTime;
                static const int INPUT_C = 3;
                static const int INPUT_H = 375;//?? 1080
                static const int INPUT_W = 500;// 1920??
                static const int IM_INFO_SIZE = 3;
                static const int OUTPUT_CLS_SIZE = 21;
                static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
                const std::string CLASSES[OUTPUT_CLS_SIZE]{"background", "person", "catcher", "pitcher", "simpan", "hitter" };


    };
}

#endif //__TRT_NET_H_
