//#include "GenerateProposalConfigs.h"
#include "GenerateProposalLayer.h"

using namespace GenerateProposal;
/*

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 375;
static const int INPUT_W = 500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
static int gUseDLACore{-1};

const std::string CLASSES[OUTPUT_CLS_SIZE]{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";
*/
namespace nvinfer1{
    GenerateProposalLayerPlugin::GenerateProposalLayerPlugin(const int cudaThread /*= 512*/):mThreadCount(cudaThread){
        /*mClassCount = CLASS_NUM;
        mGenerateProposalKernel.clear();
        mGenerateProposalKernel.push_back(yolo1);
        mGenerateProposalKernel.push_back(yolo2);
        mGenerateProposalKernel.push_back(yolo3);

        mKernelCount = mGenerateProposalKernel.size();*/
    }
    GenerateProposalLayerPlugin::~GenerateProposalLayerPlugin(){
        if(mInputBuffer)
            CUDA_CHECK(cudaFreeHost(mInputBuffer));
        if(mOutputBuffer)
            CUDA_CHECK(cudaFreeHost(mOutputBuffer));
    }
    // create the plugin at runtime from a byte stream
    GenerateProposalLayerPlugin::GenerateProposalLayerPlugin(const void* data, size_t length){
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mGenerateProposalKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(GenerateProposalKernel);
        memcpy(mGenerateProposalKernel.data(),d,kernelSize);
        d += kernelSize;

        assert(d == a + length);
    }

    void GenerateProposalLayerPlugin::serialize(void* buffer)
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(GenerateProposalKernel);
        memcpy(d,mGenerateProposalKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
    
    size_t GenerateProposalLayerPlugin::getSerializationSize()
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(GenerateProposal::GenerateProposalKernel) * mGenerateProposalKernel.size();
    }

    int GenerateProposalLayerPlugin::initialize()
    {
            /*
            int totalCount = 0;
            for(const auto& yolo : mGenerateProposalKernel)
                totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
            CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

            totalCount = 0;//detection count
            for(const auto& yolo : mGenerateProposalKernel)
                totalCount += yolo.width*yolo.height * CHECK_COUNT;
            CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection), cudaHostAllocDefault));
            */

            return 0;
    }
    
    Dims GenerateProposalLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
            //output the result to channel
            int totalCount = 0;
            for(const auto& yolo : mGenerateProposalKernel)
                totalCount += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);

            return Dims3(totalCount + 1, 1, 1);
    }

    void GenerateProposalLayerPlugin::forwardCpu(const float*const * inputs, float* outputs, cudaStream_t stream)
    {
            auto Logist = [=](float data){
                return 1./(1. + exp(-data));
            };

            CUDA_CHECK(cudaStreamSynchronize(stream));
            int i = 0;
            float* inputData = (float *)mInputBuffer; 
            for(const auto& yolo : mGenerateProposalKernel)
            {
                int size = (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
                CUDA_CHECK(cudaMemcpyAsync(inputData, inputs[i], size * sizeof(float), cudaMemcpyDeviceToHost, stream));
                inputData += size;
                ++ i;
            }

            inputData = (float *)mInputBuffer;
            std::vector <Detection> result;
            for (const auto& yolo : mGenerateProposalKernel)
            {
                int stride = yolo.width*yolo.height;
                for (int j = 0;j < stride ;++j)
                {
                    for (int k = 0;k < CHECK_COUNT; ++k )
                    {
                        int beginIdx = (LOCATIONS + 1 + mClassCount)* stride *k + j;
                        int objIndex = beginIdx + LOCATIONS*stride;
                        
                        //check obj
                        float objProb = Logist(inputData[objIndex]);   
                        if(objProb <= IGNORE_THRESH)
                            continue;

                        //classes
                        int classId = -1;
                        float maxProb = IGNORE_THRESH;
                        for (int c = 0;c< mClassCount;++c){
                            float cProb =  Logist(inputData[beginIdx + (5 + c) * stride]) * objProb;
                            if(cProb > maxProb){
                                maxProb = cProb;
                                classId = c;
                            }
                        }
            
                        if(classId >= 0) {
                            Detection det;
                            int row = j / yolo.width;
                            int cols = j % yolo.width;
    
                            //Location
                            det.bbox[0] = (cols + Logist(inputData[beginIdx]))/ yolo.width;
                            det.bbox[1] = (row + Logist(inputData[beginIdx+stride]))/ yolo.height;
                            det.bbox[2] = exp(inputData[beginIdx+2*stride]) * yolo.anchors[2*k];
                            det.bbox[3] = exp(inputData[beginIdx+3*stride]) * yolo.anchors[2*k + 1];
                            det.classId = classId;
                            det.prob = maxProb;
                            //det.objectness = objProb;

                            result.emplace_back(det);
                        }
                    }
                }

                inputData += (LOCATIONS + 1 + mClassCount) * stride * CHECK_COUNT;
            }

            
            int detCount =result.size();
            auto data = (float *)mOutputBuffer;
            //copy count;
            data[0] = (float)detCount;
            //std::cout << "detCount"<< detCount << std::endl;
            data++;
            //copy result
            memcpy(data,result.data(),result.size()*sizeof(Detection));

            //(count + det result)
            CUDA_CHECK(cudaMemcpyAsync(outputs, mOutputBuffer, sizeof(float) + result.size()*sizeof(Detection), cudaMemcpyHostToDevice, stream));
    };

    __device__ float Logist(float data){ return 1./(1. + exp(-data)); };

    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int stride = yoloWidth*yoloHeight;

        for (int k = 0;k < CHECK_COUNT; ++k )
        {
            int beginIdx = (LOCATIONS + 1 + classes)* stride *k + idx;
            int objIndex = beginIdx + LOCATIONS*stride;
            
            //check objectness
            float objProb = Logist(input[objIndex]);   
            if(objProb <= IGNORE_THRESH)
                continue;

            int row = idx / yoloWidth;
            int cols = idx % yoloWidth;
            
            //classes
            int classId = -1;
            float maxProb = IGNORE_THRESH;
            for (int c = 0;c<classes;++c){
                float cProb =  Logist(input[beginIdx + (5 + c) * stride]) * objProb;
                if(cProb > maxProb){
                    maxProb = cProb;
                    classId = c;
                }
            }

            if(classId >= 0) {
                int resCount = (int)atomicAdd(output,1);
                char* data = (char * )output + sizeof(float) + resCount*sizeof(Detection);
                Detection* det =  (Detection*)(data);

                //Location
                det->bbox[0] = (cols + Logist(input[beginIdx]))/ yoloWidth;
                det->bbox[1] = (row + Logist(input[beginIdx+stride]))/ yoloHeight;
                det->bbox[2] = exp(input[beginIdx+2*stride]) * anchors[2*k];
                det->bbox[3] = exp(input[beginIdx+3*stride]) * anchors[2*k + 1];
                det->classId = classId;
                det->prob = maxProb;
            }
        }
    }
   
    void GenerateProposalLayerPlugin::forwardGpu(const float *const * inputs,float * output,cudaStream_t stream) {
        int numElem;
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

        //first detect count init 0
        CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
        for (unsigned int i = 0;i< mGenerateProposalKernel.size();++i)
        {
            const auto& yolo = mGenerateProposalKernel[i];
            numElem = yolo.width*yolo.height;

            //copy anchor to device
	        CUDA_CHECK(cudaMemcpy(devAnchor,yolo.anchors,AnchorLen,cudaMemcpyHostToDevice));

            CalDetection<<< (yolo.width*yolo.height + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                    (inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount);
        }
        CUDA_CHECK(cudaFree(devAnchor));
    }


    int GenerateProposalLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        assert(batchSize == 1);
        
        //GPU
        forwardGpu((const float *const *)inputs,(float *)outputs[0],stream);

        //CPU
        //forwardCpu((const float *const *)inputs,(float *)outputs[0],stream);
        return 0;
    };

}
/*

TEST(GenerateProposalsTest, TestRealDownSampledGPU) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("GenerateProposals");
  def.add_input("scores");
  def.add_input("bbox_deltas");
  def.add_input("im_info");
  def.add_input("anchors");
  def.add_output("rois");
  def.add_output("rois_probs");
  def.mutable_device_option()->set_device_type(PROTO_CUDA);
  const int img_count = 2;
  const int A = 2;
  const int H = 4;
  const int W = 5;

  vector<float> scores{
      5.44218998e-03f, 1.19207997e-03f, 1.12379994e-03f, 1.17181998e-03f,
      1.20544003e-03f, 6.17993006e-04f, 1.05261997e-05f, 8.91025957e-06f,
      9.29536981e-09f, 6.09605013e-05f, 4.72735002e-04f, 1.13482002e-10f,
      1.50015003e-05f, 4.45032993e-06f, 3.21612994e-08f, 8.02662980e-04f,
      1.40488002e-04f, 3.12508007e-07f, 3.02616991e-06f, 1.97759000e-08f,
      2.66913995e-02f, 5.26766013e-03f, 5.05053019e-03f, 5.62100019e-03f,
      5.37420018e-03f, 5.26280981e-03f, 2.48894998e-04f, 1.06842002e-04f,
      3.92931997e-06f, 1.79388002e-03f, 4.79440019e-03f, 3.41609990e-07f,
      5.20430971e-04f, 3.34090000e-05f, 2.19159006e-07f, 2.28786003e-03f,
      5.16703985e-05f, 4.04523007e-06f, 1.79227004e-06f, 5.32449000e-08f};
  vector<float> bbx{
      -1.65040009e-02f, -1.84051003e-02f, -1.85930002e-02f, -2.08263006e-02f,
      -1.83814000e-02f, -2.89172009e-02f, -3.89706008e-02f, -7.52277970e-02f,
      -1.54091999e-01f, -2.55433004e-02f, -1.77490003e-02f, -1.10340998e-01f,
      -4.20190990e-02f, -2.71421000e-02f, 6.89801015e-03f,  5.71171008e-02f,
      -1.75665006e-01f, 2.30021998e-02f,  3.08554992e-02f,  -1.39333997e-02f,
      3.40579003e-01f,  3.91070992e-01f,  3.91624004e-01f,  3.92527014e-01f,
      3.91445011e-01f,  3.79328012e-01f,  4.26631987e-01f,  3.64892989e-01f,
      2.76894987e-01f,  5.13985991e-01f,  3.79999995e-01f,  1.80457994e-01f,
      4.37402993e-01f,  4.18545991e-01f,  2.51549989e-01f,  4.48318988e-01f,
      1.68564007e-01f,  4.65440989e-01f,  4.21891987e-01f,  4.45928007e-01f,
      3.27155995e-03f,  3.71480011e-03f,  3.60032008e-03f,  4.27092984e-03f,
      3.74579988e-03f,  5.95752988e-03f,  -3.14473989e-03f, 3.52022005e-03f,
      -1.88564006e-02f, 1.65188999e-03f,  1.73791999e-03f,  -3.56074013e-02f,
      -1.66615995e-04f, 3.14146001e-03f,  -1.11830998e-02f, -5.35363983e-03f,
      6.49790000e-03f,  -9.27671045e-03f, -2.83346009e-02f, -1.61233004e-02f,
      -2.15505004e-01f, -2.19910994e-01f, -2.20872998e-01f, -2.12831005e-01f,
      -2.19145000e-01f, -2.27687001e-01f, -3.43973994e-01f, -2.75869995e-01f,
      -3.19516987e-01f, -2.50418007e-01f, -2.48537004e-01f, -5.08224010e-01f,
      -2.28724003e-01f, -2.82402009e-01f, -3.75815988e-01f, -2.86352992e-01f,
      -5.28333001e-02f, -4.43836004e-01f, -4.55134988e-01f, -4.34897989e-01f,
      -5.65053988e-03f, -9.25739005e-04f, -1.06790999e-03f, -2.37016007e-03f,
      -9.71166010e-04f, -8.90910998e-03f, -1.17592998e-02f, -2.08992008e-02f,
      -4.94231991e-02f, 6.63906988e-03f,  3.20469006e-03f,  -6.44695014e-02f,
      -3.11607006e-03f, 2.02738005e-03f,  1.48096997e-02f,  4.39785011e-02f,
      -8.28424022e-02f, 3.62076014e-02f,  2.71668993e-02f,  1.38250999e-02f,
      6.76669031e-02f,  1.03252999e-01f,  1.03255004e-01f,  9.89722982e-02f,
      1.03646003e-01f,  4.79663983e-02f,  1.11014001e-01f,  9.31736007e-02f,
      1.15768999e-01f,  1.04014002e-01f,  -8.90677981e-03f, 1.13103002e-01f,
      1.33085996e-01f,  1.25405997e-01f,  1.50051996e-01f,  -1.13038003e-01f,
      7.01059997e-02f,  1.79651007e-01f,  1.41055003e-01f,  1.62841007e-01f,
      -1.00247003e-02f, -8.17587040e-03f, -8.32176022e-03f, -8.90108012e-03f,
      -8.13035015e-03f, -1.77263003e-02f, -3.69572006e-02f, -3.51580009e-02f,
      -5.92143014e-02f, -1.80795006e-02f, -5.46086021e-03f, -4.10550982e-02f,
      -1.83081999e-02f, -2.15411000e-02f, -1.17953997e-02f, 3.33894007e-02f,
      -5.29635996e-02f, -6.97528012e-03f, -3.15250992e-03f, -3.27355005e-02f,
      1.29676998e-01f,  1.16080999e-01f,  1.15947001e-01f,  1.21797003e-01f,
      1.16089001e-01f,  1.44875005e-01f,  1.15617000e-01f,  1.31586999e-01f,
      1.74735002e-02f,  1.21973999e-01f,  1.31596997e-01f,  2.48907991e-02f,
      6.18605018e-02f,  1.12855002e-01f,  -6.99798986e-02f, 9.58312973e-02f,
      1.53593004e-01f,  -8.75087008e-02f, -4.92327996e-02f, -3.32239009e-02f};
  vector<float> im_info{60, 80, 0.166667f};
  vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};

  // Doubling everything related to images, to simulate
  // num_images = 2
  scores.insert(scores.begin(), scores.begin(), scores.end());
  bbx.insert(bbx.begin(), bbx.begin(), bbx.end());
  im_info.insert(im_info.begin(), im_info.begin(), im_info.end());

  ERMatXf rois_gt(18, 5);
  rois_gt << 0, 0, 0, 79, 59, 0, 0, 5.0005703f, 51.6324f, 42.6950f, 0,
      24.13628387f, 7.51243401f, 79, 45.0663f, 0, 0, 7.50924301f, 67.4779f,
      45.0336, 0, 0, 23.09477997f, 50.61448669f, 59, 0, 0, 39.52141571f,
      51.44710541f, 59, 0, 23.57396317f, 29.98791885f, 79, 59, 0, 0,
      41.90219116f, 79, 59, 0, 0, 23.30098343f, 78.2413f, 58.7287f, 1, 0, 0, 79,
      59, 1, 0, 5.0005703f, 51.6324f, 42.6950f, 1, 24.13628387f, 7.51243401f,
      79, 45.0663f, 1, 0, 7.50924301f, 67.4779f, 45.0336, 1, 0, 23.09477997f,
      50.61448669f, 59, 1, 0, 39.52141571f, 51.44710541f, 59, 1, 23.57396317f,
      29.98791885f, 79, 59, 1, 0, 41.90219116f, 79, 59, 1, 0, 23.30098343f,
      78.2413f, 58.7287f;

  vector<float> rois_probs_gt{2.66913995e-02f,
                              5.44218998e-03f,
                              1.20544003e-03f,
                              1.19207997e-03f,
                              6.17993006e-04f,
                              4.72735002e-04f,
                              6.09605013e-05f,
                              1.50015003e-05f,
                              8.91025957e-06f};

  // Doubling everything related to images, to simulate
  // num_images = 2
  rois_probs_gt.insert(
      rois_probs_gt.begin(), rois_probs_gt.begin(), rois_probs_gt.end());

  AddInput<CUDAContext>(
      vector<int64_t>{img_count, A, H, W}, scores, "scores", &ws);
  AddInput<CUDAContext>(
      vector<int64_t>{img_count, 4 * A, H, W}, bbx, "bbox_deltas", &ws);
  AddInput<CUDAContext>(vector<int64_t>{img_count, 3}, im_info, "im_info", &ws);
  AddInput<CUDAContext>(vector<int64_t>{A, 4}, anchors, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("pre_nms_topN", 6000));
  def.add_arg()->CopyFrom(MakeArgument("post_nms_topN", 300));
  def.add_arg()->CopyFrom(MakeArgument("nms_thresh", 0.7f));
  def.add_arg()->CopyFrom(MakeArgument("min_size", 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("correct_transform_coords", true));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  // test rois
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois_gpu = rois_blob->Get<TensorCUDA>();
  Tensor rois{CPU};
  rois.CopyFrom(rois_gpu);

  EXPECT_EQ(rois.sizes(), (vector<int64_t>{rois_gt.rows(), rois_gt.cols()}));
  auto rois_data =
      Eigen::Map<const ERMatXf>(rois.data<float>(), rois.dim(0), rois.dim(1));
  EXPECT_NEAR((rois_data.matrix() - rois_gt).cwiseAbs().maxCoeff(), 0, 1e-4);

  // test rois_probs
  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs_gpu = rois_probs_blob->Get<TensorCUDA>();
  Tensor rois_probs{CPU};
  rois_probs.CopyFrom(rois_probs_gpu);
  EXPECT_EQ(
      rois_probs.sizes(), (vector<int64_t>{int64_t(rois_probs_gt.size())}));
  auto rois_probs_data =
      ConstEigenVectorArrayMap<float>(rois_probs.data<float>(), rois.dim(0));
  EXPECT_NEAR(
      (rois_probs_data.matrix() - utils::AsEArrXt(rois_probs_gt).matrix())
          .cwiseAbs()
          .maxCoeff(),
      0,
      1e-4);
}  

*/
