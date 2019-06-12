#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace Yolo;

vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, cv::COLOR_RGB2BGR);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void DoNms(vector<Detection>& detections,int classes ,float nmsThresh)
{
    auto t_start = chrono::high_resolution_clock::now();

    vector<vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };
        
        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i]; 
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    cout << "Time taken for nms is " << total << " ms." << endl;
}

vector<Bbox> postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes)
{
    using namespace cv;

    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }
    
    //nms
    float nmsThresh = parser::getFloatValue("nms");
    if(nmsThresh > 0) 
        DoNms(detections,classes,nmsThresh);

    vector<Bbox> boxes;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        Bbox bbox = 
        { 
            item.classId,   //classId
            max(int((b[0]-b[2]/2.)*width),0), //left
            min(int((b[0]+b[2]/2.)*width),width), //right
            max(int((b[1]-b[3]/2.)*height),0), //top
            min(int((b[1]+b[3]/2.)*height),height), //bot
            item.prob       //score
        };
        boxes.push_back(bbox);
    }

    return boxes;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

int main( int argc, char* argv[] ){
    parser::ADD_ARG_STRING("prototxt",Desc("input yolov3 deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_STRING("caffemodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("calib",Desc("calibration image List"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));
    parser::ADD_ARG_FLOAT("nms",Desc("non-maximum suppression value"),DefaultValue(to_string(NMS_THRESH)));

    //input
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_STRING("evallist",Desc("eval gt list"),DefaultValue(EVAL_LIST),ValueDesc("file"));

    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);

    string deployFile = parser::getStringValue("prototxt");
    string caffemodelFile = parser::getStringValue("caffemodel");

    vector<vector<float>> calibData;
    string calibFileList = parser::getStringValue("calib");
    string mode = parser::getStringValue("mode");
    if(calibFileList.length() > 0 && mode == "int8")
    {   
        cout << "find calibration file,loading ..." << endl;
      
        ifstream file(calibFileList);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << calibFileList << endl;
            exit(-1);
        }

        string strLine;  
        while( getline(file,strLine) )                               
        { 
            cv::Mat img = cv::imread(strLine);
            auto data = prepareImage(img);
            calibData.emplace_back(data);
        } 
        file.close();
    } 
    RUN_MODE run_mode = RUN_MODE::FLOAT32;
    if(mode == "int8")
    {
        if(calibFileList.length() == 0)
            cout << "run int8 please input calibration file, will run in fp32" << endl;
        else
            run_mode = RUN_MODE::INT8;
    }
    else if(mode == "fp16")
    {
        run_mode = RUN_MODE::FLOAT16;
    }
    cout << "2" << endl;	
    string outputNodes = parser::getStringValue("outputs");
    auto outputNames = split(outputNodes,',');
    
    //can load from file
    string saveName = "yolov3_" + mode + ".engine";

#define LOAD_FROM_ENGINE
#ifdef LOAD_FROM_ENGINE    
    trtNet net(saveName);
#else
    trtNet net(deployFile,caffemodelFile,outputNames,calibData,run_mode);
    cout << "save Engine..." << saveName <<endl;
    net.saveEngine(saveName);
#endif 
    int outputCount = net.getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);

    string listFile = parser::getStringValue("evallist");
    list<string> fileNames;
    list<vector<Bbox>> groundTruth;

    if(listFile.length() > 0)
    {
        std::cout << "loading from eval list " << listFile << std::endl; 
        tie(fileNames,groundTruth) = readObjectLabelFileList(listFile);
    }
    else
    {
        string inputFileName = parser::getStringValue("input");
        fileNames.push_back(inputFileName);
    }  
    int classNum = parser::getIntValue("class");

    // host memory for outputs

    int N=1;
    int nmsMaxOut=300;
    static const int OUTPUT_CLS_SIZE = 5+1;
    const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;
    float* rois = new float[N * nmsMaxOut * 4];
    float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
    float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];

    // predicted bounding boxes
    float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];

	cv::VideoCapture cap("/home/lee/workspace/TensorRT-Yolov3/test.ts");  
	// Check if camera opened successfully
	if(!cap.isOpened()){
		cout << "Error opening video stream or file" << endl;
		return -1;
	} 
	while(1){
		list<vector<Bbox>> outputs;
	
		cv::Mat img; 
		cv::Mat img2; 
		cap >> img; 
		img2=img.clone();
		//if (img.empty())
		//	break;  
        vector<float> inputData = prepareImage(img);
        if (!inputData.data())
            continue;  
        net.doInference(inputData.data(), outputData.get()); 
        //Get Output    
        auto output = outputData.get();
        //first detect count
        int count = output[0];
        //later detect result
        vector<Detection> result;
        result.resize(count);
        memcpy(result.data(), &output[1], count*sizeof(Detection));
        auto boxes = postProcessImg(img,result,classNum);
        outputs.emplace_back(boxes);
		//cv::Mat img = cv::imread(*fileNames.begin());
        auto bbox = *outputs.begin();
        for(const auto& item : bbox)
        {
			cout << "item:" << "[" << item.bot - item.top <<  ", " << item.right - item.left  <<  " ]" << endl;
            cv::rectangle(img2,cv::Point(item.left,item.top),cv::Point(item.right,item.bot),cv::Scalar(0,0,255),3,8,0);
            //cout << "class=" << item.classId << " prob=" << item.score*100 << endl;
            //cout << "left=" << item.left << " right=" << item.right << " top=" << item.top << " bot=" << item.bot << endl;
        } 
        cv::imshow("result",img2);
		char c=(char)cv::waitKey(1);
		if(c==27)
			break;
		img.release();
		img2.release();
	} 
    net.printTime();       
	cap.release(); 
	cv::destroyAllWindows();
    return 0;
}
