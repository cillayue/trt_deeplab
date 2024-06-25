#ifndef __TENSORRT_H__
#define __TENSORRT_H__
#include <iostream>
#include <cassert>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "deeplab_trt_model.cpp"
#endif

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

//nvinfer1::ILogger logger;//nvinfer1::ILogger是一个抽象类（abstract class），它不能被直接实例化
// 自定义的日志记录器类，继承自nvinfer1::ILogger  
class MyLogger : public nvinfer1::ILogger {  
public:  
    void log(Severity severity, const char* msg) noexcept override {  
        // 在这里实现你的日志记录逻辑  
        // 例如，可以将日志消息打印到控制台或写入文件  
        std::cout << msg << std::endl;  
    }  
  
    // 如果需要，还可以实现其他纯虚函数，如logVerbose等  
};

int main(int argc, char const *argv[])
{

    // // export CUDA_MODULE_LOADING=LAZY;
    // const char* value = std::getenv("CUDA_MODULE_LOADING");  
    // if (value) {  
    //     std::cout << "CUDA_MODULE_LOADING: " << value << std::endl;  
    // } else {  
    //     std::cout << "CUDA_MODULE_LOADING is not set." << std::endl;  
    // } 

    std::string trtPath = "/media/myue/AHS/c_inference/my_trt/DeepLabV3_F16.trt"; // 需要绝对路经，
    std::string imgpath = "/media/myue/AHS/c_inference/my_trt/test.jpg";
    std::cout << trtPath << std::endl; 
    const int BATCH_SIZE = 1;
    const int CHANNEL = 3;
    const int INPUT_H = 612;
    const int INPUT_W = 816;
    // 创建自定义日志记录器的实例  
    MyLogger logger;

    cudaStream_t stream = nullptr;
    
    Deeplab deeplab(trtPath,logger); 
    std::cout << "声明一个deeplab类" << std::endl; 
    std::vector<uint8_t> trt_file = deeplab.loadModel(trtPath);// char 用来存储单个字节的数据
    deeplab.build(trt_file, logger);


    // 预处理
    //////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat img = cv::imread(imgpath);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(INPUT_W, INPUT_H), CV_32FC3); // 显式创建浮点型三通道Mat 

    float mean[] = {123.675, 116.28, 103.53};
    float std[] = {58.395, 57.12, 57.375};

    // 提前申请内存，可节省推理时间
    static float mydata[BATCH_SIZE * CHANNEL * INPUT_H * INPUT_W];
    for (int r = 0; r < INPUT_H; r++)
    {
        
        for (int c = 0; c < INPUT_W; c++)
        {
            cv::Vec3b &pixel = img_resized.at<cv::Vec3b>(r, c);  
            mydata[0 * INPUT_H * INPUT_W + r * INPUT_W + c] = (pixel[0] - mean[0]) / std[0];  
            mydata[1 * INPUT_H * INPUT_W + r * INPUT_W + c] = (pixel[1] - mean[1]) / std[1];  
            mydata[2 * INPUT_H * INPUT_W + r * INPUT_W + c] = (pixel[2] - mean[2]) / std[2];  
        }
    }
    std::cout << "图片已转换" << std::endl; 
    // std::cout << sizeof(float) << std::endl; //4 float32
     //验证输入数据是否有正确的输入
    // for (int c = 0; c < BATCH_SIZE * CHANNEL * INPUT_H * INPUT_W; c++)
    //     {
    //         std::cout << mydata[c] << std::endl;
    //     }    
    //////////////////////////////////////////////////////////////////////////////////////////
    int input_index = deeplab.GetBindingIndex("input");
   
    deeplab.inference(mydata,CHANNEL * INPUT_H * INPUT_W * sizeof(float),input_index);

    // img img_resize 是否需要delect or free
    // free(mydata);

    return 0;
};






