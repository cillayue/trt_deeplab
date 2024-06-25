#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h> //是 NVIDIA TensorRT 的一部分，它针对 NVIDIA GPU 进行了优化，使得深度学习模型在 GPU 上的执行速度更快，效率更高
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "nv_utils.h" 
using namespace nvinfer1;
using namespace InferEngine;

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

bool __inline__ check_runtime(cudaError_t e, const char *call, int line, const char *file)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

class Deeplab
{
public:
    Deeplab(std::string model_path, nvinfer1::ILogger &logger){
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~Deeplab()
    {

        int nb_bindings = engine->getNbBindings();
        for (int i = 0; i < nb_bindings; i++)
        {
            if (engine->bindingIsInput(i)) {
                safeCudaFree(mBinding[i]);
            } else {
                safeCudaFreeHost(mBinding[i]);
            }
        }
        cudaStreamDestroy(stream);

    }

// private:
    const int input_w = 816;
    const int input_h = 612;
    const int input_c = 3;
    const int OUTPUT_SIZE = input_w * input_h;
    float mean[3] = {123.675, 116.28, 103.53};
    float std[3] = {58.395, 57.12, 57.375};

    std::vector<std::string> mCustomOutputs;
    std::vector<void*> mBinding;
    std::vector<size_t> mBindingSize;
    std::vector<nvinfer1::Dims> mBindingDims;
    std::vector<nvinfer1::DataType> mBindingDataType;
    std::vector<std::string> mBindingName;

    int mNbInputBindings = 0;
    int mNbOutputBindings = 0;

    nvinfer1::IBuilder *builder;          
    nvinfer1::IRuntime *runtime;          // 有了runtime ,可以用它来反序列化（即加载）一个已优化的模型引擎
    nvinfer1::ICudaEngine *engine;        // 通过 Builder 类来生成推理引擎
    nvinfer1::IExecutionContext *context; // 执行engine中的网络，enqueue 方法将输入数据传递给引擎，并等待引擎完成计算以产生输出
    cudaStream_t stream; // 一个指向CUDA流（stream）的指针，它用于控制GPU上的异步执行顺序。CUDA流是一个执行序列，允许在GPU上并行地执行多个操作，同时保持这些操作之间的依赖关系

    // 加载engine模型
    std::vector<uint8_t> loadModel(std::string &trtPath)
        {
            std::cout << "start to loadmodel " << std::endl; 
            std::ifstream infile(trtPath.c_str(), std::ifstream::binary);
            std::cout << "read model " << std::endl; 
            if (!infile.is_open())
            {
                std::cout << "error " << std::endl; 
                return {};
            }
            // 直接跳到文件末尾来获取文件大小
            infile.seekg(0, std::ios::end);
            size_t fileSize = infile.tellg();
            std::cout << "File size: " << fileSize << " bytes" << std::endl;
            // 根据文件大小预分配缓冲区空间
            std::vector<uint8_t> engine_buf(fileSize);
            // // 回到文件开始位置准备读取
            infile.seekg(0, std::ios::beg);
            // // 一次性读取整个文件到缓冲区
            infile.read((char*)engine_buf.data(), fileSize);//.data(),用于获取指向容器起始位置的指针
            infile.close();
            std::cout << "load sucess" << std::endl;
            return engine_buf;
        }
    void CreateDeviceBuffer()
        {
         // 创建input outbuf buffer
            int nb_bindings = engine->getNbBindings();
            std::cout << "nb_bindings" << std::endl; 
            std::cout << nb_bindings << std::endl; 
            mBinding.resize(nb_bindings);
            mBindingSize.resize(nb_bindings);
            mBindingName.resize(nb_bindings);
            mBindingDims.resize(nb_bindings);
            mBindingDataType.resize(nb_bindings);
            for (int i = 0; i < nb_bindings; i++)
            {
                const char* name = engine->getBindingName(i);
                nvinfer1::DataType dtype = engine->getBindingDataType(i);
                nvinfer1::Dims dims;

                dims = engine->getBindingDimensions(i);
 
                // std::cout << "Dims (nbDims=" << dims.nbDims << "): ";
                //     for (int i = 0; i < dims.nbDims; ++i) {
                //         std::cout << dims.d[i];
                //         if (i < dims.nbDims - 1) std::cout << ", ";
                //     }
                // std::cout << std::endl;   // 1 1 612 816


                int64_t total_size = volume(dims) * getElementSize(dtype);
                // std::cout << "total_size" << std::endl; 
                // std::cout << total_size << std::endl; 
                mBindingSize[i] = total_size;
                mBindingName[i] = name;
                mBindingDims[i] = dims;
                mBindingDataType[i] = dtype;

                if (engine->bindingIsInput(i))
                {
                    mNbInputBindings++;
                    mBinding[i] = safeCudaMallocDevMem(total_size);//分配内存
                }
                else
                {
                    mNbOutputBindings++;
                    mBinding[i] = safeCudaMallocHostMem(total_size);
                }
            }
        }


    bool build(const std::vector<uint8_t>& trtFile, nvinfer1::ILogger &logger)
        {
            std::cout << "start to build " << std::endl; 
            runtime =nvinfer1::createInferRuntime(logger);
            engine = runtime->deserializeCudaEngine(trtFile.data(), trtFile.size());
            context = engine->createExecutionContext();
            CreateDeviceBuffer();
            std::cout << "build success" << std::endl; 
            return true;
        }


    int GetBindingIndex(const std::string& tensor_name) const
        {
            return engine->getBindingIndex(tensor_name.c_str());
        }
    void CopyFromHostToDevice(const float* input, int size, int bindIndex, const cudaStream_t& stream)
        {
            CUDA_CHECK(cudaMemcpyAsync(mBinding[bindIndex], input, size, cudaMemcpyHostToDevice, stream));//段错误 (核心已转储)
            cudaStreamSynchronize(stream);

        }

    void CopyFromDeviceToHost(std::vector<int>& output, int size, int bindIndex, const cudaStream_t& stream)
        {
            CUDA_CHECK(cudaMemcpyAsync(output.data(), mBinding[bindIndex], size, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

    void* GetBindingPtr(const std::string& tensor_name) const
        {
            int idx = GetBindingIndex(tensor_name);
            return mBinding[idx];
        }
        
    void inference( const float *input, int size,int bindIndex)
    {
        CopyFromHostToDevice(input, size, bindIndex, stream);
        auto start = std::chrono::high_resolution_clock::now();
        // std::cout << "start to inference" << std::endl;
        context->enqueueV2(mBinding.data(), stream,nullptr);
        cudaStreamSynchronize(stream); //then you can access output without copy again
        std::cout << "end of  inference" << std::endl;
        auto end1 = std::chrono::high_resolution_clock::now();
        auto ms1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start);
        std::cout << "推理: " << (ms1 / 1000.0).count() << "ms" << std::endl;

        std::vector<std::vector<int>> COLOR_MAP{
        {120, 120, 120}, {180, 120, 120}, {6, 230, 230},   {80, 50, 50}};

        /*
        std::vector<int> y_host(1*816*612);
        int output_index = GetBindingIndex("output");
        int sizeout = 816 * 612 *  sizeof(float);
        CopyFromDeviceToHost(y_host,sizeout , output_index, stream);
        */

        // below is not work

        /*
        float* y_host = (float *)GetBindingPtr("output");

        cv::Mat map = cv::Mat::zeros(612, 816, CV_8UC3);
        for (size_t i = 0; i < 612; i++) {
        for (size_t j = 0; j < 816; j++) {
            int cls_id = y_host[i * 816 + j];
           
            if (cls_id == 1){
                //  std::cout << "cls_id "<< cls_id << std::endl;
            map.at<cv::Vec3b>(i, j) = cv::Vec3b(
                COLOR_MAP[cls_id][2], COLOR_MAP[cls_id][1], COLOR_MAP[cls_id][0]);
        }
        }
        }
        cv::imwrite("_cv_trt.jpg", map);
        */


    }
};
