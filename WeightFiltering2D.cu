#include "WeightFiltering2D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
using namespace std;

// 补充缺失的常量定义（原代码中memcpyToSymbol但未定义）
__constant__ int d_detector_num;   // 探测器通道数
__constant__ int d_angle_num;      // 角度数量

string WeightFiltering2D::name() const {
    return "加权滤波2D......";
}

/**
 * @brief 2D加权滤波CUDA核函数（原地操作：结果直接写回projection）
 * @param projection 输入/输出投影数据 [angle_num * detector_num]
 * @param weight_matrix 加权矩阵 [detector_num]
 */
__global__ void weight2d_kernel(
    float* projection,
    float* weight_matrix
) {
    // 计算当前线程对应的探测器通道u和角度v（索引从0开始）
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    // 严格边界检查：超出有效索引范围则退出
    if (u >= d_detector_num || v >= d_angle_num) {
        return;
    }

    // 计算当前投影点的一维索引
    int idx = v * d_detector_num + u;

    // 1. 第一步：对原始投影值做加权（必须先读取原始值，避免覆盖后出错）
    float original_proj = projection[idx];
    float weighted_value = original_proj * weight_matrix[u];

    // 3. 关键：将最终滤波结果写回原投影数组（原地更新）
    projection[idx] = weighted_value;
}

/**
 * @brief CUDA 1D卷积核函数（沿探测器维度卷积，角度维度并行）
 * @param projection 输入投影数据 [angle_num * detector_num]（加权后的数据）
 * @param result 输出卷积结果 [angle_num * detector_num]
 * @param filter_matrix 滤波矩阵 [2*detector_num]（中心在d_detector_num处）
 */
__global__ void filter2d_kernel(
    float* projection,
    float* result,
    float* filter_matrix
) {
    // 2. 计算当前线程对应的探测器u和角度v
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (u >= d_detector_num || v >= d_angle_num) {
        return;
    }

    // 3. 卷积核心计算
    float conv_sum = 0.0f;
    int filter_center = d_detector_num;           // 滤波核中心位置
    int max_det_idx = d_detector_num - 1;         // 探测器最大索引
    int proj_row_start = v * d_detector_num;      // 当前角度v的投影数据起始索引

    // 遍历滤波核所有元素进行卷积运算
    for (int k = 0; k < d_detector_num * 2; ++k) {
        // 计算在原始投影数据中的对应位置
        int det_offset = u + (k - filter_center);

        // 边界处理：如果超出探测器范围则跳过
        if (det_offset >= 0 && det_offset <= max_det_idx) {
            // 获取投影数据和滤波核值进行乘积累加
            float proj_val = projection[proj_row_start + det_offset];//内存合并
            float filter_val = filter_matrix[k];//内存合并
            conv_sum += proj_val * filter_val;
        }
    }

    // 4. 将卷积结果写入输出数组
    int output_idx = proj_row_start + u;
    result[output_idx] = conv_sum;
}

bool WeightFiltering2D::execute(CTData2D& ctdata) {
    // 1. 检查/生成加权&滤波矩阵
    std::ifstream file("./Config_2D/weight_matrix.raw");
    std::ifstream file2("./Config_2D/filter_matrix.raw");
    if (!file.good() || !file2.good()) {
        Gen_Weight(ctdata);
        Gen_Filter_Kernel(ctdata, _window_name);
    }

    // 2. 读取矩阵到Host内存
    ctdata.weight_matrix.allocate_host(ctdata.detector_num);
    ctdata.Filter_matrix.allocate_host(ctdata.detector_num * 2);

    // 读取权重矩阵
    std::ifstream weight_file("./Config_2D/weight_matrix.raw", std::ios::binary);
    weight_file.read(reinterpret_cast<char*>(ctdata.weight_matrix.get_host_ptr()),
        ctdata.detector_num * sizeof(float));
    weight_file.close();

    // 读取滤波矩阵
    std::ifstream filter_file("./Config_2D/filter_matrix.raw", std::ios::binary);
    filter_file.read(reinterpret_cast<char*>(ctdata.Filter_matrix.get_host_ptr()),
        ctdata.detector_num * 2 * sizeof(float));
    filter_file.close();

    // 3. 内核启动前准备
    ctdata.weight_matrix.copy_host_to_device();
    ctdata.Filter_matrix.copy_host_to_device();

    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num, &ctdata.detector_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_angle_num, &ctdata.angle_num, sizeof(int)));


    // 线程块/网格配置
    dim3 block_size(32, 32);
    dim3 grid_size((ctdata.detector_num + block_size.x - 1) / block_size.x,
        (ctdata.angle_num + block_size.y - 1) / block_size.y);

    // 创建CUDA流和计时事件
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 4. 分配Device端卷积结果内存（关键：原代码缺失）
    float* d_result = nullptr;
    size_t proj_total_size = ctdata.detector_num * ctdata.angle_num * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, proj_total_size));

    // 5. 启动核函数（计时）
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));

    // 第一步：加权核函数（原地更新projection）
    weight2d_kernel << <grid_size, block_size, 0, stream >> > (
        ctdata.projection.get_device_ptr(),
        ctdata.weight_matrix.get_device_ptr()
        );
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 第二步：卷积核函数（结果写入d_result）
    filter2d_kernel << <grid_size, block_size, 0, stream >> > (
        ctdata.projection.get_device_ptr(),
        d_result,  // 修正：传入结果的Device指针
        ctdata.Filter_matrix.get_device_ptr()
        );
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 第三步：将卷积结果写回投影数据的Device内存（覆盖原始数据）
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        ctdata.projection.get_device_ptr(),  // 目标：原始投影Device指针
        d_result,                            // 源：卷积结果Device指针
        proj_total_size,                     // 数据大小
        cudaMemcpyDeviceToDevice,            // 拷贝方向：Device->Device
        stream                               // 关联流
    ));

    // 6. 等待计算完成并计时
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));  // 流同步（替代deviceSync，更轻量）

    float time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "加权滤波2D:" << time << "ms" << endl;

    // 7. 将更新后的投影数据从Device拷贝回Host（关键：更新主机端数据）
    //ctdata.projection.copy_device_to_host();

    // 8. 资源释放
    CHECK_CUDA_ERROR(cudaFree(d_result));  // 释放临时结果内存
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return true;
}

// 生成加权矩阵
void WeightFiltering2D::Gen_Weight(CTData2D& ctdata) {
    ctdata.weight_matrix.allocate_host(ctdata.detector_num);
    for (int i = 0; i < ctdata.detector_num; i++) {
        float s = ((ctdata.detector_num - 1) / 2 - i) * ctdata.detector_spacing - ctdata.detector_offset;
        // 防止除零错误（边界保护）
        float denom = sqrt(s * s + ctdata.SDD * ctdata.SDD);
        ctdata.weight_matrix.get_host_ptr()[i] = (denom > 1e-6) ? (ctdata.SOD / denom) : 0.0f;
    }

    // 写入加权矩阵文件
    std::ofstream file("./Config_2D/weight_matrix.raw", std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(ctdata.weight_matrix.get_host_ptr()),
            ctdata.detector_num * sizeof(float));
        file.close();
    }
    else {
        cerr << "生成加权矩阵文件失败！" << endl;
    }
}

void WeightFiltering2D::Gen_Filter_Kernel(CTData2D& ctdata, string window_name) {
    // 滤波核长度 = 探测器像素数 × 2（FBP标准配置）
    int filter_length = ctdata.detector_num * 2;

    int center = filter_length / 2;       // 滤波核中心索引
    int start = -center;                  // 偏移量起始（相对中心）
    int end = (filter_length % 2 == 0) ? center - 1 : center; // 偏移量结束

    // 1. 生成基础Shepp-Logan滤波核（无窗）
    std::vector<float> base_filter(filter_length, 0.0f);
    float d = ctdata.detector_spacing;    // 探测器单元间距（物理单位）

    for (int i = start; i <= end; ++i) {
        int idx = i + center; // 转换为数组索引
        if (i == 0) {
            // 中心位置：Shepp-Logan核公式（连续形式在t=0处的离散近似）
            base_filter[idx] = (1.0f / (4.0f * d * d));
        }
        else if (i % 2 != 0) {
            // 奇数偏移：抑制低频、增强边缘（高频补偿）
            float denominator = static_cast<float>(i * i) * M_PI * M_PI * d * d;
            base_filter[idx] = -1.0f / denominator;
        }
        // 偶数偏移（非0）：值为0，无需额外处理
    }

    // 2. 生成对应窗函数（用于抑制滤波核旁瓣，减少重建伪影）
    std::vector<float> window_function(filter_length, 1.0f); // 默认矩形窗
    if (window_name == "WINDOW_HANN") {
        // 汉宁窗（余弦窗）：无截断，旁瓣抑制中等，分辨率损失小
        for (int i = 0; i < filter_length; ++i) {
            float theta = 2.0f * M_PI * static_cast<float>(i) / (filter_length - 1);
            window_function[i] = 0.5f * (1.0f - cos(theta));
        }
    }
    else if (window_name == "WINDOW_HAMMING") {
        // 汉明窗：有直流偏移，旁瓣抑制比汉宁窗强，分辨率损失略大
        for (int i = 0; i < filter_length; ++i) {
            float theta = 2.0f * M_PI * static_cast<float>(i) / (filter_length - 1);
            window_function[i] = 0.54f - 0.46f * cos(theta); // 标准汉明窗系数
        }
    }
    else if (window_name == "WINDOW_BLACKMAN") {
        // 布莱克曼窗：三阶余弦窗，旁瓣抑制最强，分辨率损失最大（适合低噪声需求）
        for (int i = 0; i < filter_length; ++i) {
            float theta = 2.0f * M_PI * static_cast<float>(i) / (filter_length - 1);
            window_function[i] = 0.42f - 0.5f * cos(theta) + 0.08f * cos(2.0f * theta);
        }
    }

    // 3. 窗函数加权滤波核（核心步骤：抑制旁瓣，平衡分辨率与伪影）
    std::vector<float> final_filter(filter_length);
    for (int i = 0; i < filter_length; ++i) {
        final_filter[i] = base_filter[i] * window_function[i];
    }

    // 4. 保存滤波核到文件
    ctdata.Filter_matrix.allocate_host(filter_length);
    for (int i = 0; i < filter_length; i++) {
        ctdata.Filter_matrix.get_host_ptr()[i] = final_filter[i];
    }

    std::ofstream file("./Config_2D/filter_matrix.raw", std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(ctdata.Filter_matrix.get_host_ptr()),
            filter_length * sizeof(float));
        file.close();
    }
    else {
        cerr << "生成滤波矩阵文件失败！" << endl;
    }
}