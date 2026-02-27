#include "WeightFiltering3D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

string WeightFiltering3D::name() const {
    return "加权滤波3D......";
}

__constant__ int d_detector_num_u;
__constant__ int d_detector_num_v;

__global__ void weight3d_kernel(
    float* projection,
    float* weight_matrix
) { 
    int detector_u = blockIdx.x * blockDim.x + threadIdx.x;
    int detector_v = blockIdx.y * blockDim.y + threadIdx.y;
    int angle_index = blockIdx.z;
    if (detector_u >= d_detector_num_u || detector_v >= d_detector_num_v)
        return;
    projection[angle_index * d_detector_num_u * d_detector_num_v + detector_v * d_detector_num_u + detector_u]
        *= weight_matrix[detector_v * d_detector_num_u + detector_u];
}

__global__ void filter3d_kernel(
    float* projection,
    float* result,
    float* filter_matrix
) {
    int detector_u = blockIdx.x * blockDim.x + threadIdx.x;
    int detector_v = blockIdx.y * blockDim.y + threadIdx.y;
    int angle_index = blockIdx.z;
    if (detector_u >= d_detector_num_u || detector_v >= d_detector_num_v)
        return;
    // 3. 卷积核心计算
    float conv_sum = 0.0f;
    int filter_center = d_detector_num_u;           // 滤波核中心位置
    int max_det_idx = d_detector_num_u - 1;         // 探测器最大索引
    int proj_row_start = angle_index * d_detector_num_u * d_detector_num_v + detector_v * d_detector_num_u;// 当前角度v的投影数据起始索引
    // 遍历滤波核所有元素进行卷积运算
    for (int k = 0; k < d_detector_num_u * 2; ++k) {
        // 计算u方向偏移（相对当前探测器像素）
        int det_offset = detector_u + (k - filter_center);

        // 镜像延拓优化：用数学运算替代if-else，提升核函数效率
        det_offset = abs(det_offset);  // 左边界镜像
        det_offset = (det_offset > max_det_idx) ? (2 * max_det_idx - det_offset) : det_offset;  // 右边界镜像

        // 3. 卷积求和（滤波核从常量内存读取，速度远快于全局内存）
        float proj_val = projection[proj_row_start + det_offset];
        float filter_val = filter_matrix[k];
        conv_sum += proj_val * filter_val;
    }
    // 4. 将卷积结果写入输出数组
    int output_idx = proj_row_start + detector_u;
    result[output_idx] = conv_sum;
}

bool WeightFiltering3D::execute(CTData3D& ctdata) { 
    // 1. 检查/生成加权&滤波矩阵
    std::ifstream file("./Config_3D/weight_matrix.raw");
    std::ifstream file2("./Config_3D/filter_matrix.raw");
    if (!file.good() || !file2.good()) {
        Gen_Weight(ctdata);
        Gen_Filter_Kernel(ctdata, _window_name);
    }

    // 2. 读取矩阵到Host内存
    ctdata.weight_matrix.allocate_host(ctdata.detector_num_u * ctdata.detector_num_v);
    ctdata.Filter_matrix.allocate_host(ctdata.detector_num_u * 2);

    // 读取权重矩阵
    std::ifstream weight_file("./Config_3D/weight_matrix.raw", std::ios::binary);
    weight_file.read(reinterpret_cast<char*>(ctdata.weight_matrix.get_host_ptr()),
        ctdata.detector_num_u * ctdata.detector_num_v * sizeof(float));
    weight_file.close();

    // 读取滤波矩阵
    std::ifstream filter_file("./Config_3D/filter_matrix.raw", std::ios::binary);
    filter_file.read(reinterpret_cast<char*>(ctdata.Filter_matrix.get_host_ptr()),
        ctdata.detector_num_u * 2 * sizeof(float));
    filter_file.close();

    // 3. 内核启动前准备
    ctdata.weight_matrix.copy_host_to_device();
    ctdata.Filter_matrix.copy_host_to_device();

    // 4. 启动内核
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num_u, &ctdata.detector_num_u, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num_v, &ctdata.detector_num_v, sizeof(int)));
    dim3 block_size(16, 16, 1);
    dim3 grid_size(ctdata.detector_num_u / block_size.x + 1, ctdata.detector_num_v / block_size.y + 1, ctdata.angle_num);
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    //加权
    weight3d_kernel << <grid_size, block_size, 0, stream >> > (
        ctdata.projection.get_device_ptr(),
        ctdata.weight_matrix.get_device_ptr()
        );
    CHECK_CUDA_ERROR(cudaGetLastError());
    //滤波
    float* d_result = nullptr;
    size_t proj_total_size = ctdata.detector_num_u * ctdata.detector_num_v * ctdata.angle_num * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, proj_total_size));
    filter3d_kernel << <grid_size, block_size, 0, stream >> > (
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

    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));  // 流同步（替代deviceSync，更轻量）

    float time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "加权滤波3D:" << time << "ms" << endl;

    // 7. 将更新后的投影数据从Device拷贝回Host（关键：更新主机端数据）
    ctdata.projection.copy_device_to_host();

    // 8. 资源释放
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return true;
}

void WeightFiltering3D::Gen_Weight(CTData3D& ctdata) {
    int one_projection_size = ctdata.detector_num_u * ctdata.detector_num_v;
    ctdata.weight_matrix.allocate_host(one_projection_size);
    float SOD = ctdata.SOD;
    float SDD_sq = ctdata.SDD * ctdata.SDD; // 预计算SDD平方，提升效率
    for (int i = 0; i < ctdata.detector_num_u; i++) {
        for (int j = 0; j < ctdata.detector_num_v; j++) {
            float s1 = ((ctdata.detector_num_u - 1.0f) / 2.0f - i) * ctdata.detector_spacing_u - ctdata.detector_offset_u;
            float s2 = ((ctdata.detector_num_v - 1.0f) / 2.0f - j) * ctdata.detector_spacing_v - ctdata.detector_offset_v;
            // 修正：FDK正确锥束加权（平方项，简化计算，无额外开方）
            float denominator = SDD_sq + s1 * s1 + s2 * s2;
            float weight = SOD / sqrt(denominator);
            // 避免分母为0（极端情况）
            if (denominator < 1e-6f) weight = 0.0f;
            ctdata.weight_matrix.get_host_ptr()[j * ctdata.detector_num_u + i] = weight;
        }
    }
    // 写入加权矩阵文件
    std::ofstream file("./Config_3D/weight_matrix.raw", std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(ctdata.weight_matrix.get_host_ptr()),
            one_projection_size * sizeof(float));
        file.close();
    }
    else {
        cerr << "生成加权矩阵文件失败！" << endl;
    }
}

void WeightFiltering3D::Gen_Filter_Kernel(CTData3D& ctdata, string window_name) {
    // 滤波核长度 = 探测器像素数 × 2（FBP标准配置）
    int filter_length = ctdata.detector_num_u * 2;

    int center = filter_length / 2;       // 滤波核中心索引
    int start = -center;                  // 偏移量起始（相对中心）
    int end = (filter_length % 2 == 0) ? center - 1 : center; // 偏移量结束

    // 1. 生成基础Shepp-Logan滤波核（无窗）
    std::vector<float> base_filter(filter_length, 0.0f);
    float d = ctdata.detector_spacing_u;    // 探测器单元间距（物理单位）

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

    std::ofstream file("./Config_3D/filter_matrix.raw", std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(ctdata.Filter_matrix.get_host_ptr()),
            filter_length * sizeof(float));
        file.close();
    }
    else {
        cerr << "生成滤波矩阵文件失败！" << endl;
    }
}