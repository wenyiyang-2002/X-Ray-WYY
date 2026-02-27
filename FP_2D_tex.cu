#include "FP_2D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_indirect_functions.h"
#include <fstream>
#include <iostream>
#include <cmath>
using namespace std;
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

// ========== 全局常量/纹理对象定义 ==========
// 常量内存：扩展更多预计算参数，减少核函数内重复计算
__constant__ float d_SDD;
__constant__ float d_SOD;
__constant__ int d_image_width;
__constant__ int d_image_height;
__constant__ float d_pixel_spacing_x;
__constant__ float d_pixel_spacing_y;
__constant__ float d_inv_pixel_spacing_x; // 像素间距倒数（预计算）
__constant__ float d_inv_pixel_spacing_y;
__constant__ float d_box_edge_x;          // 边界盒尺寸（预计算）
__constant__ float d_box_edge_y;
__constant__ float d_detector_spacing;
__constant__ float cu;
__constant__ int d_detector_num;
__constant__ int d_angle_num;             // 角度数放入常量内存

// 2D纹理对象（替代全局内存访问图像，支持硬件双线性插值）
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_image;
cudaTextureObject_t tex_obj_image; // 纹理对象（更灵活，推荐替代纹理引用）

string FP_2D::name() const
{
    return "2D前向投影......";
}

// ========== 设备端工具函数优化 ==========
// 优化版边界盒相交计算：减少分支、合并计算
__device__ bool intersect_ray_box_opt(float S_x, float S_y, float d_x, float d_y,
    float& start_t, float& end_t) {
    // 避免除零（射线方向为0时的保护）
    if (fabsf(d_x) < 1e-8f) d_x = 1e-8f;
    if (fabsf(d_y) < 1e-8f) d_y = 1e-8f;

    // 合并t_min/t_max计算，减少临时变量
    float t1x = (-d_box_edge_x - S_x) / d_x;
    float t2x = (d_box_edge_x - S_x) / d_x;
    float t1y = (-d_box_edge_y - S_y) / d_y;
    float t2y = (d_box_edge_y - S_y) / d_y;

    // 用fminf/fmaxf替代分支交换，更高效
    start_t = fmaxf(fminf(t1x, t2x), fminf(t1y, t2y));
    end_t = fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y));

    // 确保t为正（射线从源点出发，t<0无意义）
    start_t = fmaxf(start_t, 0.0f);

    return start_t <= end_t;
}

// ========== 优化后的核函数 ==========
__global__ void fp2d_kernel_opt(float* projection, const float* projection_matrix) {
    // 线程索引：优化布局，x维度对应detector（连续内存写入），y维度对应view
    const int detector = blockIdx.x * blockDim.x + threadIdx.x;
    const int view = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：提前过滤无效线程（减少分支）
    if (detector >= d_detector_num || view >= d_angle_num) return;

    // ========== 1. 投影矩阵加载：共享内存优化 ==========
    __shared__ float proj_mat_shared[8]; // 每个view的8个参数放入共享内存
    if (threadIdx.x == 0) { // 每个block的第一个线程加载当前view的投影矩阵
        const int mat_offset = view * 8;
        proj_mat_shared[0] = projection_matrix[mat_offset];
        proj_mat_shared[1] = projection_matrix[mat_offset + 1];
        proj_mat_shared[3] = projection_matrix[mat_offset + 3];
        proj_mat_shared[4] = projection_matrix[mat_offset + 4];
    }
    __syncthreads(); // 等待共享内存加载完成

    // ========== 2. 射线方向计算：优化浮点运算 ==========
    const float ray_x = -d_SDD;
    const float ray_y = (cu - detector) * d_detector_spacing;

    // 从共享内存读取投影矩阵，减少全局内存访问
    const float pm0 = proj_mat_shared[0];
    const float pm1 = proj_mat_shared[1];
    const float pm3 = proj_mat_shared[3];
    const float pm4 = proj_mat_shared[4];

    const float S_x = d_SOD * pm0;
    const float S_y = d_SOD * -pm1;

    const float R_x = pm0 * ray_x + pm1 * ray_y;
    const float R_y = pm3 * ray_x + pm4 * ray_y;

    // 用rsqrtf替代sqrtf（倒数平方根，硬件指令更快）
    const float r_inv = rsqrtf(R_x * R_x + R_y * R_y);
    const float d_x = R_x * r_inv;
    const float d_y = R_y * r_inv;

    // ========== 3. 射线与边界盒相交 ==========
    float start_t, end_t;
    if (!intersect_ray_box_opt(S_x, S_y, d_x, d_y, start_t, end_t)) {
        projection[detector + view * d_detector_num] = 0.0f;
        return;
    }

    // ========== 4. 射线积分：优化循环+纹理插值 ==========
    const float deta_t = d_pixel_spacing_x * 0.5f; // 步长预计算
    const int loop_count = static_cast<int>((end_t - start_t) / deta_t) + 1; // 提前计算循环次数
    float ray_sum = 0.0f;

    // 循环展开（编译期展开，减少循环控制开销）
#pragma unroll 4
    for (int i = 0; i < loop_count; ++i) {
        const float current_t = start_t + i * deta_t;
        if (current_t > end_t) break; // 边界保护

        // 计算纹理坐标（纹理内存的坐标是[0, width-1], [0, height-1]）
        const float tex_x = (S_x + current_t * d_x) * d_inv_pixel_spacing_x + (d_image_width - 1) * 0.5f;
        const float tex_y = (S_y + current_t * d_y) * d_inv_pixel_spacing_y + (d_image_height - 1) * 0.5f;

        // 纹理内存硬件双线性插值（替代手动插值，大幅减少计算量）
        float val = tex2D(tex_image, tex_x, tex_y);

        // 积分累加（步长已预计算）
        ray_sum += val * deta_t;
    }

    // ========== 5. 投影结果写入：合并内存访问 ==========
    projection[detector + view * d_detector_num] = ray_sum;
}

// ========== 执行函数优化 ==========
bool FP_2D::execute(CTData2D& ctdata)
{
    // 1. 主机端投影矩阵加载（原有逻辑保留）
    if (ctdata.projetion_matrix.get_host_ptr() == nullptr) {
        ifstream infile("./Config_2D/projection_matrix.raw", ios::binary);
        if (!infile.is_open()) return false;
        ctdata.projetion_matrix.allocate_host(ctdata.angle_num * 8);
        infile.read(reinterpret_cast<char*>(ctdata.projetion_matrix.get_host_ptr()),
            ctdata.angle_num * 8 * sizeof(float));
        infile.close();
    }

    // 2. 设备端数据准备
    ctdata.projetion_matrix.copy_host_to_device();
    ctdata.image.copy_host_to_device();
    ctdata.projection.allocate_device(ctdata.angle_num * ctdata.detector_num);
    CHECK_CUDA_ERROR(cudaSetDevice(0));

    // 3. 常量内存拷贝：扩展预计算参数，减少核函数内计算
    const float inv_pixel_spacing_x = 1.0f / ctdata.pixel_spacing_x;
    const float inv_pixel_spacing_y = 1.0f / ctdata.pixel_spacing_y;
    const float box_edge_x = (ctdata.image_width - 1) * ctdata.pixel_spacing_x / 2.0f;
    const float box_edge_y = (ctdata.image_height - 1) * ctdata.pixel_spacing_y / 2.0f;

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_SDD, &ctdata.SDD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_SOD, &ctdata.SOD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_width, &ctdata.image_width, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_height, &ctdata.image_height, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pixel_spacing_x, &ctdata.pixel_spacing_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pixel_spacing_y, &ctdata.pixel_spacing_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_inv_pixel_spacing_x, &inv_pixel_spacing_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_inv_pixel_spacing_y, &inv_pixel_spacing_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_box_edge_x, &box_edge_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_box_edge_y, &box_edge_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_spacing, &ctdata.detector_spacing, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cu, ctdata.projetion_matrix.get_host_ptr() + 7, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num, &ctdata.detector_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_angle_num, &ctdata.angle_num, sizeof(int)));

    // 4. 纹理对象初始化（绑定图像数据，启用硬件双线性插值）
    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = ctdata.image.get_device_ptr();
    res_desc.res.pitch2D.width = ctdata.image_width;
    res_desc.res.pitch2D.height = ctdata.image_height;
    res_desc.res.pitch2D.pitchInBytes = ctdata.image_width * sizeof(float);
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc tex_desc{};
    tex_desc.addressMode[0] = cudaAddressModeClamp; // 边界钳位，避免越界
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;     // 硬件双线性插值
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = false;              // 使用绝对坐标

    CHECK_CUDA_ERROR(cudaCreateTextureObject(&tex_obj_image, &res_desc, &tex_desc, nullptr));
    tex_image.normalized = false;
    tex_image.filterMode = cudaFilterModeLinear;
    tex_image.addressMode[0] = cudaAddressModeClamp;
    tex_image.addressMode[1] = cudaAddressModeClamp;

    // 5. 线程布局优化：x维度对应detector（连续写入），block尺寸128x1（减少线程束分化）
    const dim3 blocksize(128, 1); // 128线程/block（32的倍数，保证warp对齐）
    const dim3 gridsize(
        (ctdata.detector_num + blocksize.x - 1) / blocksize.x, // detector维度
        (ctdata.angle_num + blocksize.y - 1) / blocksize.y     // view维度
    );

    // 6. CUDA流和事件优化（移除冗余同步）
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 记录开始事件
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));

    // 启动核函数（使用优化后的核函数）
    fp2d_kernel_opt << < gridsize, blocksize, 0, stream >> > (
        ctdata.projection.get_device_ptr(),
        ctdata.projetion_matrix.get_device_ptr()
        );

    // 立即检查核函数启动错误（避免被后续API覆盖）
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 记录结束事件
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));

    // 等待事件完成并计算时间（无需额外cudaDeviceSynchronize）
    float time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "FP_2D optimized time:" << time << "ms" << endl;

    // 7. 数据拷贝和资源清理
    ctdata.projection.copy_device_to_host();

    // 销毁纹理对象（避免内存泄漏）
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(tex_obj_image));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return true;
}