#include "BP_3D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

string BP_3D::name() const {
	return "反投影3D......";
}

//四个内参
__constant__ float d_inter_param_1;
__constant__ float d_inter_param_2;
__constant__ float d_cu;
__constant__ float d_cv;
//其他参数
__constant__ int d_recon_width;
__constant__ int d_recon_height;
__constant__ int d_recon_depth;
__constant__ float d_recon_pixel_spacing_x;
__constant__ float d_recon_pixel_spacing_y;
__constant__ float d_recon_pixel_spacing_z;
__constant__ int d_angle_num;
__constant__ float d_deta_beta;
__constant__ int d_detector_num_u;
__constant__ int d_detector_num_v;
__constant__ float d_detector_spacing_u;
__constant__ float d_detector_spacing_v;

__global__  void bp3d_kernel(float* recon, float* projection, float* projection_matrix) {
    /// 1. 获取线程索引
    int width_id = blockIdx.x*blockDim.x + threadIdx.x;
    int height_id = blockIdx.y*blockDim.y + threadIdx.y;
    int depth_id = blockIdx.z*blockDim.z + threadIdx.z;
    // 2. 边界检查（必须加！避免越界访问导致的数值异常）
    if (width_id >= d_recon_width || height_id >= d_recon_height || depth_id >= d_recon_depth) {
        return;
    }
    //计算本像素对应的投影坐标
    float Pw_x = (width_id - (d_recon_width - 1) / 2) * d_recon_pixel_spacing_x;
    float Pw_y = (height_id - (d_recon_height - 1) / 2) * d_recon_pixel_spacing_y;
    float Pw_z = (depth_id - (d_recon_depth - 1) / 2) * d_recon_pixel_spacing_z;
    float recon_value = 0;
    // 正确：高维在前（depth），低维在后（width），符合C风格内存布局
    int recon_idx = depth_id * d_recon_width * d_recon_height  // 深度维度的内存步长（最大）
        + height_id * d_recon_width               // 高度维度的内存步长（中间）
        + width_id;                                // 宽度维度的内存步长（最小）
    for (int view_num = 0; view_num < d_angle_num; view_num++) {
        int vector_num = view_num * 12;
        int projection_offset = view_num * d_detector_num_u * d_detector_num_v;
        float Xc = Pw_x * projection_matrix[0 + vector_num]
            + Pw_y * projection_matrix[1 + vector_num]
            + Pw_z * projection_matrix[2 + vector_num]
            - projection_matrix[3 + vector_num];
        float Yc = Pw_x * projection_matrix[4 + vector_num]
            + Pw_y * projection_matrix[5 + vector_num]
            + Pw_z * projection_matrix[6 + vector_num]
            - projection_matrix[7 + vector_num];
        float Zc = Pw_x * projection_matrix[8 + vector_num]
            + Pw_y * projection_matrix[9 + vector_num]
            + Pw_z * projection_matrix[10 + vector_num]
            - projection_matrix[11 + vector_num];
        float u = Xc * d_inter_param_1 / Zc + d_cu;
        float v = Yc * d_inter_param_2 / Zc + d_cv;
        //检测是否在有效区域内
        if (u >= 0 && u <= d_detector_num_u - 1 && v >= 0 && v <= d_detector_num_v - 1) { 
            int u_floor = floor(u);
            int v_floor = floor(v);
            int u_ceil = min(u_floor + 1, d_detector_num_u - 1);
            int v_ceil = min(v_floor + 1, d_detector_num_v - 1);
            float deta_u = u - u_floor;
            float deta_v = v - v_floor;
            float projection_v00 = projection[projection_offset + v_floor * d_detector_num_u + u_floor];
            float projection_v01 = projection[projection_offset + v_floor * d_detector_num_u + u_ceil];
            float projection_v10 = projection[projection_offset + v_ceil * d_detector_num_u + u_floor];
            float projection_v11 = projection[projection_offset + v_ceil * d_detector_num_u + u_ceil];
            //双线性插值
            recon_value += projection_v00 * (1 - deta_u) * (1 - deta_v)
                + projection_v01 * deta_u * (1 - deta_v)
                + projection_v10 * (1 - deta_u) * deta_v
                + projection_v11 * deta_u * deta_v;
        }
    }
    if (recon_value < 0) {
        return;
    }
    recon[recon_idx] = recon_value * d_deta_beta;
}

bool BP_3D::execute(CTData3D& ctdata) {
    // 启动核函数前需要准备的
    if (ctdata.projetion_matrix.get_device_ptr() == nullptr) {
        // 从文件读取投影矩阵数据
        ifstream infile("./Config_3D/projection_matrix.raw", ios::binary);
        if (!infile.is_open()) {
            // 文件不存在或无法打开，返回错误
            return false;
        }
        // 分配主机端内存
        ctdata.projetion_matrix.allocate_host(ctdata.angle_num * 12);
        // 读取数据到主机指针
        infile.read(reinterpret_cast<char*>(ctdata.projetion_matrix.get_host_ptr()),
            ctdata.angle_num * 12 * sizeof(float));
        infile.close();
    }
    if (ctdata.projection.get_device_ptr() == nullptr) {
        // 从文件读取投影数据
        ifstream infile("./Output3D/projection.raw", ios::binary);
        if (!infile.is_open()) {
            // 文件不存在或无法打开，返回错误
            return false;
        }
        // 分配主机端内存
        ctdata.projection.allocate_host(ctdata.angle_num * ctdata.detector_num_u * ctdata.detector_num_v);
        // 读取数据到主机指针
        infile.read(reinterpret_cast<char*>(ctdata.projection.get_host_ptr()),
            ctdata.angle_num * ctdata.detector_num_u * ctdata.detector_num_v * sizeof(float));
        infile.close();
    }
    ctdata.projection.copy_host_to_device();
    ctdata.projetion_matrix.copy_host_to_device();
    ctdata.recon.allocate_device(ctdata.recon_size());

    // 开始反投影
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    float inter_param_1 = ctdata.SDD / ctdata.detector_spacing_u;
    float inter_param_2 = ctdata.SDD / ctdata.detector_spacing_v;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_inter_param_1, &inter_param_1, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_inter_param_2, &inter_param_2, sizeof(float)));
    float cu = (ctdata.detector_num_u - 1) / 2 + ctdata.detector_offset_u / ctdata.detector_spacing_u;
    float cv = (ctdata.detector_num_v - 1) / 2 - ctdata.detector_offset_v / ctdata.detector_spacing_v;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cu, &cu, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cv, &cv, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_width, &ctdata.recon_width, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_height, &ctdata.recon_height, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_depth, &ctdata.recon_depth, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_pixel_spacing_x, &ctdata.recon_pixel_spacing_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_pixel_spacing_y, &ctdata.recon_pixel_spacing_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_pixel_spacing_z, &ctdata.recon_pixel_spacing_z, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num_u, &ctdata.detector_num_u, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num_v, &ctdata.detector_num_v, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_angle_num, &ctdata.angle_num, sizeof(int)));
    float deta_beta = 2 * M_PI / ctdata.angle_num;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_deta_beta, &deta_beta, sizeof(float)));
    dim3 block(16, 16, 1);
    // 标准向上取整：(n + block_size - 1) / block_size
    dim3 grid(
        (ctdata.recon_width + block.x - 1) / block.x,
        (ctdata.recon_height + block.y - 1) / block.y,
        (ctdata.recon_depth + block.z - 1) / block.z
    );
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    bp3d_kernel << <grid, block, 0, stream >> > (ctdata.recon.get_device_ptr(), 
        ctdata.projection.get_device_ptr(), ctdata.projetion_matrix.get_device_ptr());
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("fp3d_kernel time: %f ms\n", milliseconds);
    //释放资源
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ctdata.recon.copy_device_to_host();
    return true;
}