#include "BP_2D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

string BP_2D::name() const{
    return "2D反投影......";
}

// 声明常量内存变量，只读的数据最好放常量内存里，有专门的广播机制可以读取
__constant__ float d_SDD;
__constant__ float d_SOD;
__constant__ int d_recon_width;
__constant__ int d_recon_height;
__constant__ float d_recon_pixel_spacing_x;
__constant__ float d_recon_pixel_spacing_y;
__constant__ float d_detector_spacing;
__constant__ float inter_param_1;//内参1
__constant__ float cu;//内参2
__constant__ int d_detector_num;
__constant__ float FOV;
__constant__ int d_angle_num;
__constant__ float d_deta_beta;

//CT图像反投影
__global__ void bp2d_kernel(float* recon, float* projection, float* projection_matrix) { 
    int width_id = blockIdx.y * blockDim.y + threadIdx.y;
    int Height_id = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idx = width_id * d_recon_height + Height_id;

    // 2. 边界检查（必须加！避免越界访问导致的数值异常）
    if (width_id >= d_recon_width || Height_id >= d_recon_height) {
        return;
    }

    // 2. 计算世界坐标系坐标
    float Pw_x = (width_id - (d_recon_width - 1) / 2) * d_recon_pixel_spacing_x;
    float Pw_y = (Height_id - (d_recon_height - 1) / 2) * d_recon_pixel_spacing_y;
    float r = sqrt(Pw_x * Pw_x + Pw_y * Pw_y);

    // FOV外直接置0
    if (r > FOV) {
        recon[pixel_idx] = 0.0f;
        return;
    }
    
    // 3. 累加当前像素的所有视角贡献
    float sum_contribution = 0.0f;
    for (int view_num = 0; view_num < d_angle_num; view_num++) {
        int vector_num = view_num * 8;       // 视角对应的参数矩阵偏移
        int proj_offset = view_num * d_detector_num; // 视角对应的投影数据偏移

        // 4. 计算投影坐标 u
        float Pc_x = Pw_x * projection_matrix[0 + vector_num]
            + Pw_y * projection_matrix[1 + vector_num]
            + projection_matrix[2 + vector_num];
        float Pc_y = Pw_x * projection_matrix[3 + vector_num]
            + Pw_y * projection_matrix[4 + vector_num]
            + projection_matrix[5 + vector_num];
        float u = Pc_x * projection_matrix[6 + vector_num] / Pc_y + cu;

        // 5. 探测器范围检查 + 双线性插值
        if (u >= 0 && u <= d_detector_num - 1) {
            int u_down = floor(u);
            int u_up = min(u_down + 1, d_detector_num - 1); // 避免越界
            //线性插值可以纹理内存优化
            float val_u = projection[u_down + proj_offset] * (u_up - u)
                + projection[u_up + proj_offset] * (u - u_down);
            //计算权重。
            /*float beta = d_deta_beta * view_num;
            float fai = atan2(Pw_y, Pw_x);
            float U = (d_SOD + r * sin(beta - fai)) / d_SDD;
            float weight = 1 / (2 * pow(U, 2));*/
            //赋值
            sum_contribution += val_u;
        }
    }
    // 7. 平均后赋值（无锁，仅当前线程访问该像素）
    if (sum_contribution < 0) sum_contribution = 0;

    recon[pixel_idx] = sum_contribution;
}

bool BP_2D::execute(CTData2D& ctdata){
    // 启动核函数前需要准备的
    if (ctdata.projetion_matrix.get_host_ptr() == nullptr) {
        // 从文件读取投影矩阵数据
        ifstream infile("./Config_2D/projection_matrix.raw", ios::binary);
        if (!infile.is_open()) {
            // 文件不存在或无法打开，返回错误
            return false;
        }
        // 分配主机端内存
        ctdata.projetion_matrix.allocate_host(ctdata.angle_num * 8);
        // 读取数据到主机指针
        infile.read(reinterpret_cast<char*>(ctdata.projetion_matrix.get_host_ptr()),
            ctdata.angle_num * 8 * sizeof(float));
        infile.close();
    }
    ctdata.recon.allocate_device(ctdata.recon_width * ctdata.recon_height);

    //开始反投影
    float half_detector = (ctdata.detector_num - 1) / 2.0f * ctdata.detector_spacing;
    float fov = ctdata.SOD * half_detector / sqrt(pow(ctdata.SDD, 2) + pow(half_detector, 2));
    float deta_beta = 2 * M_PI / ctdata.angle_num;
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_SDD, &ctdata.SDD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_SOD, &ctdata.SOD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_width, &ctdata.recon_width, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_height, &ctdata.recon_height, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_pixel_spacing_x, &ctdata.recon_pixel_spacing_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_recon_pixel_spacing_y, &ctdata.recon_pixel_spacing_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_spacing, &ctdata.detector_spacing, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(inter_param_1, ctdata.projetion_matrix.get_host_ptr()+6, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cu, ctdata.projetion_matrix.get_host_ptr()+7, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num, &ctdata.detector_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(FOV, &fov, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_angle_num, &ctdata.angle_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_deta_beta, &deta_beta, sizeof(float)));
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    dim3 blocksize(32, 32);
    dim3 gridsize(ctdata.recon_width / blocksize.x, ctdata.recon_height / blocksize.y);//这里需要调整集合参数来确保线程数是整数。
    //如果+1的话，「多余的 Block」，这些 Block 中的线程索引越界，计算出错误的坐标 / 投影贡献后写入了有效像素的内存地址，形成小黑块。
    //如果有多余的block，那前面需要判断线程索引是否越界，越界则不进行计算。多的线程块会导致性能的损失，所以最好保持刚好。
    bp2d_kernel << <gridsize, blocksize , 0, stream >> > (ctdata.recon.get_device_ptr(),
        ctdata.projection.get_device_ptr(),
        ctdata.projetion_matrix.get_device_ptr());
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    float time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "BP_2D time:" << time << "ms" << endl;
    // 8. 资源释放与数据拷贝
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    ctdata.recon.copy_device_to_host();
    return true;
}