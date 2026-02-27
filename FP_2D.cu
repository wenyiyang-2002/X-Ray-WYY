#include "FP_2D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

string FP_2D::name() const
{
    return "2D前向投影......";
}

// 声明常量内存变量，只读的数据最好放常量内存里，有专门的广播机制可以读取
__constant__ float d_SDD;
__constant__ float d_SOD;
__constant__ int d_image_width;
__constant__ int d_image_height;
__constant__ float d_pixel_spacing_x;
__constant__ float d_pixel_spacing_y;
__constant__ float d_detector_spacing;
__constant__ float cu;
__constant__ int d_detector_num;
__constant__ int d_angle_num;

// 边界盒计算射线与图像体相交的设备函数
__device__ bool intersect_ray_box(float S_x, float S_y, float d_x, float d_y,
    float box_edge_x, float box_edge_y,
    float& start_t, float& end_t) {
    // 计算射线与边界平面的交点参数t
    float t_min_x = (-box_edge_x - S_x) / d_x;
    float t_max_x = (box_edge_x - S_x) / d_x;
    float t_min_y = (-box_edge_y - S_y) / d_y;
    float t_max_y = (box_edge_y - S_y) / d_y;

    // 确保t_min <= t_max
    if (t_min_x > t_max_x) {
        float temp = t_min_x;
        t_min_x = t_max_x;
        t_max_x = temp;
    }

    if (t_min_y > t_max_y) {
        float temp = t_min_y;
        t_min_y = t_max_y;
        t_max_y = temp;
    }

    // 计算重叠区间
    start_t = fmaxf(t_min_x, t_min_y);
    end_t = fminf(t_max_x, t_max_y);

    // 如果没有重叠则返回false
    return start_t <= end_t;
}

__global__ void fp2d_kernel(float* image, float* projection, float* projection_matrix) {
    //世界坐标系
    int detector = blockIdx.x * blockDim.x + threadIdx.x;//探测器索引
    int view = blockIdx.y * blockDim.y + threadIdx.y;//视角索引
    if (detector >= d_detector_num || view >= d_angle_num){
        return;//多开的线程不需要计算。
    }
    //计算射线方向
    float ray_x = -d_SDD;
    float ray_y = (cu-detector)*d_detector_spacing;
    float S_x = d_SOD * projection_matrix[view * 8];
    float S_y = d_SOD * -projection_matrix[view * 8 +1];
    float R_x = projection_matrix[view * 8] * ray_x + projection_matrix[view * 8 + 1] * ray_y;
    float R_y = projection_matrix[view * 8 + 3] * ray_x + projection_matrix[view * 8 + 4] * ray_y;
    float r = sqrt(R_x * R_x + R_y * R_y);
    float d_x = R_x / r;
    float d_y = R_y / r;
    //计算射线与图像体相交的参数
    float box_edge_x = (d_image_width - 1) * d_pixel_spacing_x / 2;
    float box_edge_y = (d_image_height - 1) * d_pixel_spacing_y / 2;
    float ray_sum = 0.0;
    float start_t, end_t;
    if (!intersect_ray_box(S_x, S_y, d_x, d_y, box_edge_x, box_edge_y, start_t, end_t)) {
        //投影已经初始化为0。
        return;
    }
    float deta_t = d_pixel_spacing_x / 2;
    for (float current_t = start_t; current_t <= end_t; current_t+= deta_t) {
        //像素坐标系
        float x_index = (S_x + current_t * d_x) / d_pixel_spacing_x + (d_image_width - 1) / 2;
        float y_index = (S_y + current_t * d_y) / d_pixel_spacing_y + (d_image_height - 1) / 2;
        int x_index_down = floor(x_index);
        int y_index_down = floor(y_index);
        int x_index_up = x_index_down + 1;
        int y_index_up = y_index_down + 1;
        if (x_index_down >= 0 && x_index_up < d_image_width &&
            y_index_down >= 0 && y_index_up < d_image_height) {
            float dx = x_index - x_index_down;
            float dy = y_index - y_index_down;
            float v00 = image[x_index_down + y_index_down * d_image_width];
            float v10 = image[x_index_up + y_index_down * d_image_width];
            float v01 = image[x_index_down + y_index_up * d_image_width];
            float v11 = image[x_index_up + y_index_up * d_image_width];
            float val = v00 * (1 - dx) * (1 - dy) +
                v10 * dx * (1 - dy) +
                v01 * (1 - dx) * dy +
                v11 * dx * dy;
            ray_sum += val * deta_t;
        }
    }
    projection[detector + view * d_detector_num] = ray_sum;
}

bool FP_2D::execute(CTData2D& ctdata)
{
    //主机端读取投影矩阵，传至GPU
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
    //启动核函数前需要准备的
    ctdata.projetion_matrix.copy_host_to_device();
    ctdata.image.copy_host_to_device();
    ctdata.projection.allocate_device(ctdata.angle_num*ctdata.detector_num);
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    //设置常量内存变量
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_SDD, &ctdata.SDD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_SOD, &ctdata.SOD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_width, &ctdata.image_width, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_height, &ctdata.image_height, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pixel_spacing_x, &ctdata.pixel_spacing_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pixel_spacing_y, &ctdata.pixel_spacing_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_spacing, &ctdata.detector_spacing, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cu, ctdata.projetion_matrix.get_host_ptr()+7, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num, &ctdata.detector_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_angle_num, &ctdata.angle_num, sizeof(int)));

    //开始前向投影
    dim3 blocksize(32, 32);
    dim3 gridsize(ctdata.detector_num / blocksize.x + 1, ctdata.angle_num / blocksize.y + 1);//没必要多开
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    fp2d_kernel << < gridsize, blocksize, 0, stream >> > (ctdata.image.get_device_ptr(),
        ctdata.projection.get_device_ptr(),
        ctdata.projetion_matrix.get_device_ptr());
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    float time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    // 8. 资源释放与数据拷贝
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    cout<<"FP_2D time:"<<time<<"ms"<<endl;
    ctdata.projection.copy_device_to_host();
    return true;
}