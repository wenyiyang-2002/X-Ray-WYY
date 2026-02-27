#include "FP_3D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

string FP_3D::name() const
{
    return "3D前向投影......";
}

__constant__ int d_detector_num_u;
__constant__ int d_detector_num_v;
__constant__ int d_angle_num;
__constant__ float d_cu;
__constant__ float d_cv;
__constant__ float d_pixel_spacing_u;
__constant__ float d_pixel_spacing_v;
__constant__ float SOD;
__constant__ float SDD;
__constant__ int d_image_width;
__constant__ int d_image_height;
__constant__ int d_image_depth;
__constant__ float d_image_pixel_spacing_x;
__constant__ float d_image_pixel_spacing_y;
__constant__ float d_image_pixel_spacing_z;
__constant__ float d_image_x_max;
__constant__ float d_image_y_max;
__constant__ float d_image_z_max;

// __device__函数：计算射线与包边盒的交点
__device__ bool ray_bbox_intersection(
    float ray_start_x, float ray_start_y, float ray_start_z,
    float ray_dir_x, float ray_dir_y, float ray_dir_z,
    float bbox_min_x, float bbox_min_y, float bbox_min_z,
    float bbox_max_x, float bbox_max_y, float bbox_max_z,
    float& t_min, float& t_max) {

    // 初始化交点参数
    t_min = -FLT_MAX;
    t_max = FLT_MAX;

    // 与x平面相交
    if (fabsf(ray_dir_x) > 1e-6f) {  // 避免除零
        float t1 = (bbox_min_x - ray_start_x) / ray_dir_x;
        float t2 = (bbox_max_x - ray_start_x) / ray_dir_x;

        if (t1 > t2) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }

        t_min = fmaxf(t_min, t1);
        t_max = fminf(t_max, t2);
    }
    else {
        // 射线平行于x平面，检查是否在x范围内
        if (ray_start_x < bbox_min_x || ray_start_x > bbox_max_x) {
            // 射线在包边盒外，不相交
            return false;
        }
    }

    // 与y平面相交
    if (fabsf(ray_dir_y) > 1e-6f) {  // 避免除零
        float t1 = (bbox_min_y - ray_start_y) / ray_dir_y;
        float t2 = (bbox_max_y - ray_start_y) / ray_dir_y;

        if (t1 > t2) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }

        t_min = fmaxf(t_min, t1);
        t_max = fminf(t_max, t2);
    }
    else {
        // 射线平行于y平面，检查是否在y范围内
        if (ray_start_y < bbox_min_y || ray_start_y > bbox_max_y) {
            // 射线在包边盒外，不相交
            return false;
        }
    }

    // 与z平面相交
    if (fabsf(ray_dir_z) > 1e-6f) {  // 避免除零
        float t1 = (bbox_min_z - ray_start_z) / ray_dir_z;
        float t2 = (bbox_max_z - ray_start_z) / ray_dir_z;

        if (t1 > t2) {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }

        t_min = fmaxf(t_min, t1);
        t_max = fminf(t_max, t2);
    }
    else {
        // 射线平行于z平面，检查是否在z范围内
        if (ray_start_z < bbox_min_z || ray_start_z > bbox_max_z) {
            // 射线在包边盒外，不相交
            return false;
        }
    }

    // 检查是否有有效交点
    if (t_min > t_max || t_max < 0) {
        // 没有交点或交点在射源后方
        return false;
    }

    // 确保近端点参数不为负值（在射源后方）
    if (t_min < 0) {
        t_min = 0;
    }

    return true;
}

//三维网格配三维线程块一次求出所有投影
__global__ void fp3d_kernel(float* image, float* projection, float* projection_matrix) { 
     int detector_u = blockIdx.x * blockDim.x + threadIdx.x;//探测器索引u
     int detector_v = blockIdx.y * blockDim.y + threadIdx.y;//探测器索引v
     int angle_num = blockIdx.z;//角度索引
     int angle_num_per_projection = angle_num * 12;
     if (detector_u >= d_detector_num_u || detector_v >= d_detector_num_v || angle_num >= d_angle_num){
         return;//多开的线程不需要计算。
     }
     //世界坐标系
     ////开始计算探测器单元在世界坐标系下的坐标
     //float detector_x = (detector_u - (d_detector_num_u - 1) / 2)*d_pixel_spacing_u;
     //float detector_y = (detector_v - (d_detector_num_v - 1) / 2) * d_pixel_spacing_v;
     //float detector_z = SDD-SOD;
     ////开始计算射源在世界坐标系下的坐标
     float source_x = projection_matrix[angle_num_per_projection + 3];
     float source_y = projection_matrix[angle_num_per_projection + 7];
     float source_z = projection_matrix[angle_num_per_projection + 11];
     ////计算射线方向
     //float ray_x = detector_x - source_x;
     //float ray_y = detector_y - source_y;
     //float ray_z = detector_z - source_z;
     float ray_x = (d_cu - detector_u) * d_pixel_spacing_u;
     float ray_y = (d_cv - detector_v) * d_pixel_spacing_v;
     float ray_z = SDD;
     //计算旋转后的射线方向
     float R_x = projection_matrix[angle_num_per_projection] * ray_x + 
         projection_matrix[angle_num_per_projection + 1] * ray_y + projection_matrix[angle_num_per_projection + 2] * ray_z;
     float R_s_x = projection_matrix[angle_num_per_projection] * source_x +
         projection_matrix[angle_num_per_projection + 1] * source_y + projection_matrix[angle_num_per_projection + 2] * source_z;
     float R_y = projection_matrix[angle_num_per_projection + 4] * ray_x + 
         projection_matrix[angle_num_per_projection + 5] * ray_y + projection_matrix[angle_num_per_projection + 6] * ray_z;
     float R_s_y = projection_matrix[angle_num_per_projection + 4] * source_x +
         projection_matrix[angle_num_per_projection + 5] * source_y + projection_matrix[angle_num_per_projection + 6] * source_z;
     float R_z = projection_matrix[angle_num_per_projection + 8] * ray_x + 
         projection_matrix[angle_num_per_projection + 9] * ray_y + projection_matrix[angle_num_per_projection + 10] * ray_z;
     float R_s_z = projection_matrix[angle_num_per_projection + 8] * source_x +
         projection_matrix[angle_num_per_projection + 9] * source_y + projection_matrix[angle_num_per_projection + 10] * source_z;
     //计算射线旋转后的单位向量
     float r = 1.0f / sqrtf(R_x * R_x + R_y * R_y + R_z * R_z);
     float d_x = R_x * r;
     float d_y = R_y * r;
     float d_z = R_z * r;
     //计算射线方向和图像中心的远端点近端点
     float t_min, t_max;
     if (!ray_bbox_intersection(R_s_x, R_s_y, R_s_z,
         d_x, d_y, d_z,
         -d_image_x_max, -d_image_y_max, -d_image_z_max,
         d_image_x_max, d_image_y_max, d_image_z_max,
         t_min, t_max)) {
         // 射线与包边盒不相交
         return;
     }
     //开始计算射线远端点近端点
     float step = 0.5 * d_image_pixel_spacing_x; //采样步长
     float current_t = t_min;
     float sum = 0;
     while (current_t < t_max) {
         float current_x = R_s_x + current_t * d_x;
         float current_y = R_s_y + current_t * d_y;
         float current_z = R_s_z + current_t * d_z;
         float image_index_x = (current_x / d_image_pixel_spacing_x + (d_image_width - 1) / 2.0f);
         float image_index_y = (current_y / d_image_pixel_spacing_y + (d_image_height - 1) / 2.0f);
         float image_index_z = (current_z / d_image_pixel_spacing_z + (d_image_depth - 1) / 2.0f);
         int x_min = floor(image_index_x);
         int y_min = floor(image_index_y);
         int z_min = floor(image_index_z);
         //判断是否越界
         if (x_min >= 0 && x_min < d_image_width && y_min >= 0 && y_min < d_image_height && z_min >= 0 && z_min < d_image_depth) {
             //线性插值
             float deta_x = image_index_x - x_min;
             float deta_y = image_index_y - y_min;
             float deta_z = image_index_z - z_min;
             float v000 = image[z_min * d_image_width * d_image_height + y_min * d_image_width + x_min];
             float v001 = image[z_min * d_image_width * d_image_height + y_min * d_image_width + x_min + 1];
             float v010 = image[z_min * d_image_width * d_image_height + (y_min + 1) * d_image_width + x_min];
             float v011 = image[z_min * d_image_width * d_image_height + (y_min + 1) * d_image_width + x_min + 1];
             float v100 = image[(z_min + 1) * d_image_width * d_image_height + y_min * d_image_width + x_min];
             float v101 = image[(z_min + 1) * d_image_width * d_image_height + y_min * d_image_width + x_min + 1];
             float v110 = image[(z_min + 1) * d_image_width * d_image_height + (y_min + 1) * d_image_width + x_min];
             float v111 = image[(z_min + 1) * d_image_width * d_image_height + (y_min + 1) * d_image_width + x_min + 1];
             float v_total = v000 * (1 - deta_x) * (1 - deta_y) * (1 - deta_z) +
                 v001 * deta_x * (1 - deta_y) * (1 - deta_z) +
                 v010 * (1 - deta_x) * deta_y * (1 - deta_z) +
                 v100 * (1 - deta_x) * (1 - deta_y) * deta_z +
                 v011 * deta_x * deta_y * (1 - deta_z) +
                 v110 * (1 - deta_x) * deta_y * deta_z +
                 v101 * deta_x * (1 - deta_y) * deta_z +
                 v111 * deta_x * deta_y * deta_z;
             sum += v_total * step;
         }
         current_t = current_t + step;
         }
      projection[angle_num * d_detector_num_v * d_detector_num_u + detector_v * d_detector_num_u + detector_u] = sum;
}

bool FP_3D::execute(CTData3D& ctdata)
{
    //主机端读取投影矩阵，传至GPU
    if (ctdata.projetion_matrix.get_host_ptr() == nullptr) {
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
    //启动核函数前需要准备的
    ctdata.projetion_matrix.copy_host_to_device();
    ctdata.image.copy_host_to_device();
    ctdata.projection.allocate_device(ctdata.angle_num * ctdata.detector_num_v * ctdata.detector_num_u);//zyx

    //开始前向投影
    dim3 blocksize(16, 16, 1); // 先确定减小后的线程块
    dim3 gridsize(
        (ctdata.detector_num_u + blocksize.x - 1) / blocksize.x,  // 向上取整：避免1000无法被16/32整除导致线程缺失
        (ctdata.detector_num_v + blocksize.y - 1) / blocksize.y,
        ctdata.angle_num  // 角度维度直接用ctdata.angle_num，无需+1（多余线程会在核函数内return过滤）
    );
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    //设置常量内存变量
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_width, &ctdata.image_width, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_height, &ctdata.image_height, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_depth, &ctdata.image_depth, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_pixel_spacing_x, &ctdata.image_pixel_spacing_x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_pixel_spacing_y, &ctdata.image_pixel_spacing_y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_pixel_spacing_z, &ctdata.image_pixel_spacing_z, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num_u, &ctdata.detector_num_u, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_detector_num_v, &ctdata.detector_num_v, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_angle_num, &ctdata.angle_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pixel_spacing_u, &ctdata.detector_spacing_u, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_pixel_spacing_v, &ctdata.detector_spacing_v, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(SOD, &ctdata.SOD, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(SDD, &ctdata.SDD, sizeof(float)));
    float cu = (ctdata.detector_num_u - 1) / 2 + ctdata.detector_offset_u / ctdata.detector_spacing_u;
    float cv = (ctdata.detector_num_v - 1) / 2 - ctdata.detector_offset_v / ctdata.detector_spacing_v;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cu, &cu, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cv, &cv, sizeof(float)));
    float image_x_max = (ctdata.image_width - 1) * ctdata.image_pixel_spacing_x / 2.0f;
    float image_y_max = (ctdata.image_height - 1) * ctdata.image_pixel_spacing_y / 2.0f;
    float image_z_max = (ctdata.image_depth - 1) * ctdata.image_pixel_spacing_z / 2.0f;
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_x_max, &image_x_max, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_y_max, &image_y_max, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_z_max, &image_z_max, sizeof(float)));
    //启动核函数
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    fp3d_kernel << <gridsize, blocksize, 0, stream >> > (ctdata.image.get_device_ptr(),
        ctdata.projection.get_device_ptr(),
        ctdata.projetion_matrix.get_device_ptr());
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
    //获取结果
    ctdata.projection.copy_device_to_host();
    return true;
}