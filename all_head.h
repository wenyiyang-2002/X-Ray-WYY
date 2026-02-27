#pragma once
#include "DeviceBuffer.h"
#include <fstream>
using namespace std;
#define  M_PI   3.14159265358979323846

struct CTData2D {
    // 重建几何参数（可根据需求扩展）
    int image_width = 512;          // 图像宽度
    int image_height = 512;         // 图像高度
    int detector_num = 960;         // 探测器通道数
    int angle_num = 960;            // 投影角度数（0~180°或360°）
    float pixel_spacing_x = 0.25f;     // 体素x方向间距（mm）
    float pixel_spacing_y = 0.25f;     // 体素y方向间距（mm）
    float detector_spacing = 0.25f;  // 探测器间距（mm）
    int recon_width = 512; // 重建宽度
    int recon_height = 512; // 重建高度
    float recon_pixel_spacing_x = 0.25f;
    float recon_pixel_spacing_y = 0.25f;
    float SOD = 300.0f;			   // 源到旋转中心距离（mm）
    float SDD = 500.0f;            // 源到探测器距离（mm）
    float detector_offset = 5.0f;   // 探测器偏移量（mm）
    //内参矩阵外参矩阵
    DeviceBuffer<float> projetion_matrix; //投影矩阵
    DeviceBuffer<float> weight_matrix; //加权矩阵
    DeviceBuffer<float> Filter_matrix; //滤波矩阵

    //数据部分
    DeviceBuffer<float> image;
    DeviceBuffer<float> projection;
    DeviceBuffer<float> recon;

    // 计算投影数据的总大小（角度数 × 探测器数）
    size_t projection_size() const {
        return static_cast<size_t>(angle_num) * detector_num;
    }

    // 计算重建图像的总大小（重建宽 × 重建高）
    size_t recon_size() const {
        return static_cast<size_t>(recon_width) * recon_height;
    }

    // 计算原始图像的总大小
    size_t image_size() const {
        return static_cast<size_t>(image_width) * image_height;
    }
};

struct CTData3D { 
    // 重建几何参数（可根据需求扩展）
    int image_width = 512;           // 图像宽度
    int image_height = 512;         // 图像高度
    int image_depth = 512;          // 图像深度
    float image_pixel_spacing_x = 0.25f;
    float image_pixel_spacing_y = 0.25f;
    float image_pixel_spacing_z = 0.25f;
    int recon_width = 512;           // 重建宽度
    int recon_height = 512;          // 重建高度
    int recon_depth = 512;           // 重建深度
    float recon_pixel_spacing_x = 0.25f;     // 体素x方向间距（mm）
    float recon_pixel_spacing_y = 0.25f;     // 体素y方向间距（mm）
    float recon_pixel_spacing_z = 0.25f;     // 体素z方向间距（mm）
    int detector_num_u = 1080;         // 探测器通道数u
    int detector_num_v = 1080;         // 探测器通道数v
    float detector_offset_u = 0.0f;   // 探测器偏移量u（mm）
    float detector_offset_v = 0.0f;   // 探测器偏移量v（mm）
    float detector_spacing_u = 0.15f;  // 探测器间距u（mm）
    float detector_spacing_v = 0.15f;  // 探测器间距v（mm）
    int angle_num = 960;            // 投影角度数（0~180°或360°）
    float SOD = 800.0f;
    float SDD = 900.0f;
    float Pitch = 0.0f;  //螺旋CT专用参数：螺距（mm）
    int stream_count = 8;
    DeviceBuffer<float> projetion_matrix;
    DeviceBuffer<float> weight_matrix; //加权矩阵
    DeviceBuffer<float> Filter_matrix; //滤波矩阵
    //数据部分
    DeviceBuffer<float> image;
    DeviceBuffer<float> projection;
    DeviceBuffer<float> recon;

    // 计算投影数据总大小
    size_t projection_size() const {
        return static_cast<size_t>(angle_num) * detector_num_u * detector_num_v;
    }

    // 计算重建图像总大小
    size_t recon_size() const {
        return static_cast<size_t>(recon_width) * recon_height * recon_depth;
    }

    // 计算原始图像总大小
    size_t image_size() const {
        return static_cast<size_t>(image_width) * image_height * image_depth;
    }
};

class CtOperation2D
{
public:
    virtual ~CtOperation2D() = default;
    // 统一接口：所有操作都接收CTRecData的引用，直接修改数据（或返回状态码）
    virtual bool execute(CTData2D& ctdata) = 0;
    // 获取操作名称（用于日志/调试）
    virtual string name() const = 0;
};

class CtOperation3D
{
public:
    virtual ~CtOperation3D() = default;
    // 统一接口：所有操作都接收CTRecData的引用，直接修改数据（或返回状态码）
    virtual bool execute(CTData3D& ctdata) = 0;
    // 获取操作名称（用于日志/调试）
    virtual string name() const = 0;
};