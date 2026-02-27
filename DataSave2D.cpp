#include "DataSave2D.h"

string DataSave2D::name() const
{
    return "正在保存图像2D";
}

bool DataSave2D::execute(CTData2D& ctdata)
{ 
    saveFloatArrayToRaw(ctdata.image.get_host_ptr(), ctdata.image_size(), _path+"image.raw");
    saveFloatArrayToRaw(ctdata.recon.get_host_ptr(), ctdata.recon_size(), _path+"recon.raw");
    saveFloatArrayToRaw(ctdata.projection.get_host_ptr(), ctdata.projection_size(), _path+"projection.raw");
    if (ctdata.image.get_host_ptr() == nullptr && ctdata.recon.get_host_ptr() == nullptr
        && ctdata.projection.get_host_ptr() == nullptr) {
        std::cerr << "错误：没有数据可保存！" << std::endl;
        return false;
    }
    return true;
}

// 辅助函数：保存单个float数组为二进制.raw文件
bool DataSave2D::saveFloatArrayToRaw(const float* data, size_t elem_count, const fs::path& save_path) {
    // 空数据/空路径检查
    if (data == nullptr) {
        std::cerr << "错误：待保存数组指针为空！路径：" << save_path << std::endl;
        return false;
    }
    if (elem_count == 0) {
        std::cerr << "错误：待保存数组长度为0！路径：" << save_path << std::endl;
        return false;
    }

    // 二进制模式打开文件（ios::binary避免Windows下换行符转换）
    std::ofstream file(save_path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件写入！路径：" << save_path << std::endl;
        return false;
    }

    // 写入原始二进制数据（float的字节流）
    file.write(reinterpret_cast<const char*>(data), elem_count * sizeof(float));
    if (!file.good()) {
        std::cerr << "错误：文件写入失败！路径：" << save_path << std::endl;
        file.close();
        return false;
    }

    // 关闭文件并验证
    file.close();
    if (file.fail()) {
        std::cerr << "错误：文件关闭失败！路径：" << save_path << std::endl;
        return false;
    }

    std::cout << "成功保存.raw文件：" << save_path
        << " | 数据量：" << elem_count << "个float | 大小："
        << elem_count * sizeof(float) << "字节" << std::endl;
    return true;
}