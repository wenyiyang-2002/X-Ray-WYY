#include "DataGenerate3D.h"
#include <cmath>

string DataGenerate3D::name() const {
    return "生成三维数据中......";
}

bool DataGenerate3D::execute(CTData3D& ctdata) {
    if (_is_load) {
        std::cout << "[执行操作] DataGenerate(3D 读取数据)，路径：" << _load_path << std::endl;

        // 1. 以二进制模式打开RAW文件（必须用 ios::binary，否则会破坏二进制数据）
        std::ifstream raw_file(_load_path, std::ios::binary | std::ios::in);
        if (!raw_file.is_open()) {
            std::cerr << "[错误] DataGenerate：无法打开文件 " << _load_path << "（路径错误或文件不存在）" << std::endl;
            return false; // 读取失败，返回false
        }

        // 2. 获取文件总字节数（用于分配内存）
        raw_file.seekg(0, std::ios::end); // 指针移到文件末尾
        std::streamsize file_size = raw_file.tellg(); // 获取当前指针位置（即文件大小）
        raw_file.seekg(0, std::ios::beg); // 指针移回文件开头

        // 检查文件是否为空
        if (file_size <= 0) {
            std::cerr << "[错误] DataGenerate：文件 " << _load_path << " 为空或大小异常" << std::endl;
            raw_file.close();
            return false;
        }

        // 3. 分配内存并读取数据到 ctdata.image
        // 注意：若 CTData2D::image 不是 vector<unsigned char>，需根据实际类型调整（如 short/float）
        ctdata.image.allocate_host(static_cast<float>(file_size)); // 调整向量大小以容纳文件数据
        // 读取文件内容到 image 的内存地址（reinterpret_cast 用于类型转换）
        raw_file.read(reinterpret_cast<char*>(ctdata.image.get_host_ptr()), file_size);

        // 4. 检查读取是否完整
        if (!raw_file) {
            std::cerr << "[错误] DataGenerate：读取文件失败！已读取 " << raw_file.gcount() << " 字节（预期 " << file_size << " 字节）" << std::endl;
            raw_file.close();
            return false;
        }

        // 5. 读取成功，关闭文件并输出日志
        raw_file.close();
        std::cout << "[成功] DataGenerate：RAW文件读取完成，数据大小：" << file_size << " 字节" << std::endl;
    }
    else {
        // 若不是读取模式，则生成三维Shepp-Logan体数据
        std::cout << "[执行操作] DataGenerate(3D 生成Shepp-Logan体数据)" << std::endl;
        int width = ctdata.image_width;
        int height = ctdata.image_height;
        int slice_num = ctdata.image_depth;
        int total_size = slice_num * height * width;
        ctdata.image.allocate_host(total_size);

        // 初始化为0
        float* data_ptr = ctdata.image.get_host_ptr();
        for (int i = 0; i < total_size; ++i) {
            data_ptr[i] = 0.0f;
        }

        // 定义三维Shepp-Logan模型的椭圆参数矩阵 (10 x 8)
        // 每行代表一个椭球体的参数：
        // [x0, y0, z0, a, b, c, alpha, intensity]
        float ellipsoid_params[10][8] = {
            // 主椭球体
            {0.0f, 0.0f, 0.0f, 0.69f, 0.92f, 0.9f, 0.0f, 1.0f},
            // 内部椭球体
            {0.0f, -0.0184f, 0.0f, 0.6624f, 0.874f, 0.88f, 0.0f, -0.8f},
            // 右上椭球体
            {0.22f, 0.0f, -0.25f, 0.41f, 0.16f, 0.21f, 108.0f * M_PI / 180.0f, -0.2f},
            // 左上椭球体
            {-0.22f, 0.0f, -0.25f, 0.41f, 0.16f, 0.21f, -108.0f * M_PI / 180.0f, -0.2f},
            // 中心右侧椭球体
            {0.0f, 0.35f, -0.25f, 0.25f, 0.23f, 0.5f, 0.0f, -0.2f},
            // 中心左侧椭球体
            {0.0f, 0.1f, -0.25f, 0.23f, 0.05f, 0.2f, 0.0f, -0.2f},
            // 右下椭球体
            {0.0f, -0.1f, -0.25f, 0.22f, 0.02f, 0.2f, 0.0f, -0.1f},
            // 左下椭球体
            {0.0f, -0.3f, -0.25f, 0.2f, 0.04f, 0.2f, 0.0f, -0.1f},
            // 右上角椭球体
            {0.0f, 0.0f, 0.35f, 0.15f, 0.05f, 0.05f, 0.0f, 0.1f},
            // 左下角椭球体
            {0.0f, 0.0f, 0.35f, 0.15f, 0.05f, 0.05f, 90.0f * M_PI / 180.0f, 0.1f}
        };

        // 计算三维Shepp-Logan模型
        for (int z = 0; z < slice_num; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // 将像素坐标转换为标准化坐标 [-1, 1]
                    float x_norm = (2.0f * x / (width - 1)) - 1.0f;
                    float y_norm = (2.0f * y / (height - 1)) - 1.0f;
                    float z_norm = (2.0f * z / (slice_num - 1)) - 1.0f;

                    float value = 0.0f;

                    // 检查每个椭球体
                    for (int i = 0; i < 10; ++i) {
                        // 使用矩阵索引获取椭球体参数
                        float x0 = ellipsoid_params[i][0];  // 中心x坐标
                        float y0 = ellipsoid_params[i][1];  // 中心y坐标
                        float z0 = ellipsoid_params[i][2];  // 中心z坐标
                        float a = ellipsoid_params[i][3];   // x半轴长度
                        float b = ellipsoid_params[i][4];   // y半轴长度
                        float c = ellipsoid_params[i][5];   // z半轴长度
                        float alpha = ellipsoid_params[i][6]; // 旋转角度
                        float intensity = ellipsoid_params[i][7]; // 强度值

                        // 将坐标转换到椭球体局部坐标系
                        float x_trans = x_norm - x0;
                        float y_trans = y_norm - y0;
                        float z_trans = z_norm - z0;

                        // 旋转坐标系（简化：只考虑z轴旋转）
                        float cos_alpha = cos(alpha);
                        float sin_alpha = sin(alpha);
                        float x_rot = x_trans * cos_alpha + y_trans * sin_alpha;
                        float y_rot = -x_trans * sin_alpha + y_trans * cos_alpha;
                        float z_rot = z_trans;

                        // 检查点是否在椭球体内
                        float dist = (x_rot * x_rot) / (a * a) +
                            (y_rot * y_rot) / (b * b) +
                            (z_rot * z_rot) / (c * c);

                        if (dist <= 1.0f) {
                            value += intensity;
                        }
                    }

                    // 设置体数据值
                    int index = z * width * height + y * width + x;
                    data_ptr[index] = value;
                }
            }
        }

        std::cout << "[成功] DataGenerate：三维Shepp-Logan体数据生成完成，尺寸："
            << width << "x" << height << "x" << slice_num
            << "，总大小：" << total_size << " 像素" << std::endl;
    }
    return true;
}