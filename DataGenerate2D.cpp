#include "DataGenerate2D.h"
#include <fstream>

string DataGenerate2D::name() const
{
	return "生成2D模体中......";
}

bool DataGenerate2D::execute(CTData2D& ctdata) {
    if (_is_load) {
        std::cout << "[执行操作] DataGenerate(2D 读取数据)，路径：" << _load_path << std::endl;

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
    else {//如果不加载数据就生成仿真模体
        std::cout << "[执行操作] DataGenerate(2D Shepp-Logan模体生成)" << std::endl;
        // 1. 初始化参数
        const int width = ctdata.image_width;
        const int height = ctdata.image_height;
        const float pixel_spacing_x = ctdata.pixel_spacing_x;
        const float pixel_spacing_y = ctdata.pixel_spacing_y;
        const int total_pixels = width * height;

        // 初始化图像数据（默认0值）
        ctdata.image.allocate_host(total_pixels);

        // Shepp-Logan模体的10个椭圆参数 [a, b, x0, y0, theta(度), gray_value]
        // a: 长半轴, b: 短半轴, (x0,y0): 椭圆中心, theta: 旋转角度, gray_value: 灰度值
        double ellipses[10][6] = {
            {0.69, 0.92, 0.0, 0.0, 0.0, 2.0},      // 大椭圆(头部外轮廓)
            {0.6624, 0.874, 0.0, -0.0184, 0.0, -1}, // 头部内部
            {0.11, 0.31, 0.22, 0.0, -18.0, -1},     // 右侧椭圆
            {0.16, 0.41, -0.18, 0.0, 18.0, -1},     // 左侧椭圆
            {0.21, 0.25, 0.0, 0.25, 0.0, 0.5},        // 上方椭圆
            {0.046, 0.046, 0.0, 0.1, 0.0, 0.1},       // 小圆1
            {0.046, 0.046, 0.0, -0.1, 0.0, 0.1},      // 小圆2
            {0.046, 0.023, -0.08, -0.305, 0.0, 0.2},  // 小椭圆1
            {0.023, 0.023, 0.0, -0.305, 0.0, 0.3},    // 小圆3
            {0.023, 0.046, 0.06, -0.305, 0.0, -0.2}    // 小椭圆2
        };

        // 2. 遍历每个像素，计算灰度值
        for (int i = 0; i < height; ++i) { // 行索引（y方向）
            for (int j = 0; j < width; ++j) { // 列索引（x方向）
                // 2.1 像素坐标 (i,j) 映射到 世界坐标 (x,y)
                // 假设图像中心为世界坐标原点(0,0)，像素间距为pixel_spacing
                double x = (j - width / 2.0) * pixel_spacing_x; // 列→x轴（右为正）
                double y = (height / 2.0 - i) * pixel_spacing_y; // 行→y轴（上为正，抵消图像行索引从上到下的顺序）

                // 2.2 遍历所有椭圆，判断当前像素是否在椭圆内
                double pixel_value = 0.0;
                for (auto& ellipse : ellipses) {
                    double a = ellipse[0] * width / 2 * pixel_spacing_x;     // 长半轴
                    double b = ellipse[1] * height / 2 * pixel_spacing_y;     // 短半轴
                    double x0 = (ellipse[2] * width) * pixel_spacing_x;    // 椭圆中心x坐标
                    double y0 = (ellipse[3] * height) * pixel_spacing_y;    // 椭圆中心y坐标
                    double theta_deg = ellipse[4]; // 旋转角度（度）
                    double gray = ellipse[5];  // 椭圆灰度值

                    // 2.3 角度转换：度→弧度
                    double theta_rad = theta_deg * M_PI / 180.0;
                    double cos_theta = cos(theta_rad);
                    double sin_theta = sin(theta_rad);

                    // 2.4 椭圆方程（旋转后）：判断点(x,y)是否在椭圆内
                    // 公式：[(x-x0)cosθ + (y-y0)sinθ]^2 / a² + [-(x-x0)sinθ + (y-y0)cosθ]^2 / b² ≤ 1
                    double dx = x - x0;
                    double dy = y - y0;
                    double term1 = (dx * cos_theta + dy * sin_theta) / a;
                    double term2 = (-dx * sin_theta + dy * cos_theta) / b;
                    double ellipse_eq = term1 * term1 + term2 * term2;

                    // 若在椭圆内，累加灰度值
                    if (ellipse_eq <= 1.0 + 1e-6) { // 1e-6 容差，避免浮点误差
                        pixel_value += gray;
                    }
                }

                // 2.5 赋值到图像数据（1D数组索引：i×width + j）
                int pixel_idx = i * width + j;
                ctdata.image.get_host_ptr()[pixel_idx] = static_cast<float>(pixel_value);
            }
        }

        std::cout << "Shepp-Logan模体生成完成！尺寸：" << width << "×" << height << std::endl;
    }
    return true;
}

