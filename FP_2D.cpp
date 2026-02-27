#include "FP_2D.h"

string FP_2D::name() const
{
    return "2D前向投影......";
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
    //需要的数据传至GPU
    ctdata.projetion_matrix.copy_host_to_device();
    ctdata.image.copy_host_to_device();
    //开始前向投影 
}