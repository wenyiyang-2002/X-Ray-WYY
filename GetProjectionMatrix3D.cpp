#include "GetProjectionMatrix3D.h"

string GetProjectionMatrix3D::name() const
{
    return "生成3D扫描投影矩阵......"; //外参
}

bool GetProjectionMatrix3D::execute(CTData3D& ctdata)
{ 
    ctdata.projetion_matrix.allocate_host(12 * ctdata.angle_num);
    float deta_beta = 2 * M_PI / ctdata.angle_num;
    float deta_pitch = ctdata.Pitch / ctdata.angle_num;
    for (int i = 0; i < ctdata.angle_num; i++){ 
        int index = i * 12;
        float beta = i * deta_beta;
        float pitch = i * deta_pitch;
        //这里只创建外参，内参矩阵直接存到GPU常量内存里。 
        ctdata.projetion_matrix.get_host_ptr()[index] = cos(beta);
        ctdata.projetion_matrix.get_host_ptr()[index + 1] = 0;
        ctdata.projetion_matrix.get_host_ptr()[index + 2] = sin(beta);
        ctdata.projetion_matrix.get_host_ptr()[index + 3] = 0;
        ctdata.projetion_matrix.get_host_ptr()[index + 4] = 0;
        ctdata.projetion_matrix.get_host_ptr()[index + 5] = 1;
        ctdata.projetion_matrix.get_host_ptr()[index + 6] = ctdata.Pitch / ctdata.angle_num;
        ctdata.projetion_matrix.get_host_ptr()[index + 7] = pitch;
        ctdata.projetion_matrix.get_host_ptr()[index + 8] = -sin(beta);
        ctdata.projetion_matrix.get_host_ptr()[index + 9] = 0;
        ctdata.projetion_matrix.get_host_ptr()[index + 10] = cos(beta);
        ctdata.projetion_matrix.get_host_ptr()[index + 11] = -ctdata.SOD;
    }
    // 在for循环结束后添加
    ofstream outfile("./Config_3D/projection_matrix.raw", ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(ctdata.projetion_matrix.get_host_ptr()),
            ctdata.angle_num * 12 * sizeof(float));
        outfile.close();
    }
    return true;
}
