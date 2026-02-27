#include "GetProjectionMatrix2D.h"

string GetProjectionMatrix2D::name() const
{
    return "生成2D投影矩阵中......";
}

bool GetProjectionMatrix2D::execute(CTData2D& ctdata)
{
    ctdata.projetion_matrix.allocate_host(ctdata.angle_num * 8);
    float deta_beta = 2 * M_PI / ctdata.angle_num;
    for (int view = 0; view < ctdata.angle_num; ++view){
        int view_id = view * 8;
        float beta = view * deta_beta;
        //外参
        ctdata.projetion_matrix.get_host_ptr()[view_id + 0] = cos(beta);
        ctdata.projetion_matrix.get_host_ptr()[view_id + 1] = -sin(beta);
        ctdata.projetion_matrix.get_host_ptr()[view_id + 2] = 0.0f;
        ctdata.projetion_matrix.get_host_ptr()[view_id + 3] = sin(beta);
        ctdata.projetion_matrix.get_host_ptr()[view_id + 4] = cos(beta);
        ctdata.projetion_matrix.get_host_ptr()[view_id + 5] = -ctdata.SOD;
        //内参
        ctdata.projetion_matrix.get_host_ptr()[view_id + 6] = ctdata.SDD / ctdata.detector_spacing;
        ctdata.projetion_matrix.get_host_ptr()[view_id + 7] = -ctdata.detector_offset / ctdata.detector_spacing + (ctdata.detector_num - 1) / 2; //探测器中心（索引）
    }
    // 在for循环结束后添加
    ofstream outfile("./Config_2D/projection_matrix.raw", ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(ctdata.projetion_matrix.get_host_ptr()),
            ctdata.angle_num * 8 * sizeof(float));
        outfile.close();
    }
    return true;
}