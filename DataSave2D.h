#pragma once
#include "all_head.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>

// 适配C++17及以上的文件系统（C++14及以下可替换为boost/filesystem）
namespace fs = std::filesystem;
class DataSave2D : public CtOperation2D
{
public:
    explicit  DataSave2D(string path) :_path(path) {}
    virtual bool execute(CTData2D& ctdata) override;
    virtual string name() const override;
private:
    // 辅助函数：保存单个float数组为二进制.raw文件
    bool saveFloatArrayToRaw(const float* data, size_t elem_count, const fs::path& save_path);
    string _path;
};