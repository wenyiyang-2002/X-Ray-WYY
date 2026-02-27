#pragma once
#include "all_head.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
namespace fs = std::filesystem;

class DataSave3D : public CtOperation3D{
public:
    explicit  DataSave3D(string path) :_path(path) {}
	string name() const override;
    bool execute(CTData3D& ctdata) override;
private:
	bool saveFloatArrayToRaw(const float* data, size_t elem_count, const fs::path& save_path);
	string _path;
};

