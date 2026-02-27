#pragma once
#include "all_head.h"

class DataGenerate3D : public CtOperation3D
{
public:
	DataGenerate3D(string load_path) :_load_path(load_path), _is_load(true) {}
	DataGenerate3D() :_is_load(false) {}
	string name() const override;
	bool execute(CTData3D& ctdata) override;
private:
    string _load_path;
    bool _is_load;
};

