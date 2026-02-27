#pragma once
#include "all_head.h"
class DataGenerate2D :public CtOperation2D
{
public:
	DataGenerate2D(string load_path)
		: _is_load(true), _load_path(load_path) {
	}
	DataGenerate2D() :_is_load(false) {}
	bool execute(CTData2D& ctdata) override;
	string name() const override;
private:
	bool _is_load;
	string _load_path;
};