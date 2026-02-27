#pragma once
#include "all_head.h"
class GetProjectionMatrix3D : public CtOperation3D{
public:
	string name() const override;
	bool execute(CTData3D& ctdata) override;
};

