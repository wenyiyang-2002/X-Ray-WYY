#pragma once
#include "all_head.h"

class BP_3D :public CtOperation3D {
	string name() const override;
	bool execute(CTData3D& ctdata) override;
};