#pragma once
#include "all_head.h"

class BP_2D :public CtOperation2D {
	string name() const override;
	bool execute(CTData2D& ctdata) override;
};