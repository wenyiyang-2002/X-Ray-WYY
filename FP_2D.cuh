#pragma once
#include "all_head.h"

class FP_2D : public CtOperation2D {
	bool execute(CTData2D& ctdata) override;
	string name() const override;
};
