#include "all_head.h"

class FP_3D :public CtOperation3D
{
public:
	string name() const override;
	bool execute(CTData3D& ctdata) override;
};