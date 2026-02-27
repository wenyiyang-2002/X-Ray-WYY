#pragma once
#include "all_head.h"
#include <fstream>
class GetProjectionMatrix2D : public CtOperation2D
{
    public:
        bool execute(CTData2D& ctdata) override;
        string name() const override;
};

