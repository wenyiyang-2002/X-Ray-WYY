#pragma once
#include "all_head.h"

class WeightFiltering2D :public CtOperation2D
{ 
public:
	WeightFiltering2D(string window_name):_window_name(window_name){}//需要传入滤波核名称来构造
	string name() const override;
	bool execute(CTData2D& ctdata) override;
private:
	//返回滤波核与图像宽的比值
	void Gen_Filter_Kernel(CTData2D& ctdata, string filter_name);
	void Gen_Weight(CTData2D& ctdata);
	string _window_name;
};