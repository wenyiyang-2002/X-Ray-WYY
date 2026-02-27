#include "all_head.h"
#include "DataGenerate2D.h"
#include "DataSave2D.h"
#include "GetProjectionMatrix2D.h"
#include "FP_2D.cuh"
#include "WeightFiltering2D.cuh"
#include "BP_2D.cuh"

#include "DataGenerate3D.h"
#include "DataSave3D.h"
#include "GetProjectionMatrix3D.h"
#include "FP_3D.cuh"
#include "WeightFiltering3D.cuh"
#include "BP_3D.cuh"

void test2d() {
	CTData2D* ctdata2d = new CTData2D;
	//执行操作添加
	vector<unique_ptr<CtOperation2D>> operation_list;
    operation_list.emplace_back(make_unique<DataGenerate2D>("./Output2D/image.raw"));
    operation_list.emplace_back(make_unique<GetProjectionMatrix2D>());
    operation_list.emplace_back(make_unique<FP_2D>());
	operation_list.emplace_back(make_unique<WeightFiltering2D>("WINDOW_RECTANGULAR"));
	operation_list.emplace_back(make_unique<BP_2D>());
	operation_list.emplace_back(make_unique<DataSave2D>("./Output2D/"));

	//执行所有已添加的操作
	for (auto& operation : operation_list)
	{
		cout << operation->name() << endl;
		operation->execute(*ctdata2d);
	}
	delete ctdata2d;
}

void TestFDK() { 
	CTData3D* ctdata3d = new CTData3D;
	//执行操作添加
    vector<unique_ptr<CtOperation3D>> operation_list;
	operation_list.emplace_back(make_unique<DataGenerate3D>("./Output3D/image.raw"));
    operation_list.emplace_back(make_unique<GetProjectionMatrix3D>());
    operation_list.emplace_back(make_unique<FP_3D>());
	operation_list.emplace_back(make_unique<WeightFiltering3D>("WINDOW_RECTANGULAR"));
	operation_list.emplace_back(make_unique<BP_3D>());
	operation_list.emplace_back(make_unique<DataSave3D>("./Output3D/"));

	for (auto& operation : operation_list)
	{
		cout << operation->name() << endl;
		operation->execute(*ctdata3d);
	}
	delete ctdata3d;
}

int main() {
    //test2d();
    TestFDK();
	system("pause");
	return 0;
}