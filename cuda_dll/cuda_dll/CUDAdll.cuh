#include <stdio.h>       //引入C函数库-实际上本程序就是应该以C的方式编译，尽管其后缀为cpp类型

#include <stdlib.h>

#include <cuda_runtime.h> //引入CUDA运行时库头文件

#include "cuda_runtime.h"

#ifdef __cplusplus //指明函数的编译方式，以得到没有任何修饰的函数名

extern "C"

{

#endif



#ifdef CUDADLLTEST_EXPORTS

#define CUDADLLTEST_API __declspec(dllexport) //导出符号宏定义

#else

#define CUDADLLTEST_API __declspec(dllimport)

#endif



	extern CUDADLLTEST_API int count;       //要导出的全局变量
	   float*h_Z;       //要导出的全局变量
	   float*d_Z;
	

	CUDADLLTEST_API bool InitCUDA(void);    //要导出的CUDA初始化函数

	CUDADLLTEST_API void showHelloCuda(void); //要导出的测试函数


	CUDADLLTEST_API	void Allocate_Memory(int n);

	CUDADLLTEST_API void Free_Memory();

	CUDADLLTEST_API void CopyMemToDevice(float *data, int n);

	CUDADLLTEST_API void CopyMemToHost(float *data, int n);
	
	CUDADLLTEST_API void Call_cuda_CalZ(float toolx, float tooly, float toolz, float toolr, float dx, float dy, int max_ix, int max_iy, int n);

	CUDADLLTEST_API __global__ void Cal_Z(float*Z_data, float toolx, float tooly, float toolz, float toolr, float dx, float dy, int max_ix, int max_iy, int init_index,int n);

#ifdef __cplusplus

}

#endif

