
#include "stdafx.h" //引入预编译头文件

#include "CUDAdll.cuh" //引入导出函数声明头文件

#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

//初始化CUDA

extern int count = 0;

bool InitCUDA(void)//CUDA初始化函数

{

	printf("Start to detecte devices.........\n");//显示检测到的设备数

	cudaGetDeviceCount(&count);//检测计算能力大于等于1.0的设备数

	if (count == 0){

		fprintf(stderr, "There is no device.\n");

		return false;

	}

	printf("%d device/s detected.\n", count);//显示检测到的设备数

	int i;

	for (i = 0; i < count; i++){//依次验证检测到的设备是否支持CUDA

		cudaDeviceProp prop;

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {//获得设备属性并验证是否正确

			if (prop.major >= 1)//验证主计算能力，即计算能力的第一位数是否大于1

			{

				printf("Device %d: %s supports CUDA %d.%d.\n", i + 1, prop.name, prop.major, prop.minor);//显示检测到的设备支持的CUDA版本

				break;



			}

		}

	}

	if (i == count) {//没有支持CUDA1.x的设备

		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");

		return false;

	}

	cudaSetDevice(i);//设置设备为主叫线程的当前设备

	return true;

}

void showHelloCuda(void)//测试CUDA初始化函数

{

	if (!InitCUDA()) //初始化失败

	{

		printf("Sorry,CUDA has not been initialized.\n");

		return;

	}

	printf("Hello GPU! CUDA has been initialized.\n");

}

void Allocate_Memory(int n)
{
	size_t size = n*sizeof(float);

	h_Z = (float*)malloc(size);
	cudaError_t error = cudaMalloc((void**)&d_Z, size);
	printf("Allocate mem : %s\n", cudaGetErrorString(error));
}

void Free_Memory()
{
	if (h_Z)
		free(h_Z);
	cudaError_t error = cudaFree(d_Z);
	printf("Free mem : %s\n", cudaGetErrorString(error));
}

void CopyMemToDevice(float *data,int n)
{
	for (int i = 0; i < n; i++)
	{
		h_Z[i] = data[i];
	}
	size_t size = n*sizeof(float);
	cudaError_t error = cudaMemcpy(d_Z, h_Z, size, cudaMemcpyHostToDevice);
	printf("Memcpy Host to Device : %s\n", cudaGetErrorString(error));
}

void CopyMemToHost(float *data,int n)
{
	cudaError_t error = cudaMemcpy(h_Z, d_Z, n*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Memcpy Device to Host : %s\n", cudaGetErrorString(error));
	for (int i = 0; i < n; i++)
	{
		data[i] = h_Z[i];
	}
}

__global__ void Cal_Z(float*Z_data, float toolx, float tooly,float toolz, float toolr, float dx, float dy, int max_ix, int max_iy, int n)
{
	int I = blockDim.x*blockIdx.x + threadIdx.x;

	int i, j;
	if (I < n)
	{


		int ix = toolx / dx - toolr / dx;
		int iy = tooly / dy - toolr / dy;
		if (ix < 0)
			ix = 0;
		if (iy < 0)
			iy = 0;

		int xcount = ix + 2 * toolr / dx;
		int ycount = iy + 2 * toolr / dy;

		if (xcount > max_ix)
			xcount = max_ix;
		if (ycount > max_iy)
			ycount = max_iy;
		i = I % max_ix;
		j = I / max_ix;
		/*for (int i = ix; i < xcount; i++)
		{
			for (int j = iy; j < ycount; j++)
			{*/

				if (pow((i*dx - toolx), 2) + pow((j*dy - tooly), 2) <= pow(toolr, 2)&& Z_data[i*max_iy + j] >= toolz)
				{
					float z_ball = -pow(pow(toolr, 2) - pow((i*dx - toolx), 2) - pow((j*dy - tooly), 2), float(0.5)) + toolr + toolz;
					if (Z_data[i*max_iy + j] > z_ball&&z_ball >= 0){
						Z_data[i*max_iy + j] = z_ball;
					}
					if (z_ball < 0){
						Z_data[i*max_iy + j] = 0;
					}
				}
			/*}
		}*/
	}
}