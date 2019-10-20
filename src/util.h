#pragma once

#include <iostream>
#include <functional>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR( err ) (util::checkCudaError( err, __FILE__, __LINE__))

namespace util
{
	__host__ inline void checkCudaError(cudaError_t err, const char* file, int line)
	{
		if (err != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
			system("PAUSE");
		}
	}

	__host__ inline float runKernelGetExecutionTime(std::function<void(void)> kernel)
	{
		cudaEvent_t start, end;
		CHECK_CUDA_ERROR(cudaEventCreate(&start));
		CHECK_CUDA_ERROR(cudaEventCreate(&end));

		CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
		kernel();
		CHECK_CUDA_ERROR(cudaEventRecord(end, 0));
		CHECK_CUDA_ERROR(cudaEventSynchronize(end));

		float elapsed_time = 0.0f;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));

		CHECK_CUDA_ERROR(cudaEventDestroy(start));
		CHECK_CUDA_ERROR(cudaEventDestroy(end));

		return elapsed_time;
	}
}