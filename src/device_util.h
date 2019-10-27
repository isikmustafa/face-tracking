#pragma once

#include <iostream>
#include <functional>
#include <cuda_runtime.h>

namespace util
{
#ifdef __NVCC__
	__device__ inline unsigned int getThreadIndex1D()
	{
		return { threadIdx.x + blockIdx.x * blockDim.x };
	}

	__device__ inline uint2 getThreadIndex2D()
	{
		return { threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y };
	}

	__device__ inline uint3 getThreadIndex3D()
	{
		return { threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y, threadIdx.z + blockIdx.z * blockDim.z };
	}

	__device__ inline unsigned int rgbToUint(const float3& rgb)
	{
		unsigned int pixel = (255) << 8;
		pixel = (pixel | static_cast<unsigned char>(rgb.x)) << 8;
		pixel = (pixel | static_cast<unsigned char>(rgb.y)) << 8;
		pixel = (pixel | static_cast<unsigned char>(rgb.z));

		return pixel;
	}
#endif
}