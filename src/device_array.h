#pragma once

#include "util.h"
#include "prior_sparse_features.h"

#include <assert.h>
#include <vector>
#include <cuda_runtime.h>

namespace util
{
	//This class is a wrapper for device memory.
	template<typename T>
	class DeviceArray
	{
	public:
		explicit DeviceArray(int size = 0)
			: m_size(size)
		{
			if (m_size > 0)
			{
				CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_size * sizeof(T)));
			}
		}

		DeviceArray(const std::vector<T>& vector)
			: m_size(vector.size())
		{
			if (m_size > 0)
			{
				CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_size * sizeof(T)));
				CHECK_CUDA_ERROR(cudaMemcpy(m_ptr, vector.data(), m_size * sizeof(T), cudaMemcpyHostToDevice));
			}
		}

		DeviceArray(const DeviceArray& da)
			: m_size(da.m_size)
		{
			if (m_size > 0)
			{
				CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_size * sizeof(T)));
				CHECK_CUDA_ERROR(cudaMemcpy(m_ptr, da.m_ptr, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
			}
		}

		DeviceArray(DeviceArray&& da)
		{
			std::swap(m_size, da.m_size);
			std::swap(m_ptr, da.m_ptr);
		}

		DeviceArray& operator = (DeviceArray da)
		{
			std::swap(m_size, da.m_size);
			std::swap(m_ptr, da.m_ptr);

			return *this;
		}

		~DeviceArray()
		{
			CHECK_CUDA_ERROR(cudaFree(m_ptr));
			m_ptr = nullptr;
			m_size = 0;
		}

		void memset(int value)
		{
			CHECK_CUDA_ERROR(cudaMemset(m_ptr, value, m_size * sizeof(T)));
		}

		int getSize() const
		{
			return m_size;
		}

		T* getPtr()
		{
			return m_ptr;
		}

		const T* getPtr() const
		{
			return m_ptr;
		}

	private:
		int m_size{ 0 };
		T* m_ptr{ nullptr };
	};

	//Some copy functions on DeviceArray and std::vector.
	//User has to be sure that "dst" and "src" is as large as "size".
	//For the sake of completeness, host to host copy is also implemented.
	template<typename T>
	void copy(DeviceArray<T>& dst, const DeviceArray<T>& src, int size, int offset_dst = 0, int offset_src = 0)
	{
		assert(dst.getSize() >= size && src.getSize() >= size);
		CHECK_CUDA_ERROR(cudaMemcpy(dst.getPtr() + offset_dst, src.getPtr() + offset_src, size * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void copy(DeviceArray<T>& dst, const std::vector<T>& src, int size, int offset_dst = 0, int offset_src = 0)
	{
		assert(dst.getSize() >= size && src.size() >= size);
		CHECK_CUDA_ERROR(cudaMemcpy(dst.getPtr() + offset_dst, src.data() + offset_src, size * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void copy(std::vector<T>& dst, const std::vector<T>& src, int size, int offset_dst = 0, int offset_src = 0)
	{
		assert(dst.size() >= size && src.size() >= size);
		CHECK_CUDA_ERROR(cudaMemcpy(dst.data() + offset_dst, src.data() + offset_src, size * sizeof(T), cudaMemcpyHostToHost));
	}

	template<typename T>
	void copy(std::vector<T>& dst, const DeviceArray<T>& src, int size, int offset_dst = 0, int offset_src = 0)
	{
		assert(dst.size() >= size && src.getSize() >= size);
		CHECK_CUDA_ERROR(cudaMemcpy(dst.data() + offset_dst, src.getPtr() + offset_src, size * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void copy(DeviceArray<T>& dst, const T* src, int size, int offset_dst = 0, int offset_src = 0)
	{
		cudaPointerAttributes pointer_attributes;
		memset(&pointer_attributes, 0, sizeof(pointer_attributes));
		cudaPointerGetAttributes(&pointer_attributes, src);

		if (pointer_attributes.type == cudaMemoryTypeDevice)
		{
			CHECK_CUDA_ERROR(cudaMemcpy(dst.getPtr() + offset_dst, src + offset_src, size * sizeof(T), cudaMemcpyDeviceToDevice));
		}
		else
		{
			cudaGetLastError();
			CHECK_CUDA_ERROR(cudaMemcpy(dst.getPtr() + offset_dst, src + offset_src, size * sizeof(T), cudaMemcpyHostToDevice));
		}
	}

	template<typename T>
	void copy(T* dst, const DeviceArray<T>& src, int size, int offset_dst = 0, int offset_src = 0)
	{
		cudaPointerAttributes pointer_attributes;
		memset(&pointer_attributes, 0, sizeof(pointer_attributes));
		cudaPointerGetAttributes(&pointer_attributes, dst);

		if (pointer_attributes.type == cudaMemoryTypeDevice)
		{
			CHECK_CUDA_ERROR(cudaMemcpy(dst + offset_dst, src.getPtr() + offset_src, size * sizeof(T), cudaMemcpyDeviceToDevice));
		}
		else
		{
			cudaGetLastError();
			CHECK_CUDA_ERROR(cudaMemcpy(dst + offset_dst, src.getPtr() + offset_src, size * sizeof(T), cudaMemcpyDeviceToHost));
		}
	}
}