#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>

namespace util
{
	__host__ __device__ inline void computeNormalGradient(Eigen::Matrix<float, 3, 3>& dndv0, Eigen::Matrix<float, 3, 3>& dndv1, Eigen::Matrix<float, 3, 3>& dndv2,
		const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
	{
		//In face.cu, normal is compute as below
		//glm::vec3 face_normal = glm::cross((v1 - v0), (v2 - v0));

		auto vec1 = v1 - v0;
		auto vec2 = v2 - v0;

		dndv1 <<
			0.0f, vec2.z, -vec2.y,
			-vec2.z, 0.0f, vec2.x,
			vec2.y, -vec2.x, 0.0f;

		dndv2 <<
			0.0f, -vec1.z, vec1.y,
			vec1.z, 0.0f, -vec1.x,
			-vec1.y, vec1.x, 0.0f;

		dndv0 = -dndv1 - dndv2;
	}

	__host__ __device__ inline void computeNormalizationGradient(Eigen::Matrix<float, 3, 3>& d, const glm::vec3& unnormalized_vector)
	{
		auto x2 = unnormalized_vector.x * unnormalized_vector.x;
		auto y2 = unnormalized_vector.y * unnormalized_vector.y;
		auto z2 = unnormalized_vector.z * unnormalized_vector.z;
		auto xy = unnormalized_vector.x * unnormalized_vector.y;
		auto xz = unnormalized_vector.x * unnormalized_vector.z;
		auto yz = unnormalized_vector.y * unnormalized_vector.z;

		d <<
			(y2 + z2), -xy, -xz,
			-xy, (x2 + z2), -yz,
			-xz, -yz, (x2 + y2);

		d *= glm::pow(x2 + y2 + z2, -1.5f);
	}
}