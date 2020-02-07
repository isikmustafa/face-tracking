#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>

namespace jacobian_util
{
	//Computation of derivative of triangle normal with respect to triangle vertices.
	__host__ __device__ inline void computeNormalJacobian(Eigen::Matrix<float, 3, 3>& dndv0, Eigen::Matrix<float, 3, 3>& dndv1, Eigen::Matrix<float, 3, 3>& dndv2,
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

	//Computation of derivative of normal with respect to unnormalized vector that goes into normalize function.
	__host__ __device__ inline void computeNormalizationJacobian(Eigen::Matrix<float, 3, 3>& d, const glm::vec3& unnormalized_vector)
	{
		auto x2 = unnormalized_vector.x * unnormalized_vector.x;
		auto y2 = unnormalized_vector.y * unnormalized_vector.y;
		auto z2 = unnormalized_vector.z * unnormalized_vector.z;
		auto xy = unnormalized_vector.x * unnormalized_vector.y;
		auto xz = unnormalized_vector.x * unnormalized_vector.z;
		auto yz = unnormalized_vector.y * unnormalized_vector.z;

		d <<
			y2 + z2, -xy, -xz,
			-xy, x2 + z2, -yz,
			-xz, -yz, x2 + y2;

		d *= glm::pow(x2 + y2 + z2, -1.5f);
	}

	//Computation of derivative of light with respect to final normal that goes into SH light calculations.
	__host__ __device__ inline void computeDLightDNormal(Eigen::Matrix<float, 1, 3>& d, const glm::vec3& normal, const float* coefficients_sh)
	{
		d(0, 0) = coefficients_sh[3] + coefficients_sh[4] * normal.y + coefficients_sh[7] * normal.z + 2.0f * coefficients_sh[8] * normal.x;
		d(0, 1) = coefficients_sh[1] + coefficients_sh[4] * normal.x + coefficients_sh[5] * normal.z - 2.0f * coefficients_sh[8] * normal.y;
		d(0, 2) = coefficients_sh[2] + coefficients_sh[5] * normal.y + 6.0f * coefficients_sh[6] * normal.z + coefficients_sh[7] * normal.x;
	}
}