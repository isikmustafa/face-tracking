#pragma once 


#include "gauss_newton_solver.h"
#include "util.h"
#include "device_util.h"
#include "device_array.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

__global__ void cuComputeJacobianSparseFeatures(
	//shared memory
	const int nFeatures, const int imageWidth, const int imageHeight,
	const int nShapeCoeffs, const int nExpressionCoeffs, const int nAlbedoCoeffs,
	const int nUnknowns, const int nResiduals,
	const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal, const int nAlbedoCoeffsTotal,
	const float sqrt_wreg,

	uchar* image,

	glm::mat4 face_pose, glm::mat3 drx, glm::mat3 dry, glm::mat3 drz, glm::mat4 projection, Eigen::Matrix3f jacobian_local,

	//device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features,

	float* p_shape_basis,
	float* p_expression_basis,
	float* p_albedo_basis,

	float* p_coefficients_shape,
	float* p_coefficients_expression,
	float* p_coefficients_albedo,

	cudaTextureObject_t rgb,
	cudaTextureObject_t barycentrics,
	cudaTextureObject_t vertex_ids,

	//device memory output
	float* p_jacobian, float* p_residuals)
{
	int index = util::getThreadIndex1D();
	int stride = blockDim.x * gridDim.x;

	const int nFaceCoeffs = nShapeCoeffs + nExpressionCoeffs + nAlbedoCoeffs;
	const int nPixels = imageWidth * imageHeight;
	const int n = nFeatures + nPixels + nFaceCoeffs;

	Eigen::Map<Eigen::MatrixXf> jacobian(p_jacobian, nResiduals, nUnknowns);
	Eigen::Map<Eigen::VectorXf> residuals(p_residuals, nResiduals);

	Eigen::Map<Eigen::MatrixXf> shape_basis(p_shape_basis, nVerticesTimes3, nShapeCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> expression_basis(p_expression_basis, nVerticesTimes3, nExpressionCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> albedo_basis(p_albedo_basis, nVerticesTimes3, nAlbedoCoeffsTotal);

	int offset_rows = nFeatures * 2 + nPixels * 3;
	int offset_cols = 7;

	for (int i = index; i < n; i += stride)
	{
		// Regularization terms

		if (i >= nResiduals - nFaceCoeffs)
		{
			const int shape_expression = nShapeCoeffs + nExpressionCoeffs;
			const int current_index = i - nResiduals - nFaceCoeffs;
			// Range of offset for each coefficient
			// 0 -> nShapeCoeffs -> (nShapeCoeffs + nExpressionCoeffs)
			const int shift = current_index >= nShapeCoeffs ?
				current_index >= shape_expression ? shape_expression : nShapeCoeffs : 0;

			offset_rows += shift;
			offset_cols += shift;

			const int relative_index = current_index - shift;

			// Depending which shift is used the proper coefficient is selected
			const float coefficient = shift == 0 ? p_coefficients_shape[relative_index] : shift == nShapeCoeffs ?
				p_coefficients_expression[relative_index] : p_coefficients_albedo[relative_index];

			jacobian(offset_rows + relative_index, offset_cols + relative_index) = sqrt_wreg;
			residuals(offset_rows + relative_index) = coefficient * sqrt_wreg;

			return;
		}

		// Dense terms

		if (i >= nFeatures)
		{
			int idx = i - nFeatures;
			int xp = idx % imageWidth;
			int yp = idx / imageWidth;
			idx *= 3;

			int ygl = imageHeight - 1 - yp; // "height - 1 - index.y" OpenGL uses left-bottom corner as texture origin.
			float4 face_rgb_sampled = tex2D<float4>(rgb, xp, ygl);

			if (face_rgb_sampled.w < 1.0f) return; // pixel is not covered by face

			float4 bary_sampled = tex2D<float4>(barycentrics, xp, ygl);
			int4 verts_s = tex2D<int4>(vertex_ids, xp, ygl);
			Eigen::Map<Eigen::Vector3f> face_rgb(reinterpret_cast<float*>(&face_rgb_sampled));
			Eigen::Vector3f frame_rgb;

			frame_rgb.x() = image[idx] / 255.0f;
			frame_rgb.y() = image[idx + 1] / 255.0f;
			frame_rgb.z() = image[idx + 2] / 255.0f;

			Eigen::Vector3f residual = face_rgb - frame_rgb;

			residuals.block(i * 3, 0, 3, 1) = residual;

			// Albedo

			auto light = bary_sampled.w;

			auto A = light * bary_sampled.x * albedo_basis.block(3 * verts_s.x, 0, 3, nAlbedoCoeffs);
			auto B = light * bary_sampled.y * albedo_basis.block(3 * verts_s.y, 0, 3, nAlbedoCoeffs);
			auto C = light * bary_sampled.z * albedo_basis.block(3 * verts_s.z, 0, 3, nAlbedoCoeffs);

			jacobian.block(i * 3, 7 + nShapeCoeffs + nExpressionCoeffs, 3, nAlbedoCoeffs) = A + B + C;

			// Shape and expression

			jacobian.block(i * 3, 7, 3, nShapeCoeffs) = Eigen::MatrixXf::Zero(3, nShapeCoeffs);
			jacobian.block(i * 3, 7 + nShapeCoeffs, 3, nExpressionCoeffs) = Eigen::MatrixXf::Zero(3, nExpressionCoeffs);

			return;
		}

		// Sparse terms

		Eigen::Matrix<float, 2, 3> jacobian_proj = Eigen::MatrixXf::Zero(2, 3);

		Eigen::Matrix<float, 3, 3> jacobian_world = Eigen::MatrixXf::Zero(3, 3);
		jacobian_world(1, 1) = projection[1][1];
		jacobian_world(2, 2) = -1.0f;

		Eigen::Matrix<float, 3, 1> jacobian_intrinsics = Eigen::MatrixXf::Zero(3, 1);

		Eigen::Matrix<float, 3, 6> jacobian_pose = Eigen::MatrixXf::Zero(3, 6);
		jacobian_pose(0, 3) = 1.0f;
		jacobian_pose(1, 4) = 1.0f;
		jacobian_pose(2, 5) = 1.0f;

		auto vertex_id = prior_local_ids[i];
		auto local_coord = current_face[vertex_id];

		auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
		auto proj_coord = projection * world_coord;
		auto uv = glm::vec2(proj_coord.x, proj_coord.y) / proj_coord.w;

		//Residual
		auto residual = uv - sparse_features[i];

		residuals(i * 2) = residual.x;
		residuals(i * 2 + 1) = residual.y;

		//Jacobian for homogenization (AKA division by w)
		auto one_over_wp = 1.0f / proj_coord.w;
		jacobian_proj(0, 0) = one_over_wp;
		jacobian_proj(0, 2) = -proj_coord.x * one_over_wp * one_over_wp;

		jacobian_proj(1, 1) = one_over_wp;
		jacobian_proj(1, 2) = -proj_coord.y * one_over_wp * one_over_wp;

		//Jacobian for projection
		jacobian_world(0, 0) = projection[0][0];

		//Jacobian for intrinsics
		jacobian_intrinsics(0, 0) = world_coord.x;
		jacobian.block<2, 1>(i * 2, 0) = jacobian_proj * jacobian_intrinsics;

		//Derivative of world coordinates with respect to rotation coefficients
		auto dx = drx * local_coord;
		auto dy = dry * local_coord;
		auto dz = drz * local_coord;

		jacobian_pose(0, 0) = dx[0];
		jacobian_pose(1, 0) = dx[1];
		jacobian_pose(2, 0) = dx[2];
		jacobian_pose(0, 1) = dy[0];
		jacobian_pose(1, 1) = dy[1];
		jacobian_pose(2, 1) = dy[2];
		jacobian_pose(0, 2) = dz[0];
		jacobian_pose(1, 2) = dz[1];
		jacobian_pose(2, 2) = dz[2];

		auto jacobian_proj_world = jacobian_proj * jacobian_world;
		jacobian.block<2, 6>(i * 2, 1) = jacobian_proj_world * jacobian_pose;

		//Derivative of world coordinates with respect to local coordinates.
		//This is basically the rotation matrix.
		auto jacobian_proj_world_local = jacobian_proj_world * jacobian_local;

		//Derivative of local coordinates with respect to shape and expression parameters
		//This is basically the corresponding (to unique vertices we have chosen) rows of basis matrices.
		auto jacobian_shape = jacobian_proj_world_local * shape_basis.block(3 * vertex_id, 0, 3, nShapeCoeffs);
		jacobian.block(i * 2, 7, 2, nShapeCoeffs) = jacobian_shape;

		auto jacobian_expression = jacobian_proj_world_local * expression_basis.block(3 * vertex_id, 0, 3, nExpressionCoeffs);
		jacobian.block(i * 2, 7 + nShapeCoeffs, 2, nExpressionCoeffs) = jacobian_expression;
	}
}

void GaussNewtonSolver::computeJacobianSparseFeatures(
	//shared memory
	const int nFeatures, const int imageWidth, const int imageHeight,
	const int nShapeCoeffs, const int nExpressionCoeffs, const int nAlbedoCoeffs,
	const int nUnknowns, const int nResiduals,
	const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal, const int nAlbedoCoeffsTotal,
	const float regularizationWeight,

	uchar* image,

	const glm::mat4& face_pose, const glm::mat3& drx, const glm::mat3& dry, const glm::mat3& drz, const glm::mat4& projection, const Eigen::Matrix3f& jacobian_local,

	//device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features,

	float* p_shape_basis,
	float* p_expression_basis,
	float* p_albedo_basis,

	float* p_coefficients_shape,
	float* p_coefficients_expression,
	float* p_coefficients_albedo,

	//device memory output
	float* p_jacobian, float* p_residuals
) const
{
	const int nPixels = imageWidth * imageHeight;
	const int n = nFeatures + nPixels + m_params.num_shape_coefficients + m_params.num_expression_coefficients + m_params.num_albedo_coefficients;

	const int threads = 256;
	const int block = (n + threads - 1) / threads;

	cuComputeJacobianSparseFeatures << <block, threads >> > (
		//shared memory
		nFeatures, imageWidth, imageHeight,
		nShapeCoeffs, nExpressionCoeffs, nAlbedoCoeffs,
		nUnknowns, nResiduals,
		nVerticesTimes3, nShapeCoeffsTotal, nExpressionCoeffsTotal, nAlbedoCoeffsTotal,
		glm::sqrt(regularizationWeight),

		image,

		face_pose, drx, dry, drz, projection, jacobian_local,

		//device memory input
		prior_local_ids, current_face, sparse_features,

		p_shape_basis,
		p_expression_basis,
		p_albedo_basis,

		p_coefficients_shape,
		p_coefficients_expression,
		p_coefficients_albedo,

		m_texture_rgb,
		m_texture_barycentrics,
		m_texture_vertex_ids,

		//device memory output
		p_jacobian, p_residuals);

	cudaDeviceSynchronize();
}

__global__ void cuComputeJacobiPreconditioner(const int nUnknowns, const int nResiduals, float* p_jacobian, float* p_preconditioner)
{
	extern __shared__ float temp[];

	int col = blockIdx.x;
	int row = threadIdx.x;
	float v = p_jacobian[col * nResiduals + row];
	temp[row] = v * v;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sum = 0;
		for (int i = 0; i < nResiduals; i++)
		{
			sum += temp[i];
		}
		p_preconditioner[col] = 1.0f / (fmaxf(2.0f*sum, 1e-8f));
	}

}

__global__ void cuElementwiseMultiplication(float* v1, float* v2, float* out)
{
	int i = util::getThreadIndex1D();
	out[i] = v1[i] * v2[i];
}

void GaussNewtonSolver::computeJacobiPreconditioner(const int nUnknowns, const int nResiduals, float* p_jacobian, float* p_preconditioner)
{
	//TODO: split this up into proper blocks, once we have more that 1024 resiudals 
	cuComputeJacobiPreconditioner << <nUnknowns, nResiduals, sizeof(float)*nResiduals >> > (nUnknowns, nResiduals, p_jacobian, p_preconditioner);
	cudaDeviceSynchronize();
}

void GaussNewtonSolver::elementwiseMultiplication(const int nElements, float* v1, float* v2, float* out)
{
	cuElementwiseMultiplication << <1, nElements >> > (v1, v2, out);
	cudaDeviceSynchronize();
}