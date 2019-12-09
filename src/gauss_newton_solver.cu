#pragma once 

#include "gauss_newton_solver.h"
#include "util.h"
#include "device_util.h"
#include "device_array.h"

__global__ void cuComputeJacobianSparseFeatures( 
	//shared memory
	const int nFeatures,
	const int nShapeCoeffs, const int nExpressionCoeffs,
	const int nUnknowns, const int nResiduals,
	const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal,

	glm::mat4 face_pose, glm::mat3 drx, glm::mat3 dry, glm::mat3 drz, glm::mat4 projection, 
	Eigen::Matrix<float, 2, 3> jacobian_proj, Eigen::Matrix<float, 3, 3> jacobian_world, 
	Eigen::Matrix<float, 3, 1> jacobian_intrinsics, Eigen::Matrix<float, 3, 6> jacobian_pose, Eigen::Matrix3f jacobian_local, 

	//device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features, 
	float* p_shape_basis,  float* p_expression_basis,

	//device memory output
	float* p_jacobian, float* p_residuals
	)
{
	int i = util::getThreadIndex1D(); 

	Eigen::Map<Eigen::MatrixXf> shape_basis(p_shape_basis, nVerticesTimes3, nShapeCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> expression_basis(p_expression_basis, nVerticesTimes3, nExpressionCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> jacobian(p_jacobian, nResiduals, nUnknowns);

	auto vertex_id = prior_local_ids[i];
	//auto local_coord = prior_local_positions[i];
	auto local_coord = current_face[vertex_id];

	auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
	auto proj_coord = projection * world_coord;
	auto uv = glm::vec2(proj_coord.x, proj_coord.y) / proj_coord.w;

	//Residual
	auto residual = sparse_features[i] - uv;

	p_residuals[i * 2] = residual.x;
	p_residuals[i * 2 + 1] = residual.y;

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

__global__ void cuComputeRegularizer(
	const int nUnknowns, const int nResiduals,
	int offset_rows, int offset_cols,
	const float regularizationWeight,

	//device memory input
	float* p_coefficients, 

	//device memory output
	float* p_jacobian, float* p_residuals
)
{
	int i = util::getThreadIndex1D();

	Eigen::Map<Eigen::MatrixXf> jacobian(p_jacobian, nResiduals, nUnknowns);
	//3rd division, because we are getting the denormalized coefficients computed in computeFace()
	jacobian(offset_rows + i, offset_cols + i) = p_coefficients[i] * regularizationWeight * 2;
	p_residuals[offset_rows + i] = p_coefficients[i] * glm::sqrt(regularizationWeight);
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

void GaussNewtonSolver::computeJacobianSparseFeatures(
	//shared memory
	const int nFeatures,
	const int nShapeCoeffs, const int nExpressionCoeffs,
	const int nUnknowns, const int nResiduals,
	const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal,

	const glm::mat4& face_pose, const glm::mat3& drx, const glm::mat3& dry, const glm::mat3& drz, const glm::mat4& projection,
	const Eigen::Matrix<float, 2, 3>& jacobian_proj, const Eigen::Matrix<float, 3, 3>& jacobian_world,
	const Eigen::Matrix<float, 3, 1>& jacobian_intrinsics, const Eigen::Matrix<float, 3, 6>& jacobian_pose, const Eigen::Matrix3f& jacobian_local,

	//device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features,
	float* p_shape_basis, float* p_expression_basis,

	//device memory output
	float* p_jacobian, float* p_residuals
)
{
	//TODO: block stuff? 
	cuComputeJacobianSparseFeatures << <1, nFeatures >> > (
		//shared memory
		nFeatures,
		nShapeCoeffs, nExpressionCoeffs,
		nUnknowns, nResiduals,
		nVerticesTimes3, nShapeCoeffsTotal, nExpressionCoeffsTotal,

		face_pose, drx, dry, drz, projection,
		jacobian_proj, jacobian_world,
		jacobian_intrinsics, jacobian_pose, jacobian_local,

		//device memory input
		prior_local_ids, current_face, sparse_features,
		p_shape_basis, p_expression_basis,

		//device memory output
		p_jacobian, p_residuals
		);
	cudaDeviceSynchronize();
}


void GaussNewtonSolver::computeRegularizer(
	Face& face,
	int offset_rows,
	const int nUnknowns, const int nResiduals,
	const float regularizationWeight,
	//device memory output
	float* p_jacobian, float* p_residuals
)
{
	//Shape
	int offset_cols = 7; 
	cuComputeRegularizer<<<1,m_params.num_shape_coefficients >>>(
		nUnknowns, nResiduals,
		offset_rows, offset_cols,
		regularizationWeight,

		//device memory input
		face.m_shape_coefficients_gpu.getPtr(),

		//device memory output
		p_jacobian, p_residuals
	); 

	//Expression
	offset_rows += m_params.num_shape_coefficients; 
	offset_cols += m_params.num_shape_coefficients;
	cuComputeRegularizer << <1, m_params.num_expression_coefficients >> > (
		nUnknowns, nResiduals,
		offset_rows, offset_cols,
		regularizationWeight,

		//device memory input
		face.m_expression_coefficients_gpu.getPtr(),

		//device memory output
		p_jacobian, p_residuals
		);

	//Albedo
	//offset_rows += m_params.num_expression_coefficients; 
	//offset_cols += m_params.num_expression_coefficients;
	//cuComputeRegularizer <<<1,	m_params.num_albedo_coefficients >>> (
	//	nUnknowns, nResiduals,
	//	offset_rows, offset_cols,
	//	regularization_weight,

	//	//device memory input
	//	face.m_albedo_std_dev_gpu.getPtr(), face.m_albedo_coefficients_gpu.getPtr(),

	//	//device memory output
	//	p_jacobian, p_residuals
	//	);
	cudaDeviceSynchronize();
}

void GaussNewtonSolver::computeJacobiPreconditioner(const int nUnknowns, const int nResiduals, float* p_jacobian, float* p_preconditioner)
{
	//TODO: split this up into proper blocks, once we have more that 1024 resiudals 
	cuComputeJacobiPreconditioner<<<nUnknowns, nResiduals, sizeof(float)*nResiduals>>>(nUnknowns, nResiduals, p_jacobian, p_preconditioner);
	cudaDeviceSynchronize();
}

void GaussNewtonSolver::elementwiseMultiplication(const int nElements, float* v1, float* v2, float* out)
{
	cuElementwiseMultiplication << <1, nElements>> > (v1, v2, out);
	cudaDeviceSynchronize();
}
