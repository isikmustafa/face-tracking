#pragma once 

#include "gauss_newton_solver.h"
#include "util.h"
#include "device_util.h"
#include "device_array.h"


__global__ void cuComputeJacobianSparseFeatures( 
	//shared memory
	int nFeatures, 
	int nShapeCoeffs, int nExpressionCoeffs, 
	int nUnknowns, int nResiduals, 
	int nVertsX3, int nShapeCoeffsTotal, int nExpressionCoeffsTotal, 

	glm::mat4 face_pose, glm::mat3 drx, glm::mat3 dry, glm::mat3 drz, glm::mat4 projection, 
	Eigen::Matrix<float, 2, 3> jacobian_proj, Eigen::Matrix<float, 3, 3> jacobian_world, 
	Eigen::Matrix<float, 3, 1> jacobian_intrinsics, Eigen::Matrix<float, 3, 6> jacobian_pose, Eigen::Matrix3f jacobian_local, 

	//device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features, 
	float* pShapeBasis, float* pExpressionBasis, 

	//device memory output
	float* pJacobian, float* residuals
	)
{
	int i = util::getThreadIndex1D(); 

	Eigen::Map<Eigen::MatrixXf> shape_basis(pShapeBasis, nVertsX3, nShapeCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> expression_basis(pExpressionBasis, nVertsX3, nExpressionCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> jacobian(pJacobian, nResiduals, nUnknowns);


	auto vertexId = prior_local_ids[i];
	//auto local_coord = prior_local_positions[i];
	auto local_coord = current_face[vertexId];

	auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
	auto proj_coord = projection * world_coord;
	auto uv = glm::vec2(proj_coord.x, proj_coord.y) / proj_coord.w;

	//Residual
	auto residual = sparse_features[i] - uv;

	residuals[i * 2] = residual.x;
	residuals[i * 2 + 1] = residual.y;

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

	auto jacobian_shape = jacobian_proj_world_local * shape_basis.block(3 * vertexId, 0, 3, nShapeCoeffs);

	jacobian.block(i * 2, 7, 2, nShapeCoeffs) = jacobian_shape;

	auto jacobian_expression = jacobian_proj_world_local * expression_basis.block(3 * vertexId, 0, 3, nExpressionCoeffs);
	jacobian.block(i * 2, 7 + nShapeCoeffs, 2, nExpressionCoeffs) = jacobian_expression;
}


__global__ void cuComputeRegularizer(
	int nUnknowns, int nResiduals,
	int offsetRows, int offsetCols,
	float wReg, 

	//device memory input
	float* pSTD, float* pCoeffs, 

	//device memory output
	float* pJacobian, float* residuals
)
{
	int i = util::getThreadIndex1D();

	Eigen::Map<Eigen::MatrixXf> jacobian(pJacobian, nResiduals, nUnknowns);
	float divSigma = 1.0f / pSTD[i];
	//3rd division, because we are getting the denormalized coefficients computed in computeFace()
	jacobian(offsetRows + i, offsetCols + i) = divSigma * divSigma * divSigma*pCoeffs[i] * wReg * 2; 
	residuals[offsetRows + i] = 0;
}

void GaussNewtonSolver::computeJacobianSparseFeatures(
	//shared memory
	int nFeatures,
	int nShapeCoeffs, int nExpressionCoeffs,
	int nUnknowns, int nResiduals,
	int nVertsX3, int nShapeCoeffsTotal, int nExpressionCoeffsTotal,

	glm::mat4 face_pose, glm::mat3 drx, glm::mat3 dry, glm::mat3 drz, glm::mat4 projection,
	Eigen::Matrix<float, 2, 3> jacobian_proj, Eigen::Matrix<float, 3, 3> jacobian_world,
	Eigen::Matrix<float, 3, 1> jacobian_intrinsics, Eigen::Matrix<float, 3, 6> jacobian_pose, Eigen::Matrix3f jacobian_local,

	//device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features,
	float* pShapeBasis, float* pExpressionBasis,

	//device memory output
	float* pJacobian, float* residuals
)
{
	//TODO: block stuff? 
	cuComputeJacobianSparseFeatures << <1, nFeatures >> > (
		//shared memory
		nFeatures,
		nShapeCoeffs, nExpressionCoeffs,
		nUnknowns, nResiduals,
		nVertsX3, nShapeCoeffsTotal, nExpressionCoeffsTotal,

		face_pose, drx, dry, drz, projection,
		jacobian_proj, jacobian_world,
		jacobian_intrinsics, jacobian_pose, jacobian_local,

		//device memory input
		prior_local_ids, current_face, sparse_features,
		pShapeBasis, pExpressionBasis,

		//device memory output
		pJacobian, residuals
		);
}


void GaussNewtonSolver::computeRegularizer(

	Face& face,
	int offsetRows,
	int nUnknowns, int nResiduals,

	float wReg,

	//device memory output
	float* pJacobian, float* residuals
)
{
	//Shape
	int offsetCols = 7; 
	cuComputeRegularizer<<<1,m_params.numShapeCoefficients >>>(
		nUnknowns, nResiduals,
		offsetRows, offsetCols,
		wReg,

		//device memory input
		face.m_shape_std_dev_gpu.getPtr(), face.m_shape_coefficients_gpu.getPtr(),

		//device memory output
		pJacobian, residuals
	); 

	//Expression
	offsetRows += m_params.numShapeCoefficients; 
	offsetCols += m_params.numShapeCoefficients;
	cuComputeRegularizer << <1, m_params.numExpressionCoefficients >> > (
		nUnknowns, nResiduals,
		offsetRows, offsetCols,
		wReg,

		//device memory input
		face.m_expression_std_dev_gpu.getPtr(), face.m_expression_coefficients_gpu.getPtr(),

		//device memory output
		pJacobian, residuals
		);

	//Albedo
	//offsetRows += m_params.numExpressionCoefficients; 
	//offsetCols += m_params.numExpressionCoefficients;
	//cuComputeRegularizer <<<1,	m_params.numAlbedoCoefficients >>> (
	//	nUnknowns, nResiduals,
	//	offsetRows, offsetCols,
	//	wReg,

	//	//device memory input
	//	face.m_albedo_std_dev_gpu.getPtr(), face.m_albedo_coefficients_gpu.getPtr(),

	//	//device memory output
	//	pJacobian, residuals
	//	);
}