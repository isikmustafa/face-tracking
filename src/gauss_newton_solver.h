#pragma once

#include "face.h"
#include <Eigen/Dense>

//Default
struct SolverParameters
{
	float regularisation_weight_exponent = -4.6f;

	int num_gn_iterations = 5;
	int num_pcg_iterations = 15;

	int num_shape_coefficients = 30;
	int num_albedo_coefficients = 0;
	int num_expression_coefficients = 76;

	const float kNearZero = 1.0e-8;		// interpretation of "zero"
	const float kTolerance = 1.0e-3;	//convergence if rtr < TOLERANCE
};

////Debug
//struct SolverParameters
//{
//	float regularisation_weight_exponent = -3.0f; 
//
//	int num_gn_iterations = 1; 
//	int num_pcg_iterations = 10000; 
//
//	int num_shape_coefficients = 5; 
//	int num_albedo_coefficients = 0;
//	int num_expression_coefficients = 1; 
//	const float kNearZero = 1.0e-8;		// interpretation of "zero"
//	const float kTolerance = 1.0e-10;	//convergence if rtr < TOLERANCE
//};

class GaussNewtonSolver
{
public:
	GaussNewtonSolver();
	~GaussNewtonSolver();

	void solve(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection);
	void solve_CPU(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection);

	SolverParameters& getSolverParameters() { return m_params; }
	const SolverParameters& getSolverParameters() const { return m_params; }

private:
	cublasHandle_t m_cublas;

	SolverParameters m_params;

	void computeJacobianSparseFeatures(
		//shared memory
		const int nFeatures,
		const int nShapeCoeffs, const int nExpressionCoeffs,
		const int nUnknowns, const int nResiduals,
		const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal,
		const float regularizationWeight,

		const glm::mat4& face_pose, const glm::mat3& drx, const glm::mat3& dry, const glm::mat3& drz, const glm::mat4& projection,
		const Eigen::Matrix<float, 2, 3>& jacobian_proj, const Eigen::Matrix<float, 3, 3>& jacobian_world,
		const Eigen::Matrix<float, 3, 1>& jacobian_intrinsics, const Eigen::Matrix<float, 3, 6>& jacobian_pose, const Eigen::Matrix3f& jacobian_local,

		//device memory input
		int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features,
		float* p_shape_basis, float* p_expression_basis, float* p_coefficients_shape, float* p_coefficients_expression,

		//device memory output
		float* p_jacobian, float* p_residuals) const;

	void elementwiseMultiplication(const int nElements, float* v1, float* v2, float* out);

	void computeJacobiPreconditioner(const int nUnknowns, const int nResiduals, float* p_jacobian, float* p_preconditioner);

	void solveUpdateCG(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian,
		util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);

	void solveUpdatePCG(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian,
		util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);

	void solveUpdateLU(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian,
		util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);

	void updateParameters(const std::vector<float>& result, glm::mat4& projection,
		glm::vec3& rotation_coefficients, glm::vec3& translation_coefficients, Face& face, const int nShapeCoeffs, const int nExpressionCoeffs);
};