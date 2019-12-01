#pragma once

#include "face.h"
#include <Eigen/Dense>

//Default
struct SolverParameters
{
	float regularisationWeightExponent = -3.0f;

	int numGNiterations = 4;
	int numPCGiterations = 100;

	int numShapeCoefficients = 80;
	int numAlbedoCoefficients = 0;
	int numExpressionCoefficients = 76;
};

//Debug
//struct SolverParameters
//{
//	float regularisationWeightExponent = -3.0f; 
//
//	int numGNiterations = 1; 
//	int numPCGiterations = 200; 
//
//	int numShapeCoefficients = 5; 
//	int numAlbedoCoefficients = 0;
//	int numExpressionCoefficients = 5; 
//};

class GaussNewtonSolver
{
public:

	GaussNewtonSolver(); 
	~GaussNewtonSolver(); 

	void solve(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection);
	void solve_CPU(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection);


	void solveUpdatePCG(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian, util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);
	void solveUpdateLU(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian, util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);

	SolverParameters* getParameters() { return &m_params;  }

private: 
	cublasHandle_t m_cublas; 

	SolverParameters m_params; 

	void computeJacobianSparseFeatures(
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
	); 

	void computeRegularizer(

		Face& face,
		int offsetRows,
		int nUnknowns, int nResiduals,

		float wReg,

		//device memory output
		float* pJacobian, float* residuals
	);

};