#pragma once

#include "face.h"
#include <Eigen/Dense>

#define PCG_ITERS 300

class GaussNewtonSolver
{
public:

	GaussNewtonSolver(); 
	~GaussNewtonSolver(); 

	void solve(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection);


	void solveUpdatePCG(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian, util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);
	void solveUpdateLU(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian, util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS = 1, const float alphaRHS = 1);


private: 
	cublasHandle_t m_cublas; 

	//void computeJacobianCoeffs(int nFeatures, util::DeviceArray<int>& ids_gpu, util::DeviceArray <glm::vec2>& keypts_gpu,
	//	util::DeviceArray < glm::mat4>& model_gpu, util::DeviceArray < glm::mat4>& projection_gpu, Eigen::Matrix<float, 2, 3> jacobian_mvp,
	//	util::DeviceArray<float>&  jacobian, int nUnknowns, util::DeviceArray<float>&  residuals, Face& face);

};