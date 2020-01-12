#pragma once

#include "face.h"
#include <Eigen/Dense>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//Default
struct SolverParameters
{
	float regularisation_weight_exponent = -4.6f;

	int num_gn_iterations = 5;
	int num_pcg_iterations = 15;

	int num_shape_coefficients = 30;
	int num_albedo_coefficients = 30;
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

	void solve(const std::vector<glm::vec2>& sparse_features, Face& face, cv::Mat& frame, glm::mat4& projection);
	void solve_CPU(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection);

	SolverParameters& getSolverParameters() { return m_params; }
	const SolverParameters& getSolverParameters() const { return m_params; }

private:
	cublasHandle_t m_cublas;
	SolverParameters m_params;
	cudaTextureObject_t m_texture_rgb{ 0 };
	cudaTextureObject_t m_texture_barycentrics{ 0 };
	cudaTextureObject_t m_texture_vertex_ids{ 0 };

private:
	void computeJacobianSparseFeatures(
		//shared memory
		int nFeatures, const int imageWidth, const int imageHeight,
		int nShapeCoeffs, int nExpressionCoeffs, int nAlbedoCoeffs,
		int nUnknowns, int nResiduals,
		int nVerticesTimes3, int nShapeCoeffsTotal, int nExpressionCoeffsTotal, int nAlbedoCoeffsTotal,
		float regularizationWeight,

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
		float* p_jacobian, float* p_residuals) const;

	void elementwiseMultiplication(int nElements, float* v1, float* v2, float* out);

	void computeJacobiPreconditioner(int nUnknowns, int nResiduals, float* p_jacobian, float* p_preconditioner);

	void solveUpdateCG(const cublasHandle_t& cublas, int nUnknowns, int nResiduals, util::DeviceArray<float>& jacobian,
		util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, float alphaLHS = 1, float alphaRHS = 1);

	void solveUpdatePCG(const cublasHandle_t& cublas, int nUnknowns, int nResiduals, util::DeviceArray<float>& jacobian,
		util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, float alphaLHS = 1, float alphaRHS = 1);

	void solveUpdateLU(const cublasHandle_t& cublas, int nUnknowns, int nResiduals, util::DeviceArray<float>& jacobian,
		util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, float alphaLHS = 1, float alphaRHS = 1);

	void updateParameters(const std::vector<float>& result, glm::mat4& projection,
		glm::vec3& rotation_coefficients, glm::vec3& translation_coefficients, Face& face, int nShapeCoeffs, int nExpressionCoeffs, int nAlbedoCoeffs);

	void mapRenderTargets(Face& face);
	void unmapRenderTargets(Face& face);
	void debugFrameBufferTextures(Face& face, uchar* frame, const std::string& rgb_filepath, const std::string& deferred_filepath);
	void destroyTextures();
};