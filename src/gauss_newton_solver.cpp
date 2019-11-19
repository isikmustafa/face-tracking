#include "gauss_newton_solver.h"
#include "prior_sparse_features.h"

#include <Eigen/Dense>

void GaussNewtonSolver::solve(const std::vector<glm::vec2>& sparse_features, Face& face, const glm::mat4& projection)
{
	//TODO(Wojtek): We should have 60 sparse features..
	int numof_sparse_features = sparse_features.size();
	
	//TODO(Wojtek): When we also optimize for expression and shape coefficients, prior_local_positions
	//will not be valid since these are taken from vertices of the average face.
	const auto& prior_local_positions = PriorSparseFeatures::getPriorPositions();
	const auto& prior_local_ids = PriorSparseFeatures::getPriorIds();

	Eigen::VectorXf residuals(numof_sparse_features * 2);
	Eigen::MatrixXf jacobian_hom(numof_sparse_features * 2, 3);
	Eigen::Matrix<float, 3, 3> jacobian_proj;
	Eigen::Matrix<float, 3, 6> jacobian_pose;

	jacobian_proj <<
		projection[0][0], 0.0f, 0.0f,
		0.0f, projection[1][1], 0.0f,
		0.0f, 0.0f, -1.0f;

	//Construct residuals
	//TODO(Mustafa): Process this loop with OpenMP and compare.
	auto face_pose = face.computeModelMatrix();
	for (int i = 0; i < numof_sparse_features; ++i)
	{
		auto local_coord = prior_local_positions[prior_local_ids[i]];
		auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
		auto proj_coord = projection * world_coord;
		auto uv = glm::vec2(proj_coord.x, proj_coord.y) / proj_coord.w;

		int index_u = i * 2;
		int index_v = index_u + 1;
		
		//Residual
		auto residual = sparse_features[i] - uv;
		residual *= residual;
		residuals(index_u) = residual.x;
		residuals(index_v) = residual.y;

		//Jacobian for homogenization (AKA division by w)
		auto one_over_wp = 1.0f / proj_coord.w;
		jacobian_hom(index_u, 0) = one_over_wp;
		jacobian_hom(index_u, 1) = 0.0f;
		jacobian_hom(index_u, 2) = -proj_coord.x * one_over_wp * one_over_wp;

		jacobian_hom(index_v, 0) = 0.0f;
		jacobian_hom(index_v, 1) = one_over_wp;
		jacobian_hom(index_v, 2) = -proj_coord.y * one_over_wp * one_over_wp;
	}
}