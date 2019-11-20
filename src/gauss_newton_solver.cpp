#include "gauss_newton_solver.h"
#include "prior_sparse_features.h"

#include <Eigen/Dense>

void GaussNewtonSolver::solve(const std::vector<glm::vec2>& sparse_features, Face& face, const glm::mat4& projection)
{
	//TODO(Wojtek): We should have 60 sparse features..
	int numof_sparse_features = sparse_features.size();
	std::cout << numof_sparse_features << std::endl;

	//TODO(Wojtek): When we also optimize for expression and shape coefficients, prior_local_positions
	//will not be valid since these are taken from vertices of the average face.
	const auto& prior_local_positions = PriorSparseFeatures::get().getPriorPositions();
	const auto& prior_local_ids = PriorSparseFeatures::get().getPriorIds();
	auto& rotation_coefficients = face.getRotationCoefficients();
	auto& translation_coefficients = face.getTranslationCoefficients();

	Eigen::VectorXf residuals(numof_sparse_features * 2);
	Eigen::MatrixXf jacobian(numof_sparse_features * 2, 6);
	Eigen::Matrix<float, 2, 3> jacobian_hom;
	Eigen::Matrix<float, 3, 3> jacobian_proj;
	Eigen::Matrix<float, 3, 6> jacobian_pose;

	jacobian_proj <<
		projection[0][0], 0.0f, 0.0f,
		0.0f, projection[1][1], 0.0f,
		0.0f, 0.0f, -1.0f;

	int number_of_gn_iterations = 5;
	for (int iteration = 0; iteration < number_of_gn_iterations; ++iteration)
	{
		auto face_pose = face.computeModelMatrix();
		glm::mat3 drx, dry, drz;
		face.computeRotationDerivatives(drx, dry, drz);

		//Construct residuals
		//TODO(Mustafa): Process this loop with OpenMP and compare.
		for (int i = 0; i < numof_sparse_features; ++i)
		{
			auto local_coord = prior_local_positions[i];
			auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
			auto proj_coord = projection * world_coord;
			auto uv = glm::vec2(proj_coord.x, proj_coord.y) / proj_coord.w;

			//Residual
			auto residual = sparse_features[i] - uv;

			residuals(i * 2) = residual.x;
			residuals(i * 2 + 1) = residual.y;

			//Jacobian for homogenization (AKA division by w)
			auto one_over_wp = 1.0f / proj_coord.w;
			jacobian_hom(0, 0) = one_over_wp;
			jacobian_hom(0, 1) = 0.0f;
			jacobian_hom(0, 2) = -proj_coord.x * one_over_wp * one_over_wp;

			jacobian_hom(1, 0) = 0.0f;
			jacobian_hom(1, 1) = one_over_wp;
			jacobian_hom(1, 2) = -proj_coord.y * one_over_wp * one_over_wp;

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

			jacobian_pose(0, 3) = 1.0f;
			jacobian_pose(1, 3) = 0.0f;
			jacobian_pose(2, 3) = 0.0f;
			jacobian_pose(0, 4) = 0.0f;
			jacobian_pose(1, 4) = 1.0f;
			jacobian_pose(2, 4) = 0.0f;
			jacobian_pose(0, 5) = 0.0f;
			jacobian_pose(1, 5) = 0.0f;
			jacobian_pose(2, 5) = 1.0f;

			jacobian.block<2, 6>(i * 2, 0) = jacobian_hom * jacobian_proj * jacobian_pose;
		}

		// Apply step and update poses
		auto jacobian_t = jacobian.transpose();
		auto jtj = jacobian_t * jacobian;
		auto jtr = -jacobian_t * residuals;

		Eigen::JacobiSVD<Eigen::MatrixXf> svd(jtj, Eigen::ComputeThinU | Eigen::ComputeThinV);
		auto result = svd.solve(jtr);

		std::cout << svd.rank() << std::endl;

		rotation_coefficients.x -= result(0);
		rotation_coefficients.y -= result(1);
		rotation_coefficients.z -= result(2);

		translation_coefficients.x -= result(3);
		translation_coefficients.y -= result(4);
		translation_coefficients.z -= result(5);

		/*std::cout << "result: " << result << std::endl;
		std::cout << "Iteration: " << iteration << " , Loss: " << (residuals.array() * residuals.array()).sum() << std::endl;*/
	}
}