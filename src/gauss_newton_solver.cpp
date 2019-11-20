#include "gauss_newton_solver.h"
#include "prior_sparse_features.h"

#include <Eigen/Dense>

void GaussNewtonSolver::solve(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection)
{
	int numof_sparse_features = sparse_features.size();

	//TODO(Wojtek): When we also optimize for expression and shape coefficients, prior_local_positions
	//will not be valid since these are taken from vertices of the average face.
	const auto& prior_local_positions = PriorSparseFeatures::get().getPriorPositions();
	const auto& prior_local_ids = PriorSparseFeatures::get().getPriorIds();
	auto& rotation_coefficients = face.getRotationCoefficients();
	auto& translation_coefficients = face.getTranslationCoefficients();

	Eigen::VectorXf residuals(numof_sparse_features * 2);
	Eigen::MatrixXf jacobian(numof_sparse_features * 2, 7); //3+3+1 = 7 DoF for rotation, translation and intrinsics.

	//Some parts of jacobians are constants. That's why thet are intialized here only once.
	//Do not touch them inside the for loops.
	Eigen::Matrix<float, 2, 3> jacobian_proj;
	jacobian_proj(0, 1) = 0.0f;
	jacobian_proj(1, 0) = 0.0f;

	Eigen::Matrix<float, 3, 3> jacobian_world;
	jacobian_world(0, 1) = 0.0f;
	jacobian_world(0, 2) = 0.0f;
	jacobian_world(1, 0) = 0.0f;
	jacobian_world(1, 1) = projection[1][1];
	jacobian_world(1, 2) = 0.0f;
	jacobian_world(2, 0) = 0.0f;
	jacobian_world(2, 1) = 0.0f;
	jacobian_world(2, 2) = -1.0f;

	Eigen::Matrix<float, 3, 1> jacobian_intrinsics;
	jacobian_intrinsics(1, 0) = 0.0f;
	jacobian_intrinsics(2, 0) = 0.0f;

	Eigen::Matrix<float, 3, 6> jacobian_pose;
	jacobian_pose(0, 3) = 1.0f;
	jacobian_pose(1, 3) = 0.0f;
	jacobian_pose(2, 3) = 0.0f;
	jacobian_pose(0, 4) = 0.0f;
	jacobian_pose(1, 4) = 1.0f;
	jacobian_pose(2, 4) = 0.0f;
	jacobian_pose(0, 5) = 0.0f;
	jacobian_pose(1, 5) = 0.0f;
	jacobian_pose(2, 5) = 1.0f;

	Eigen::Matrix<float, 3, 3> jacobian_local;

	int number_of_gn_iterations = 5;
	for (int iteration = 0; iteration < number_of_gn_iterations; ++iteration)
	{
		auto face_pose = face.computeModelMatrix();
		jacobian_local <<
			face_pose[0][0], face_pose[1][0], face_pose[2][0],
			face_pose[0][1], face_pose[1][1], face_pose[2][1],
			face_pose[0][2], face_pose[1][2], face_pose[2][2];

		glm::mat3 drx, dry, drz;
		face.computeRotationDerivatives(drx, dry, drz);

		//Construct residuals and jacobian
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
			//TODO:
		}

		//Apply step and update poses
		auto jacobian_t = jacobian.transpose();
		auto jtj = jacobian_t * jacobian;
		auto jtr = -jacobian_t * residuals;

		Eigen::JacobiSVD<Eigen::MatrixXf> svd(jtj, Eigen::ComputeThinU | Eigen::ComputeThinV);
		auto result = svd.solve(jtr);

		projection[0][0] -= result(0);

		rotation_coefficients.x -= result(1);
		rotation_coefficients.y -= result(2);
		rotation_coefficients.z -= result(3);

		translation_coefficients.x -= result(4);
		translation_coefficients.y -= result(5);
		translation_coefficients.z -= result(6);

		/*
		std::cout << "Aspect Ratio: " << projection[1][1] / projection[0][0] << std::endl;
		std::cout << "System Rank: " << svd.rank() << std::endl;
		std::cout << "Result: " << result << std::endl;
		std::cout << "Iteration: " << iteration << " , Loss: " << (residuals.array() * residuals.array()).sum() << std::endl;
		*/
	}
}