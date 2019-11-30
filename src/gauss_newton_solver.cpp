#include "gauss_newton_solver.h"
#include "prior_sparse_features.h"
#include "util.h"
#include "device_util.h"
#include "device_array.h"
#include <Eigen/Dense>
#include <chrono>


GaussNewtonSolver::GaussNewtonSolver()
{
	cublasCreate(&m_cublas); 
}

GaussNewtonSolver::~GaussNewtonSolver()
{
	cublasDestroy(m_cublas); 
}



void GaussNewtonSolver::solve_CPU(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection)
{

	if (sparse_features.empty()) //no tracking -> cublas doesnt like a getting matrix/vector of size 0
		return; 

	int nFeatures = sparse_features.size();
	int nResiduals = 2 * nFeatures;
	int nShapeCoeffs = face.m_shape_coefficients.size(); 
	nShapeCoeffs = 30;
	int nExpressionCoeffs = face.m_expression_coefficients.size(); 
	//nExpressionCoeffs = 20; 
	int nFaceCoeffs = nShapeCoeffs + nExpressionCoeffs; 
	int nUnknowns = 7 + nFaceCoeffs;

	//TODO(Wojtek): When we also optimize for expression and shape coefficients, prior_local_positions
	//will not be valid since these are taken from vertices of the average face.
	//const auto& prior_local_positions = PriorSparseFeatures::get().getPriorPositions();
	const auto& prior_local_ids = PriorSparseFeatures::get().getPriorIds();
	auto& rotation_coefficients = face.getRotationCoefficients();
	auto& translation_coefficients = face.getTranslationCoefficients();

	Eigen::VectorXf residuals(nFeatures * 2);
	Eigen::MatrixXf jacobian(nFeatures * 2, nUnknowns); //3+3+1 = 7 DoF for rotation, translation and intrinsics.
	jacobian.setZero(); 

	auto jacobian_gpu = util::DeviceArray<float>(nUnknowns*nResiduals);
	auto residuals_gpu = util::DeviceArray<float>(nResiduals);
	auto result_gpu = util::DeviceArray<float>(nUnknowns);
	std::vector<float> result(nUnknowns);

	
	//auto ids_gpu = util::DeviceArray<int>(prior_local_ids);
	//auto keyPts_gpu = util::DeviceArray<glm::vec2>(sparse_features);
	//auto result_coeffs_gpu = util::DeviceArray<float>(nFaceCoeffs);

	Eigen::Map<Eigen::MatrixXf> shape_basis(face.m_shape_basis.data(), face.m_number_of_vertices * 3, face.m_shape_coefficients.size());
	Eigen::Map<Eigen::MatrixXf> expression_basis(face.m_expression_basis.data(), face.m_number_of_vertices * 3, face.m_expression_coefficients.size());

	//auto& shape_basis = face.m_shape_coefficients; 
	//auto& expression_basis = face.m_expression_basis; 

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

	//clear since we are tracking to model right now, so gradients wrt. eigenvalues are given wrt. average face
	//for (int i = 0; i < nShapeCoeffs; ++i)
	//{
	//	face.m_shape_coefficients[i] = 0;
	//}
	//for (int i = 0; i < nExpressionCoeffs; ++i)
	//{
	//	face.m_expression_coefficients[i] = 0;
	//}



	int number_of_gn_iterations = 5;
	for (int iteration = 0; iteration < number_of_gn_iterations; ++iteration)
	{
		face.computeFace();
		std::vector<glm::vec3> current_face(face.m_number_of_vertices);
		util::copy(current_face, face.m_current_face_gpu, face.m_number_of_vertices);

		auto face_pose = face.computeModelMatrix();
		jacobian_local <<
			face_pose[0][0], face_pose[1][0], face_pose[2][0],
			face_pose[0][1], face_pose[1][1], face_pose[2][1],
			face_pose[0][2], face_pose[1][2], face_pose[2][2];

		glm::mat3 drx, dry, drz;
		face.computeRotationDerivatives(drx, dry, drz);

		//Construct residuals and jacobian
		for (int i = 0; i < nFeatures; ++i)
		{
			auto vertexId = prior_local_ids[i]; 
			//auto local_coord = prior_local_positions[i];
			auto local_coord = current_face[vertexId];

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

			auto jacobian_shape =  jacobian_proj_world_local * shape_basis.block(3 * vertexId, 0, 3, nShapeCoeffs);
		//	std::cout << nUnknowns << " " << nFaceCoeffs << " " << nShapeCoeffs << " " << nExpressionCoeffs << " " << vertexId <<" "<< face.m_shape_basis.size() << " " << face.m_number_of_vertices<< std::endl; 
		//	std::cout << jacobian.size() <<" " <<jacobian.rows()<< " "<<jacobian.cols() <<std::endl; 
		
		//	std::cout << "J_block\n" << jacobian.block(i * 2, 7, 2, nShapeCoeffs) << std::endl; 
		//	std::cout << "shape_block\n" << shape_basis.block(0,0,3, nShapeCoeffs) << std::endl;

		//	std::cout << "shape_block\n" << shape_basis.block(3 * vertexId, 0, 3, nShapeCoeffs) << std::endl;

		//	std::cout << "J_shape\n"  <<jacobian_shape << std::endl;
			jacobian.block(i * 2, 7, 2, nShapeCoeffs) = jacobian_shape; 

			auto jacobian_expression = jacobian_proj_world_local * expression_basis.block(3 * vertexId, 0, 3, nExpressionCoeffs);
			jacobian.block(i * 2, 7 + nShapeCoeffs, 2, nExpressionCoeffs) = jacobian_expression;
		}

		//Apply step and update poses CPU
		/**/
		auto jacobian_t = jacobian.transpose();
		auto jtj = jacobian_t * jacobian;
		auto jtr = -jacobian_t * residuals;

		Eigen::JacobiSVD<Eigen::MatrixXf> svd(jtj, Eigen::ComputeThinU | Eigen::ComputeThinV);
		//auto result_eigen = svd.solve(jtr);
		// projection[0][0] -= result_eigen(0);

		//rotation_coefficients.x -= result_eigen(1);
		//rotation_coefficients.y -= result_eigen(2);
		//rotation_coefficients.z -= result_eigen(3);

		//translation_coefficients.x -= result_eigen(4);
		//translation_coefficients.y -= result_eigen(5);
		//translation_coefficients.z -= result_eigen(6);
		/**/

		//Apply step and update poses GPU

		util::copy(jacobian_gpu, jacobian.data(), nUnknowns*nResiduals); 
		util::copy(residuals_gpu, residuals.data(), nResiduals);

		solveUpdatePCG(m_cublas, nUnknowns, nResiduals, jacobian_gpu, residuals_gpu, result_gpu, 1, -1);
		util::copy(result, result_gpu, nUnknowns);


		projection[0][0] -= result[0];

		rotation_coefficients.x -= result[1];
		rotation_coefficients.y -= result[2];
		rotation_coefficients.z -= result[3];

		translation_coefficients.x -= result[4];
		translation_coefficients.y -= result[5];
		translation_coefficients.z -= result[6];




		float sca = 1;
#pragma omp parallel for
		for (int i = 0; i < nShapeCoeffs; ++i)
		{
			auto c = face.m_shape_coefficients[i] - result[7+ i] * sca;
			face.m_shape_coefficients[i] = std::max(-5.0f, std::min(5.0f, c));
			face.m_shape_coefficients[i] = c;

		}
#pragma omp parallel for
		for (int i = 0; i < nExpressionCoeffs; ++i)
		{
			auto c = face.m_expression_coefficients[i] - result[7 + nShapeCoeffs + i] * sca;
			face.m_expression_coefficients[i] = std::max(0.0f,std::min(0.1f,c));
			//face.m_expression_coefficients[i] = c;
		}


		//if (iteration % 5 == 0)
		//{
		//	//std::cout << "Aspect Ratio: " << projection[1][1] / projection[0][0] << std::endl;
		//	std::cout << "Unknowns: " << nUnknowns << ", Residuals: " << nResiduals << std::endl;
		//	std::cout << "System Rank: " << svd.rank() << std::endl;
		//	//std::cout << "Result: " << result << std::endl;
		//	std::cout << "Iteration: " << iteration << " , Loss: " << (residuals.array() * residuals.array()).sum() << std::endl;
		//}
			
	}
	//std::cout << "================END OF FRAME================" << std::endl; 
}


void GaussNewtonSolver::solve(const std::vector<glm::vec2>& sparse_features, Face& face, glm::mat4& projection)
{

	if (sparse_features.empty()) //no tracking -> cublas doesnt like a getting matrix/vector of size 0
		return;

	int nFeatures = sparse_features.size();
	int nResiduals = 2 * nFeatures;
	int nShapeCoeffs = face.m_shape_coefficients.size();
	nShapeCoeffs = 30;
	int nExpressionCoeffs = face.m_expression_coefficients.size();
	//nExpressionCoeffs = 20; 
	int nFaceCoeffs = nShapeCoeffs + nExpressionCoeffs;
	int nUnknowns = 7 + nFaceCoeffs;

	//TODO(Wojtek): When we also optimize for expression and shape coefficients, prior_local_positions
	//will not be valid since these are taken from vertices of the average face.
	//const auto& prior_local_positions = PriorSparseFeatures::get().getPriorPositions();
	const auto& prior_local_ids = PriorSparseFeatures::get().getPriorIds();
	auto& rotation_coefficients = face.getRotationCoefficients();
	auto& translation_coefficients = face.getTranslationCoefficients();


	auto jacobian_gpu = util::DeviceArray<float>(nUnknowns*nResiduals);
	auto residuals_gpu = util::DeviceArray<float>(nResiduals);
	auto result_gpu = util::DeviceArray<float>(nUnknowns);
	std::vector<float> result(nUnknowns);


	auto ids_gpu = util::DeviceArray<int>(prior_local_ids);
	auto keyPts_gpu = util::DeviceArray<glm::vec2>(sparse_features);

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

	//clear since we are tracking to model right now, so gradients wrt. eigenvalues are given wrt. average face
	//for (int i = 0; i < nShapeCoeffs; ++i)
	//{
	//	face.m_shape_coefficients[i] = 0;
	//}
	//for (int i = 0; i < nExpressionCoeffs; ++i)
	//{
	//	face.m_expression_coefficients[i] = 0;
	//}




	int number_of_gn_iterations = 15;
	for (int iteration = 0; iteration < number_of_gn_iterations; ++iteration)
	{

		auto start = std::chrono::high_resolution_clock::now();
		face.computeFace();
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "compute face time:  " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0 << std::endl;


		start = std::chrono::high_resolution_clock::now();


		auto face_pose = face.computeModelMatrix();
		jacobian_local <<
			face_pose[0][0], face_pose[1][0], face_pose[2][0],
			face_pose[0][1], face_pose[1][1], face_pose[2][1],
			face_pose[0][2], face_pose[1][2], face_pose[2][2];

		glm::mat3 drx, dry, drz;
		face.computeRotationDerivatives(drx, dry, drz);

		//CUDA
		//TODO: block stuff
		computeJacobianSparseFeatures(
			//shared memory
			nFeatures, nShapeCoeffs, nExpressionCoeffs, nUnknowns, nResiduals,
			face.m_number_of_vertices * 3, face.m_shape_coefficients.size(), face.m_expression_coefficients.size(),
			face_pose, drx, dry, drz, projection,
			jacobian_proj, jacobian_world, jacobian_intrinsics, jacobian_pose, jacobian_local,

			//device memory input
			ids_gpu.getPtr(), face.m_current_face_gpu.getPtr(), keyPts_gpu.getPtr(),
			face.m_shape_basis_gpu.getPtr(), face.m_expression_basis_gpu.getPtr(),

			//device memory output
			jacobian_gpu.getPtr(), residuals_gpu.getPtr()
			); 

		stop = std::chrono::high_resolution_clock::now();
		std::cout << "sparse feature time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0 << std::endl;

		//Apply step and update poses GPU
		start = std::chrono::high_resolution_clock::now();

		solveUpdatePCG(m_cublas, nUnknowns, nResiduals, jacobian_gpu, residuals_gpu, result_gpu, 1, -1);
		util::copy(result, result_gpu, nUnknowns);

		stop = std::chrono::high_resolution_clock::now();
		std::cout << "PCG time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0 << std::endl;

		projection[0][0] -= result[0];

		rotation_coefficients.x -= result[1];
		rotation_coefficients.y -= result[2];
		rotation_coefficients.z -= result[3];

		translation_coefficients.x -= result[4];
		translation_coefficients.y -= result[5];
		translation_coefficients.z -= result[6];




		float sca = 1;
#pragma omp parallel for
		for (int i = 0; i < nShapeCoeffs; ++i)
		{
			auto c = face.m_shape_coefficients[i] - result[7 + i] * sca;
			face.m_shape_coefficients[i] = std::max(-5.0f, std::min(5.0f, c));
			face.m_shape_coefficients[i] = c;

		}
#pragma omp parallel for
		for (int i = 0; i < nExpressionCoeffs; ++i)
		{
			auto c = face.m_expression_coefficients[i] - result[7 + nShapeCoeffs + i] * sca;
			face.m_expression_coefficients[i] = std::max(0.0f, std::min(0.1f, c));
			//face.m_expression_coefficients[i] = c;
		}


		//if (iteration % 5 == 0)
		//{
		//	//std::cout << "Aspect Ratio: " << projection[1][1] / projection[0][0] << std::endl;
		//	std::cout << "Unknowns: " << nUnknowns << ", Residuals: " << nResiduals << std::endl;
		//	std::cout << "System Rank: " << svd.rank() << std::endl;
		//	//std::cout << "Result: " << result << std::endl;
		//	std::cout << "Iteration: " << iteration << " , Loss: " << (residuals.array() * residuals.array()).sum() << std::endl;
		//}

	}
	//std::cout << "================END OF FRAME================" << std::endl; 
}

void GaussNewtonSolver::solveUpdateLU(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian, util::DeviceArray<float>& residuals, util::DeviceArray<float>& result, const float alphaLHS, const float alphaRHS)
{

	float alpha = 1, beta = 0;

	////transpose jacobian bc of stupid col major cublas BS
	//auto jacobian = util::DeviceArray<float>(nUnknowns*nResiduals);
	//cublasSgeam(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nResiduals, nUnknowns, &alpha, jacobianT.getPtr(), nUnknowns, &beta, jacobian.getPtr(), nResiduals, jacobian.getPtr(), nResiduals);



	//solve JTJd = JTf by computeing JTJ and JTf and using cublas LU solver(very bad)
	auto& JTf = result;
	auto JTJ = util::DeviceArray<float>(nUnknowns*nUnknowns);
	auto JTJinv = util::DeviceArray<float>(nUnknowns*nUnknowns);
	JTJ.memset(0);

	alpha = alphaRHS, beta = 0;
	//JTf
	cublasSgemv(cublas, CUBLAS_OP_T, nResiduals, nUnknowns, &alpha, jacobian.getPtr(), nResiduals, residuals.getPtr(), 1, &beta, JTf.getPtr(), 1);
	alpha = alphaLHS, beta = 0;
	//JTJ
	cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nUnknowns, nUnknowns, nResiduals, &alpha, jacobian.getPtr(), nResiduals, jacobian.getPtr(), nResiduals, &beta, JTJ.getPtr(), nUnknowns);

	cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);

	//int info = 0;
	auto batch = util::DeviceArray<float*>({ JTJ.getPtr() });
	auto info = util::DeviceArray<int>(1);
	auto pivot = util::DeviceArray<int>(nUnknowns);

	cublasSgetrfBatched(cublas, nUnknowns, batch.getPtr(), nUnknowns, pivot.getPtr(), info.getPtr(), 1);

	auto ibatch = util::DeviceArray<float*>({ JTJinv.getPtr() });
	cublasSgetriBatched(cublas, nUnknowns, batch.getPtr(), nUnknowns, pivot.getPtr(), ibatch.getPtr(), nUnknowns, info.getPtr(), 1);

	cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
	alpha = 1, beta = 0;
	cublasSgemv(cublas, CUBLAS_OP_N, nUnknowns, nUnknowns, &alpha, JTJinv.getPtr(), nUnknowns, JTf.getPtr(), 1, &beta, result.getPtr(), 1);


	/*cublasStrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, nUnknowns, 1, &alpha, JTJ.getPtr(), nUnknowns, JTf.getPtr(), nUnknowns);
	cublasStrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, nUnknowns, 1, &alpha, JTJ.getPtr(), nUnknowns, JTf.getPtr(), nUnknowns);
	*/


}

void GaussNewtonSolver::solveUpdatePCG(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian, util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS, const float alphaRHS)
{
	const float alpha = 1, beta = 0;
	const float NEAR_ZERO = 1.0e-8;		// interpretation of "zero"
	float TOLERANCE = 1.0e-10;			//convergence if rtr < TOLERANCE
	x.memset(0);
	//r = JTf;
	auto r = util::DeviceArray<float>(nUnknowns);	//curren residual
	auto p = util::DeviceArray<float>(nUnknowns);	//gradient 
	auto Jp = util::DeviceArray<float>(nResiduals);
	auto JTJp = util::DeviceArray<float>(nUnknowns);
	cublasSgemv(cublas, CUBLAS_OP_T, nResiduals, nUnknowns, &alphaRHS, jacobian.getPtr(), nResiduals, residuals.getPtr(), 1, &beta, r.getPtr(), 1);
	//p=r;
	cublasScopy(cublas, nUnknowns, r.getPtr(), 1, p.getPtr(), 1);

	float rtr_old = 0, rtr;
	float pTJTJp;
	//rTr
	cublasSdot(cublas, nUnknowns, r.getPtr(), 1, r.getPtr(), 1, &rtr);
	int i = 0;
	for (; i < std::min(nUnknowns, PCG_ITERS); ++i)
	{
		//apply JTJ
		cublasSgemv(cublas, CUBLAS_OP_N, nResiduals, nUnknowns, &alphaLHS, jacobian.getPtr(), nResiduals, p.getPtr(), 1, &beta, Jp.getPtr(), 1);
		cublasSgemv(cublas, CUBLAS_OP_T, nResiduals, nUnknowns, &alpha, jacobian.getPtr(), nResiduals, Jp.getPtr(), 1, &beta, JTJp.getPtr(), 1);

		rtr_old = rtr;

		cublasSdot(cublas, nUnknowns, p.getPtr(), 1, JTJp.getPtr(), 1, &pTJTJp);

		float ak = rtr / std::max(pTJTJp, NEAR_ZERO);
		//x = ak*p + x
		cublasSaxpy(cublas, nUnknowns, &ak, p.getPtr(), 1, x.getPtr(), 1);

		//r = r - ak* JTJp
		ak *= -1;
		cublasSaxpy(cublas, nUnknowns, &ak, JTJp.getPtr(), 1, r.getPtr(), 1);

		//rTr
		cublasSdot(cublas, nUnknowns, r.getPtr(), 1, r.getPtr(), 1, &rtr);

		if (rtr < TOLERANCE) break;

		float bk = rtr / std::max(rtr_old, NEAR_ZERO);

		//p = r + bk*p        
		cublasSscal(cublas, nUnknowns, &bk, p.getPtr(), 1);
		cublasSaxpy(cublas, nUnknowns, &alpha, r.getPtr(), 1, p.getPtr(), 1);
	}
}
