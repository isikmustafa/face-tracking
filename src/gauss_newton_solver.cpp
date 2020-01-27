#include "gauss_newton_solver.h"
#include "prior_sparse_features.h"
#include "util.h"
#include "device_util.h"
#include "device_array.h"

#include <Eigen/Dense>
#include <chrono>

GaussNewtonSolver::GaussNewtonSolver()
	: m_face_bb(1)
	, m_sh_coefficients_gpu(9)
{
	cublasCreate(&m_cublas);
}

GaussNewtonSolver::~GaussNewtonSolver()
{
	cublasDestroy(m_cublas);
	destroyTextures();
}

void GaussNewtonSolver::solve(const std::vector<glm::vec2>& sparse_features, Face& face, cv::Mat& frame, glm::mat4& projection)
{
	if (sparse_features.empty()) //no tracking -> cublas doesnt like a getting matrix/vector of size 0
	{
		return;
	}

	const int frameWidth = face.m_graphics_settings.texture_width;
	const int frameHeight = face.m_graphics_settings.texture_height;
	const int nPixels = frameWidth * frameHeight;
	const int nFeatures = sparse_features.size();
	const int nShapeCoeffs = m_params.num_shape_coefficients;
	const int nExpressionCoeffs = m_params.num_expression_coefficients;
	const int nAlbedoCoeffs = m_params.num_albedo_coefficients;
	const int nFaceCoeffs = nShapeCoeffs + nExpressionCoeffs + nAlbedoCoeffs;
	const int nResiduals = 2 * nFeatures + 3 * nPixels + nFaceCoeffs; //nFaceCoeffs -> regularizer
	const int nUnknowns = 7 + nFaceCoeffs + 9; //3+3+1 = 7 DoF for rotation, translation and intrinsics. Plus nFaceCoeffs for face parameters and 9 for lighting.

	const float wSparse = std::powf(10, m_params.sparse_weight_exponent);
	const float wDense = std::powf(10, m_params.dense_weight_exponent);
	const float wReg = std::powf(10, m_params.regularisation_weight_exponent);

	const auto& prior_local_ids = PriorSparseFeatures::get().getPriorIds();

	//TODO: Allocate all of the objects below once. So, move them out of here.
	auto jacobian_gpu = util::DeviceArray<float>(nResiduals * nUnknowns);
	auto residuals_gpu = util::DeviceArray<float>(nResiduals);
	auto result_gpu = util::DeviceArray<float>(nUnknowns);
	std::vector<float> result(nUnknowns);
	auto ids_gpu = util::DeviceArray<int>(prior_local_ids);
	auto key_pts_gpu = util::DeviceArray<glm::vec2>(sparse_features);

	cv::Mat processed_frame;
	cv::resize(frame, processed_frame, cv::Size(frameWidth, frameHeight));
	cv::cvtColor(processed_frame, processed_frame, cv::COLOR_BGR2RGB);
	util::DeviceArray<uchar> frame_gpu = util::DeviceArray<uchar>(3 * nPixels);
	util::copy(frame_gpu, processed_frame.data, 3 * nPixels);

	//Some parts of jacobians are constants. That's why they are intialized here only once.
	//Do not touch them inside the for loops.
	Eigen::Matrix<float, 3, 3> jacobian_local = Eigen::MatrixXf::Zero(3, 3);

	for (int iteration = 0; iteration < m_params.num_gn_iterations; ++iteration)
	{
		jacobian_gpu.memset(0);
		residuals_gpu.memset(0);
		face.computeFace();
		face.updateVertexBuffer();
		face.draw();

		auto face_pose = face.computeModelMatrix();
		jacobian_local <<
			face_pose[0][0], face_pose[1][0], face_pose[2][0],
			face_pose[0][1], face_pose[1][1], face_pose[2][1],
			face_pose[0][2], face_pose[1][2], face_pose[2][2];

		glm::mat3 drx, dry, drz;
		face.computeRotationDerivatives(drx, dry, drz);

		mapRenderTargets(face);
		FaceBoundingBox face_bb = computeFaceBoundingBox(face.m_graphics_settings.texture_width, face.m_graphics_settings.texture_height);

		int n_current_residuals = 2 * nFeatures + nFaceCoeffs + 3 * face_bb.width * face_bb.height;

		//debugFrameBufferTextures(face, frame_gpu.getPtr(), "..//..//rgb.png", "..//..//rgb-deferred.png");
		util::copy(m_sh_coefficients_gpu, face.m_sh_coefficients, 9);

		//CUDA
		computeJacobian(
			//shared memory
			face_bb,
			nFeatures, frameWidth, frameHeight,
			nShapeCoeffs, nExpressionCoeffs, nAlbedoCoeffs, nUnknowns, n_current_residuals,
			face.m_number_of_vertices * 3,
			face.m_shape_coefficients.size(),
			face.m_expression_coefficients.size(),
			face.m_albedo_coefficients.size(),
			face.m_sh_coefficients.size(),
			wSparse, wDense, wReg,

			frame_gpu.getPtr(),

			face_pose, drx, dry, drz, projection, jacobian_local,

			//device memory input
			ids_gpu.getPtr(), face.m_current_face_gpu.getPtr(), key_pts_gpu.getPtr(),

			face.m_shape_basis_gpu.getPtr(),
			face.m_expression_basis_gpu.getPtr(),
			face.m_albedo_basis_gpu.getPtr(),

			face.m_shape_coefficients_gpu.getPtr(),
			face.m_expression_coefficients_gpu.getPtr(),
			face.m_albedo_coefficients_gpu.getPtr(),
			m_sh_coefficients_gpu.getPtr(),

			//device memory output
			jacobian_gpu.getPtr(), residuals_gpu.getPtr()
		);

		unmapRenderTargets(face);

		//Apply step and update poses GPU
		auto solve_time = util::runKernelGetExecutionTime([&]() {
			solveUpdatePCG(m_cublas, nUnknowns, n_current_residuals, nResiduals, jacobian_gpu, residuals_gpu, result_gpu, 1.0f, -1.0f);
		});
		std::cout << "Solve time: " << solve_time << std::endl;
		util::copy(result, result_gpu, nUnknowns);

		updateParameters(result, projection, face, nShapeCoeffs, nExpressionCoeffs, nAlbedoCoeffs);

		/*std::cout << "Aspect Ratio: " << projection[1][1] / projection[0][0] << std::endl;
		std::cout << "Unknowns: " << nUnknowns << ", Residuals: " << nResiduals << std::endl;
		std::cout << "Iteration: " << iteration << " , Loss: " << glm::sqrt(error) << std::endl;*/
	}
}

void GaussNewtonSolver::solveUpdateLU(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian,
	util::DeviceArray<float>& residuals, util::DeviceArray<float>& result, const float alphaLHS, const float alphaRHS)
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

/**/
void GaussNewtonSolver::solveUpdatePCG(const cublasHandle_t& cublas, const int nUnknowns, const int nCurrentResiduals, const int nResiduals, util::DeviceArray<float>& J,
	util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS, const float alphaRHS)
{
	const float alpha = 1.f, beta = 0;
	x.memset(0);

	auto r = util::DeviceArray<float>(nUnknowns);	//current residual
	auto p = util::DeviceArray<float>(nUnknowns);	//gradient 
	auto M = util::DeviceArray<float>(nUnknowns);	//preconditioner

	auto z = util::DeviceArray<float>(nUnknowns);	//preconditioned residual
	auto Jp = util::DeviceArray<float>(nCurrentResiduals);
	auto JTJp = util::DeviceArray<float>(nUnknowns);

	auto diagJTJ = util::DeviceArray<float>(nUnknowns);
	auto diagJTJp = util::DeviceArray<float>(nUnknowns);
	
	M.memset(0);
	diagJTJ.memset(0);
	diagJTJp.memset(0);

	computeDiagJTJ(nUnknowns, nCurrentResiduals, nResiduals, J.getPtr(), diagJTJ.getPtr());
	computeInverseJTJ(nUnknowns, diagJTJ.getPtr(), M.getPtr());

	//r = -JTf;
	cublasSgemv(cublas, CUBLAS_OP_T, nCurrentResiduals, nUnknowns, &alphaRHS, J.getPtr(), nCurrentResiduals, residuals.getPtr(), 1, &beta, r.getPtr(), 1);

	//z = Mr
	elementwiseMultiplication(nUnknowns, M.getPtr(), r.getPtr(), z.getPtr());

	//p=z;
	cublasScopy(cublas, nUnknowns, z.getPtr(), 1, p.getPtr(), 1);

	float lambda = 0.0000001f; // TODO adaptive lambda parameter
	float zTr_old = 0.f, zTr = 0.f;
	float pTJTJp;
	//zTr
	cublasSdot(cublas, nUnknowns, z.getPtr(), 1, r.getPtr(), 1, &zTr_old);
	int i = 0;
	for (; i < std::min(nUnknowns, m_params.num_pcg_iterations); ++i)
	{
		//apply JTJp
		cublasSgemv(cublas, CUBLAS_OP_N, nCurrentResiduals, nUnknowns, &alphaLHS, J.getPtr(), nCurrentResiduals, p.getPtr(), 1, &beta, Jp.getPtr(), 1);
		cublasSgemv(cublas, CUBLAS_OP_T, nCurrentResiduals, nUnknowns, &alpha, J.getPtr(), nCurrentResiduals, Jp.getPtr(), 1, &beta, JTJp.getPtr(), 1);

		//apply diag(JTJ)p
		elementwiseMultiplication(nUnknowns, diagJTJ.getPtr(), p.getPtr(), diagJTJp.getPtr());

		// add (lambda*diagJTJp + JTJp)
		cublasSaxpy(cublas, nUnknowns, &lambda, diagJTJp.getPtr(), 1, JTJp.getPtr(), 1);

		cublasSdot(cublas, nUnknowns, p.getPtr(), 1, JTJp.getPtr(), 1, &pTJTJp);

		float ak = zTr_old / std::max(pTJTJp, m_params.kNearZero);
		//x = ak*p + x
		cublasSaxpy(cublas, nUnknowns, &ak, p.getPtr(), 1, x.getPtr(), 1);

		//r = r - ak* JTJp
		ak *= -1;
		cublasSaxpy(cublas, nUnknowns, &ak, JTJp.getPtr(), 1, r.getPtr(), 1);

		//z=Mr
		elementwiseMultiplication(nUnknowns, M.getPtr(), r.getPtr(), z.getPtr());

		//zTr
		cublasSdot(cublas, nUnknowns, z.getPtr(), 1, r.getPtr(), 1, &zTr);

		if (zTr < m_params.kTolerance)
		{
			break;
		}

		float bk = zTr / std::max(zTr_old, m_params.kNearZero);

		//p = z + bk*p        
		cublasSscal(cublas, nUnknowns, &bk, p.getPtr(), 1);
		cublasSaxpy(cublas, nUnknowns, &alpha, z.getPtr(), 1, p.getPtr(), 1);

		zTr_old = zTr;
	}
	//	std::cout << "PCG iters: " << i << std::endl; 
}

float GaussNewtonSolver::solveUpdateCG(const cublasHandle_t& cublas, const int nUnknowns, const int nResiduals, util::DeviceArray<float>& jacobian,
	util::DeviceArray<float>& residuals, util::DeviceArray<float>& x, const float alphaLHS, const float alphaRHS)
{
	const float alpha = 1, beta = 0;
	x.memset(0);

	auto r = util::DeviceArray<float>(nUnknowns);	//current residual
	auto p = util::DeviceArray<float>(nUnknowns);	//gradient 
	auto Jp = util::DeviceArray<float>(nResiduals);
	auto JTJp = util::DeviceArray<float>(nUnknowns);

	//r = -JTf;
	cublasSgemv(cublas, CUBLAS_OP_T, nResiduals, nUnknowns, &alphaRHS, jacobian.getPtr(), nResiduals, residuals.getPtr(), 1, &beta, r.getPtr(), 1);

	//p=r;
	cublasScopy(cublas, nUnknowns, r.getPtr(), 1, p.getPtr(), 1);

	float rTr_old = 0, rTr;
	float pTJTJp;
	//rTr
	cublasSdot(cublas, nUnknowns, r.getPtr(), 1, r.getPtr(), 1, &rTr);
	int i = 0;
	auto num_of_iterations = std::min(nUnknowns, m_params.num_pcg_iterations);
	for (; i < num_of_iterations; ++i)
	{
		//apply JTJ
		cublasSgemv(cublas, CUBLAS_OP_N, nResiduals, nUnknowns, &alphaLHS, jacobian.getPtr(), nResiduals, p.getPtr(), 1, &beta, Jp.getPtr(), 1);
		cublasSgemv(cublas, CUBLAS_OP_T, nResiduals, nUnknowns, &alpha, jacobian.getPtr(), nResiduals, Jp.getPtr(), 1, &beta, JTJp.getPtr(), 1);

		rTr_old = rTr;

		cublasSdot(cublas, nUnknowns, p.getPtr(), 1, JTJp.getPtr(), 1, &pTJTJp);

		float ak = rTr / std::max(pTJTJp, m_params.kNearZero);
		//x = ak*p + x
		cublasSaxpy(cublas, nUnknowns, &ak, p.getPtr(), 1, x.getPtr(), 1);

		//r = r - ak* JTJp
		ak *= -1;
		cublasSaxpy(cublas, nUnknowns, &ak, JTJp.getPtr(), 1, r.getPtr(), 1);

		//rTr
		cublasSdot(cublas, nUnknowns, r.getPtr(), 1, r.getPtr(), 1, &rTr);

		if (rTr < m_params.kTolerance)
		{
			break;
		}

		float bk = rTr / std::max(rTr_old, m_params.kNearZero);

		//p = r + bk*p        
		cublasSscal(cublas, nUnknowns, &bk, p.getPtr(), 1);
		cublasSaxpy(cublas, nUnknowns, &alpha, r.getPtr(), 1, p.getPtr(), 1);
	}

	return rTr;
}

void GaussNewtonSolver::updateParameters(const std::vector<float>& result, glm::mat4& projection, Face& face,
	const int nShapeCoeffs, const int nExpressionCoeffs, const int nAlbedoCoeffs)
{
	projection[0][0] += result[0];

	face.m_rotation_coefficients.x += result[1];
	face.m_rotation_coefficients.y += result[2];
	face.m_rotation_coefficients.z += result[3];

	face.m_translation_coefficients.x += result[4];
	face.m_translation_coefficients.y += result[5];
	face.m_translation_coefficients.z += result[6];

#pragma omp parallel num_threads(4)
	{
#pragma omp single
		for (int i = 0; i < nShapeCoeffs; ++i)
		{
			face.m_shape_coefficients[i] += result[7 + i];
		}

#pragma omp single
		for (int i = 0; i < nExpressionCoeffs; ++i)
		{
			auto c = face.m_expression_coefficients[i] + result[7 + nShapeCoeffs + i];
			face.m_expression_coefficients[i] = glm::clamp(c, 0.0f, 1.0f);
		}

#pragma omp single
		for (int i = 0; i < nAlbedoCoeffs; ++i)
		{
			face.m_albedo_coefficients[i] += result[7 + nShapeCoeffs + nExpressionCoeffs + i];
		}

#pragma omp single
		for (int i = 0; i < 9; ++i)
		{
			face.m_sh_coefficients[i] += result[7 + nShapeCoeffs + nExpressionCoeffs + nAlbedoCoeffs + i];
		}
	}
}

void GaussNewtonSolver::mapRenderTargets(Face& face)
{
	if (face.m_graphics_settings.mapped_to_cuda)
	{
		std::cout << "Warning: mapRenderTargets is called while rts is already mapped!" << std::endl;
		return;
	}

	cudaGraphicsResource* resources[] = { face.m_graphics_settings.rt_rgb_cuda_resource,
		face.m_graphics_settings.rt_barycentrics_cuda_resource,
		face.m_graphics_settings.rt_vertex_ids_cuda_resource };
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(3, resources, 0));

	cudaArray* array_rgb{ nullptr };
	cudaArray* array_barycentrics{ nullptr };
	cudaArray* array_vertex_ids{ nullptr };

	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array_rgb, resources[0], 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array_barycentrics, resources[1], 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array_vertex_ids, resources[2], 0, 0));

	//RGB texture
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = array_rgb;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeWrap);
	tex_desc.addressMode[1] = cudaTextureAddressMode(cudaAddressModeWrap);
	tex_desc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	tex_desc.normalizedCoords = 0;
	CHECK_CUDA_ERROR(cudaCreateTextureObject(&m_texture_rgb, &res_desc, &tex_desc, nullptr));

	//Barycentrics texture
	res_desc.res.array.array = array_barycentrics;
	tex_desc.filterMode = cudaTextureFilterMode(cudaFilterModePoint);
	tex_desc.readMode = cudaReadModeElementType;
	CHECK_CUDA_ERROR(cudaCreateTextureObject(&m_texture_barycentrics, &res_desc, &tex_desc, nullptr));

	//Vertex ids texture
	res_desc.res.array.array = array_vertex_ids;
	CHECK_CUDA_ERROR(cudaCreateTextureObject(&m_texture_vertex_ids, &res_desc, &tex_desc, nullptr));

	face.m_graphics_settings.mapped_to_cuda = true;
}

void GaussNewtonSolver::unmapRenderTargets(Face& face)
{
	if (!face.m_graphics_settings.mapped_to_cuda)
	{
		std::cout << "Warning: unmapRenderTargets is called while rts is already unmapped!" << std::endl;
		return;
	}

	destroyTextures();

	cudaGraphicsResource* resources[] = { face.m_graphics_settings.rt_rgb_cuda_resource,
		face.m_graphics_settings.rt_barycentrics_cuda_resource,
		face.m_graphics_settings.rt_vertex_ids_cuda_resource };
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(3, resources, 0));

	face.m_graphics_settings.mapped_to_cuda = false;
}

void GaussNewtonSolver::destroyTextures()
{
	if (m_texture_rgb)
	{
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(m_texture_rgb));
		m_texture_rgb = 0;
	}
	if (m_texture_barycentrics)
	{
		m_texture_barycentrics = 0;
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(m_texture_barycentrics));
	}
	if (m_texture_vertex_ids)
	{
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(m_texture_vertex_ids));
		m_texture_vertex_ids = 0;
	}
}
