﻿#pragma once

#include "gauss_newton_solver.h"
#include "util.h"
#include "jacobian_util.h"
#include "device_util.h"
#include "device_array.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

/**
 * Compute Jacobian matrix for parametric model w.r.t fov of virtual camera, Rotation, Translation, α, β, δ, γ
 * (α, β, δ) are parametric face model Eigen basis scaling factors
 * γ is illumination model (Spherical harmonics)
 * 
 * Optimization energy
 * P = (intrinsics, pose, α, β, δ, γ)
 * E(P) = w_col * E_col(P) + w_lan * E_lan(P) + w_reg * E_reg(P)
 * 
 */
__global__ void cuComputeJacobianSparseDense(
	// Shared memory
	FaceBoundingBox face_bb,
	const int nFeatures, const int imageWidth, const int imageHeight,
	const int nFaceCoeffs, const int nPixels, const int n,
	const int nShapeCoeffs, const int nExpressionCoeffs, const int nAlbedoCoeffs,
	const int nUnknowns, const int nResiduals,
	const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal, const int nAlbedoCoeffsTotal,
	const float wSparse, float wDense, const float wReg,

	uchar* image,

	glm::mat4 face_pose, glm::mat3 drx, glm::mat3 dry, glm::mat3 drz, glm::mat4 projection, Eigen::Matrix3f jacobian_local,

	// Device memory input
	int* prior_local_ids, glm::vec3* current_face, glm::vec2* sparse_features,

	float* p_shape_basis,
	float* p_expression_basis,
	float* p_albedo_basis,

	float* p_coefficients_shape,
	float* p_coefficients_expression,
	float* p_coefficients_albedo,
	float* p_coefficients_sh,

	cudaTextureObject_t rgb,
	cudaTextureObject_t barycentrics,
	cudaTextureObject_t vertex_ids,

	// Device memory output
	float* p_jacobian, float* p_residuals)
{
	int i = util::getThreadIndex1D();

	if (i >= n)
	{
		return;
	}

	Eigen::Map<Eigen::MatrixXf> jacobian(p_jacobian, nResiduals, nUnknowns);
	Eigen::Map<Eigen::VectorXf> residuals(p_residuals, nResiduals);

	Eigen::Map<Eigen::MatrixXf> shape_basis(p_shape_basis, nVerticesTimes3, nShapeCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> expression_basis(p_expression_basis, nVerticesTimes3, nExpressionCoeffsTotal);
	Eigen::Map<Eigen::MatrixXf> albedo_basis(p_albedo_basis, nVerticesTimes3, nAlbedoCoeffsTotal);

	// Regularization terms based on the assumption of a normal distributed population
	if (i >= nFeatures + nPixels)
	{
		int offset_rows = nFeatures * 2 + nPixels * 3;
		int offset_cols = 7;

		const int current_index = i - nFeatures - nPixels;
		const int expression_shift = nShapeCoeffs;
		const int albedo_shift = nShapeCoeffs + nExpressionCoeffs;

		float coefficient = 0.0f;
		int relative_index = current_index;

		// Shape
		if (current_index < expression_shift)
		{
			coefficient = p_coefficients_shape[relative_index];
		}
		// Expression
		else if (current_index < albedo_shift)
		{
			offset_rows += expression_shift;
			offset_cols += expression_shift;
			relative_index -= expression_shift;

			coefficient = p_coefficients_expression[relative_index];
		}
		// Albedo
		else
		{
			offset_rows += albedo_shift;
			offset_cols += albedo_shift;
			relative_index -= albedo_shift;

			coefficient = p_coefficients_albedo[relative_index];
		}

		jacobian(offset_rows + relative_index, offset_cols + relative_index) = wReg;
		residuals(offset_rows + relative_index) = coefficient * wReg;

		return;
	}

	/*
	 * Photo-Consistency dense energy term
	 * E = sum(norm_l2(C_S - C_I))
	 * where C_S is synthesized image
	 *	 C_I is input RGB image
	 *
	 */
	if (i >= nFeatures)
	{
		int offset_rows = nFeatures * 2;

		const int current_index = i - nFeatures;
		unsigned int xp = current_index % face_bb.width + face_bb.x_min;
		unsigned int yp = current_index / face_bb.width + face_bb.y_min;

		int background_index = 3 * (xp + yp * imageWidth);
		int ygl = imageHeight - 1 - yp; // "height - 1 - index.y" OpenGL uses left-bottom corner as texture origin.
		float4 rgb_sampled = tex2D<float4>(rgb, xp, ygl);

		if (rgb_sampled.w < 1.0f) // pixel is not covered by face
		{
			return;
		}

		float4 barycentrics_sampled = tex2D<float4>(barycentrics, xp, ygl);
		int4 vertex_ids_sampled = tex2D<int4>(vertex_ids, xp, ygl);
		Eigen::Map<Eigen::Vector3f> face_rgb(reinterpret_cast<float*>(&rgb_sampled));
		Eigen::Vector3f frame_rgb;

		frame_rgb.x() = image[background_index] / 255.0f;
		frame_rgb.y() = image[background_index + 1] / 255.0f;
		frame_rgb.z() = image[background_index + 2] / 255.0f;

		Eigen::Vector3f residual = face_rgb - frame_rgb;

		// IRLS with L1 norm.
		wDense /= glm::sqrt(glm::max(residual.norm(), 1.0e-8f));

		residuals.block(offset_rows + current_index * 3, 0, 3, 1) = residual * wDense;

		/*
		 * Energy derivation
		 * dE/dC_S
		 * The derivative with respect to synthesized image
		 * Fragment shader derivation
		 *
		 * Albedo derivation
		 * Color = Light * Albedo
		 * Albedo = A(E_alb_A * β) + B(E_alb_B * β) + C(E_alb_C * β) => barycentric coordinates
		 * dColor/dAlbedo
		 */
		jacobian.block(offset_rows + current_index * 3, 7 + nShapeCoeffs + nExpressionCoeffs, 3, nAlbedoCoeffs) =
			(barycentrics_sampled.w * wDense * barycentrics_sampled.x) * albedo_basis.block(3 * vertex_ids_sampled.x, 0, 3, nAlbedoCoeffs) +
			(barycentrics_sampled.w * wDense * barycentrics_sampled.y) * albedo_basis.block(3 * vertex_ids_sampled.y, 0, 3, nAlbedoCoeffs) +
			(barycentrics_sampled.w * wDense * barycentrics_sampled.z) * albedo_basis.block(3 * vertex_ids_sampled.z, 0, 3, nAlbedoCoeffs);

		/*
		 * Spherical harmonics derivation
		 * Color = SH(normal) * Albedo
		 * dSH/d{9 coefficients}
		 *
		 * see file /shaders/face.frag computeSH(normal)
		 */
		auto number_of_vertices = nVerticesTimes3 / 3;
		auto albedos = current_face + number_of_vertices;
		auto normals = current_face + 2 * number_of_vertices;

		auto normal_a_unnorm_glm = glm::mat3(face_pose) * normals[vertex_ids_sampled.x];
		auto normal_b_unnorm_glm = glm::mat3(face_pose) * normals[vertex_ids_sampled.y];
		auto normal_c_unnorm_glm = glm::mat3(face_pose) * normals[vertex_ids_sampled.z];

		auto normal_a_glm = glm::normalize(normal_a_unnorm_glm);
		auto normal_b_glm = glm::normalize(normal_b_unnorm_glm);
		auto normal_c_glm = glm::normalize(normal_c_unnorm_glm);

		auto albedo_glm = barycentrics_sampled.x * albedos[vertex_ids_sampled.x] + barycentrics_sampled.y * albedos[vertex_ids_sampled.y] + barycentrics_sampled.z * albedos[vertex_ids_sampled.z];
		auto normal_unnorm_glm = barycentrics_sampled.x * normal_a_glm + barycentrics_sampled.y * normal_b_glm + barycentrics_sampled.z * normal_c_glm;
		auto normal_glm = glm::normalize(normal_unnorm_glm);

		Eigen::Vector3f albedo;
		albedo << albedo_glm.x, albedo_glm.y, albedo_glm.z;

		// dSH/d{9 coefficients}
		Eigen::Matrix<float, 1, 9> bands(9);

		bands(0, 0) = 1.0f;
		bands(0, 1) = normal_glm.y;
		bands(0, 2) = normal_glm.z;
		bands(0, 3) = normal_glm.x;
		bands(0, 4) = normal_glm.x * normal_glm.y;
		bands(0, 5) = normal_glm.y * normal_glm.z;
		bands(0, 6) = 3.0f * normal_glm.z * normal_glm.z - 1.0f;
		bands(0, 7) = normal_glm.x * normal_glm.z;
		bands(0, 8) = normal_glm.x * normal_glm.x - normal_glm.y * normal_glm.y;

		jacobian.block<3, 9>(offset_rows + current_index * 3, 7 + nShapeCoeffs + nExpressionCoeffs + nAlbedoCoeffs) = wDense * albedo * bands;

		/* 
		 * Expression and shape derivations
		 * Triangle = (A, B, C)
		 * Normal{X, Y, Z} = normalize(cross(B - A, C - A))
		 * Color = SH(Normal) * Albedo
		 * @ref https://www.lighthouse3d.com/tutorials/glsl-12-tutorial/normalization-issues/
		 *
		 * dColor/dα and dColor/dδ
		 *
		 * Example of chain rule:
		 * dColor/dα = dColor/dSH * dSH/dNormal * dNormal/dNormalize() * dNormalize/dCross() * dCross()/d{(B - A), (C - A), A}
		 */
		Eigen::Matrix<float, 1, 3> dlight_dnormal;
		jacobian_util::computeDLightDNormal(dlight_dnormal, normal_glm, p_coefficients_sh);

		Eigen::Matrix<float, 3, 3> dnormal_dunnormnormal;
		jacobian_util::computeNormalizationJacobian(dnormal_dunnormnormal, normal_unnorm_glm);

		Eigen::Matrix<float, 3, 3> unnormnormal_jacobian = albedo * dlight_dnormal * dnormal_dunnormnormal;

		Eigen::Matrix<float, 3, 3> v0_jacobian;
		Eigen::Matrix<float, 3, 3> v1_jacobian;
		Eigen::Matrix<float, 3, 3> v2_jacobian;

		jacobian_util::computeNormalJacobian(v0_jacobian, v1_jacobian, v2_jacobian,
			current_face[vertex_ids_sampled.x], current_face[vertex_ids_sampled.y], current_face[vertex_ids_sampled.z]);

		unnormnormal_jacobian = wDense * unnormnormal_jacobian * jacobian_local;
		v0_jacobian = unnormnormal_jacobian * v0_jacobian;
		v1_jacobian = unnormnormal_jacobian * v1_jacobian;
		v2_jacobian = unnormnormal_jacobian * v2_jacobian;

		// dColor/dα
		jacobian.block(offset_rows + current_index * 3, 7, 3, nShapeCoeffs) =
			v0_jacobian * shape_basis.block(3 * vertex_ids_sampled.x, 0, 3, nShapeCoeffs) +
			v1_jacobian * shape_basis.block(3 * vertex_ids_sampled.y, 0, 3, nShapeCoeffs) +
			v2_jacobian * shape_basis.block(3 * vertex_ids_sampled.z, 0, 3, nShapeCoeffs);

		// dColor/dδ
		jacobian.block(offset_rows + current_index * 3, 7 + nShapeCoeffs, 3, nExpressionCoeffs) =
			v0_jacobian * expression_basis.block(3 * vertex_ids_sampled.x, 0, 3, nExpressionCoeffs) +
			v1_jacobian * expression_basis.block(3 * vertex_ids_sampled.y, 0, 3, nExpressionCoeffs) +
			v2_jacobian * expression_basis.block(3 * vertex_ids_sampled.z, 0, 3, nExpressionCoeffs);

		Eigen::Matrix<float, 3, 3> dnormal_dunnormnormal_sum = Eigen::MatrixXf::Zero(3, 3);

		// For 1st vertex normal
		jacobian_util::computeNormalizationJacobian(dnormal_dunnormnormal, normal_a_unnorm_glm);
		dnormal_dunnormnormal_sum += barycentrics_sampled.x * dnormal_dunnormnormal;

		// For 2nd vertex normal
		jacobian_util::computeNormalizationJacobian(dnormal_dunnormnormal, normal_b_unnorm_glm);
		dnormal_dunnormnormal_sum += barycentrics_sampled.y * dnormal_dunnormnormal;

		// For 3rd vertex normal
		jacobian_util::computeNormalizationJacobian(dnormal_dunnormnormal, normal_c_unnorm_glm);
		dnormal_dunnormnormal_sum += barycentrics_sampled.z * dnormal_dunnormnormal;

		Eigen::Matrix<float, 3, 3> jacobian_rotation;

		auto dx = drx * normals[vertex_ids_sampled.x];
		auto dy = dry * normals[vertex_ids_sampled.y];
		auto dz = drz * normals[vertex_ids_sampled.z];

		jacobian_rotation <<
			dx[0], dy[0], dz[0],
			dx[1], dy[1], dz[1],
			dx[2], dy[2], dz[2];

		jacobian.block<3, 3>(offset_rows + current_index * 3, 1) = unnormnormal_jacobian * dnormal_dunnormnormal_sum * jacobian_rotation * wDense;

		/* 
		 * Energy derivation
		 * -dE/dC_I
		 * The derivative with respect to source image (frame_rgb)
		 * Full perspective derivation
		 * @ref http://www.songho.ca/opengl/gl_transform.html
		 */

		// Take into account barycentric interpolation in the fragment shader for vertices and their attributes
		auto local_coord = 
			barycentrics_sampled.x * current_face[vertex_ids_sampled.x] +
			barycentrics_sampled.y * current_face[vertex_ids_sampled.y] +
			barycentrics_sampled.z * current_face[vertex_ids_sampled.z];

		auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
		auto proj_coord = projection * world_coord;

		// Derivative of source image (screen coordinate system) with respect to (u,v)
		// TODO: Check for boundary for xp and yp
		Eigen::Matrix<float, 3, 2> jacobian_uv;

		int background_index_left = 3 * (xp - 1 + yp * imageWidth);
		int background_index_right = 3 * (xp + 1 + yp * imageWidth);
		int background_index_up = 3 * (xp + (yp - 1) * imageWidth);
		int background_index_down = 3 * (xp + (yp + 1) * imageWidth);

		/*
		 * Central difference derivation for color ([c(i+1)-c(i-1)]/2p) with viewport transformation derivation
		 * Color => Screen => NDC
		 */

		// dColor/du
		jacobian_uv(0, 0) = -(image[background_index_right] / 255.0f - image[background_index_left] / 255.0f) * 0.25f * imageWidth;
		jacobian_uv(1, 0) = -(image[background_index_right + 1] / 255.0f - image[background_index_left + 1] / 255.0f) *  0.25f * imageWidth;
		jacobian_uv(2, 0) = -(image[background_index_right + 2] / 255.0f - image[background_index_left + 2] / 255.0f) *  0.25f * imageWidth;

		// dColor/dv
		jacobian_uv(0, 1) = (image[background_index_down] / 255.0f - image[background_index_up] / 255.0f) *  0.25f  * imageHeight;
		jacobian_uv(1, 1) = (image[background_index_down + 1] / 255.0f - image[background_index_up + 1] / 255.0f) *  0.25f  * imageHeight;
		jacobian_uv(2, 1) = (image[background_index_down + 2] / 255.0f - image[background_index_up + 2] / 255.0f) *  0.25f  * imageHeight;

		// Jacobian for homogenization (AKA division by w)
		// NCD => Clip coordinates
		Eigen::Matrix<float, 2, 3> jacobian_proj;

		auto one_over_wp = 1.0f / proj_coord.w;

		jacobian_proj(0, 0) = one_over_wp;
		jacobian_proj(0, 1) = 0.0f;
		jacobian_proj(0, 2) = -proj_coord.x * one_over_wp * one_over_wp;

		jacobian_proj(1, 0) = 0.0f;
		jacobian_proj(1, 1) = one_over_wp;
		jacobian_proj(1, 2) = -proj_coord.y * one_over_wp * one_over_wp;

		/*
		 * Jacobian for projection
		 * Clip coordinates => Eye coordinates (which is identity matrix I, since the camera is assumed to be at the origin)
		 * dProjection/dX_world
		 * dProjection/dY_world
		 * dProjection/dW_world
		 */
		Eigen::Matrix<float, 3, 3> jacobian_world = Eigen::MatrixXf::Zero(3, 3);

		jacobian_world(0, 0) = projection[0][0];
		jacobian_world(1, 1) = projection[1][1];
		jacobian_world(2, 2) = -1.0f;

		/*
		 * Jacobian for intrinsics (change of fov in our virtual camera)
		 * dPerspectiveRH_NO/dFov
		 * @ref glm::perspectiveRH_NO()
		 */
		Eigen::Matrix<float, 3, 1> jacobian_intrinsics = Eigen::MatrixXf::Zero(3, 1);

		jacobian_intrinsics(0, 0) = world_coord.x;
		jacobian.block<3, 1>(offset_rows + current_index * 3, 0) = jacobian_uv * jacobian_proj * jacobian_intrinsics * wDense;

		/*
		 * Derivative of world coordinates with respect to rotation coefficients
		 * Since this node (world space) in our computation graph is common for [R, T] as well as expression and shape
		 * we can branch the calculations out and derive jacobian_pose first.
		 * World coordinates => Local coordinates
		 * X_world = R * X_local + T
		 * dX_world/dR and dX_world/dT
		 */
		dx = drx * local_coord;
		dy = dry * local_coord;
		dz = drz * local_coord;

		Eigen::Matrix<float, 3, 6> jacobian_pose = Eigen::MatrixXf::Zero(3, 6);

		jacobian_pose(0, 3) = 1.0f;
		jacobian_pose(1, 4) = 1.0f;
		jacobian_pose(2, 5) = 1.0f;
		jacobian_pose(0, 0) = dx[0];
		jacobian_pose(1, 0) = dx[1];
		jacobian_pose(2, 0) = dx[2];
		jacobian_pose(0, 1) = dy[0];
		jacobian_pose(1, 1) = dy[1];
		jacobian_pose(2, 1) = dy[2];
		jacobian_pose(0, 2) = dz[0];
		jacobian_pose(1, 2) = dz[1];
		jacobian_pose(2, 2) = dz[2];

		auto jacobian_proj_world = jacobian_uv * jacobian_proj * jacobian_world;

		jacobian.block<3, 6>(offset_rows + current_index * 3, 1) += jacobian_proj_world * jacobian_pose * wDense;

		/*
		 * Derivative of world coordinates with respect to local coordinates.
		 * This is the rotation matrix.
		 * X_world = R * X_local + T
		 * dX_world/dX_local = R
		 */
		auto jacobian_proj_world_local = jacobian_proj_world * jacobian_local * wDense;

		// Derivative of local coordinates with respect to shape and expression parameters
		jacobian.block(offset_rows + current_index * 3, 7, 3, nShapeCoeffs) +=
			(jacobian_proj_world_local * barycentrics_sampled.x) * shape_basis.block(3 * vertex_ids_sampled.x, 0, 3, nShapeCoeffs) +
			(jacobian_proj_world_local * barycentrics_sampled.y) * shape_basis.block(3 * vertex_ids_sampled.y, 0, 3, nShapeCoeffs) +
			(jacobian_proj_world_local * barycentrics_sampled.z) * shape_basis.block(3 * vertex_ids_sampled.z, 0, 3, nShapeCoeffs);

		jacobian.block(offset_rows + current_index * 3, 7 + nShapeCoeffs, 3, nExpressionCoeffs) +=
			(jacobian_proj_world_local * barycentrics_sampled.x) * expression_basis.block(3 * vertex_ids_sampled.x, 0, 3, nExpressionCoeffs) +
			(jacobian_proj_world_local * barycentrics_sampled.y) * expression_basis.block(3 * vertex_ids_sampled.y, 0, 3, nExpressionCoeffs) +
			(jacobian_proj_world_local * barycentrics_sampled.z) * expression_basis.block(3 * vertex_ids_sampled.z, 0, 3, nExpressionCoeffs);

		return;
	}

	/*
	 * Sparse terms for Feature Alignment
	 * Feature similarity between a set of salient facial feature point pairs detect
	 * 
	 * E = sum(l2_norm(f - Π(Φ(local_coord))^2)
	 * where Π(Φ()) is full perspective projection
	 */
	auto vertex_id = prior_local_ids[i];
	auto local_coord = current_face[vertex_id];

	auto world_coord = face_pose * glm::vec4(local_coord, 1.0f);
	auto proj_coord = projection * world_coord;
	auto uv = glm::vec2(proj_coord.x, proj_coord.y) / proj_coord.w;

	// Residual
	auto residual = uv - sparse_features[i];

	residuals(i * 2) = residual.x * wSparse;
	residuals(i * 2 + 1) = residual.y * wSparse;

	// Jacobians follow the same description like in the case of dense features

	// Jacobian for homogenization (AKA division by w)
	Eigen::Matrix<float, 2, 3> jacobian_proj;

	auto one_over_wp = 1.0f / proj_coord.w;
	jacobian_proj(0, 0) = one_over_wp;
	jacobian_proj(0, 1) = 0.0f;
	jacobian_proj(0, 2) = -proj_coord.x * one_over_wp * one_over_wp;

	jacobian_proj(1, 0) = 0.0f;
	jacobian_proj(1, 1) = one_over_wp;
	jacobian_proj(1, 2) = -proj_coord.y * one_over_wp * one_over_wp;

	// Jacobian for projection
	Eigen::Matrix<float, 3, 3> jacobian_world = Eigen::MatrixXf::Zero(3, 3);

	jacobian_world(0, 0) = projection[0][0];
	jacobian_world(1, 1) = projection[1][1];
	jacobian_world(2, 2) = -1.0f;

	// Jacobian for intrinsics
	Eigen::Matrix<float, 3, 1> jacobian_intrinsics = Eigen::MatrixXf::Zero(3, 1);

	jacobian_intrinsics(0, 0) = world_coord.x;
	jacobian.block<2, 1>(i * 2, 0) = jacobian_proj * jacobian_intrinsics * wSparse;

	// Derivative of world coordinates with respect to rotation coefficients
	auto dx = drx * local_coord;
	auto dy = dry * local_coord;
	auto dz = drz * local_coord;

	Eigen::Matrix<float, 3, 6> jacobian_pose = Eigen::MatrixXf::Zero(3, 6);

	jacobian_pose(0, 3) = 1.0f;
	jacobian_pose(1, 4) = 1.0f;
	jacobian_pose(2, 5) = 1.0f;
	jacobian_pose(0, 0) = dx[0];
	jacobian_pose(1, 0) = dx[1];
	jacobian_pose(2, 0) = dx[2];
	jacobian_pose(0, 1) = dy[0];
	jacobian_pose(1, 1) = dy[1];
	jacobian_pose(2, 1) = dy[2];
	jacobian_pose(0, 2) = dz[0];
	jacobian_pose(1, 2) = dz[1];
	jacobian_pose(2, 2) = dz[2];

	auto jacobian_proj_world = jacobian_proj * jacobian_world * wSparse;
	jacobian.block<2, 6>(i * 2, 1) = jacobian_proj_world * jacobian_pose;

	// Derivative of world coordinates with respect to local coordinates.
	// This is basically the rotation matrix.
	auto jacobian_proj_world_local = jacobian_proj_world * jacobian_local;

	// Derivative of local coordinates with respect to shape and expression parameters
	// This is basically the corresponding (to unique vertices we have chosen) rows of basis matrices.
	auto jacobian_shape = jacobian_proj_world_local * shape_basis.block(3 * vertex_id, 0, 3, nShapeCoeffs);
	jacobian.block(i * 2, 7, 2, nShapeCoeffs) = jacobian_shape;

	auto jacobian_expression = jacobian_proj_world_local * expression_basis.block(3 * vertex_id, 0, 3, nExpressionCoeffs);
	jacobian.block(i * 2, 7 + nShapeCoeffs, 2, nExpressionCoeffs) = jacobian_expression;
}

__global__ void cuComputeVisiblePixelsAndBB(cudaTextureObject_t texture, FaceBoundingBox* face_bb, int width, int height)
{
	auto index = util::getThreadIndex2D();
	if (index.x >= width || index.y >= height)
	{
		return;
	}

	int y = height - 1 - index.y; // "height - 1 - index.y" is used since OpenGL uses left-bottom corner as texture origin.
	float4 color = tex2D<float4>(texture, index.x, y);

	if (color.w > 0.0f)
	{
		atomicInc(&face_bb->num_visible_pixels, UINT32_MAX);
		atomicMin(&face_bb->x_min, index.x);
		atomicMin(&face_bb->y_min, index.y);
		atomicMax(&face_bb->x_max, index.x);
		atomicMax(&face_bb->y_max, index.y);
	}
}

FaceBoundingBox GaussNewtonSolver::computeFaceBoundingBox(const int imageWidth, const int imageHeight)
{
	FaceBoundingBox bb;
	util::copy(m_face_bb, &bb, 1);

	//TODO: Arrange this (16,16) according to TitanX when we use it.
	dim3 threads_meta(16, 16);
	dim3 blocks_meta(imageWidth / threads_meta.x + 1, imageHeight / threads_meta.y + 1);

	cuComputeVisiblePixelsAndBB << <blocks_meta, threads_meta >> > (m_texture_rgb, m_face_bb.getPtr(), imageWidth, imageHeight);

	util::copy(&bb, m_face_bb, 1);
	//std::cout << bb.num_visible_pixels << " " << bb.x_min << " " << bb.y_min << " " << bb.x_max << " " << bb.y_max << std::endl;

	if (bb.num_visible_pixels <= 0 || bb.x_min >= bb.x_max || bb.y_min >= bb.y_max)
	{
		std::cout << "Warning: invalid face bounding box!" << std::endl;
	}

	bb.width = bb.x_max - bb.x_min;
	bb.height = bb.y_max - bb.y_min;

	return bb;
}

void GaussNewtonSolver::computeJacobian(
	//shared memory
	const FaceBoundingBox face_bb,
	const int nFeatures, const int imageWidth, const int imageHeight,
	const int nShapeCoeffs, const int nExpressionCoeffs, const int nAlbedoCoeffs,
	const int nUnknowns, const int nResiduals,
	const int nVerticesTimes3, const int nShapeCoeffsTotal, const int nExpressionCoeffsTotal, const int nAlbedoCoeffsTotal, const int nShcoeffsTotal,
	float sparseWeight, float denseWeight, float regularizationWeight,

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
	float* p_coefficients_sh,

	//device memory output
	float* p_jacobian, float* p_residuals
) const
{
	const int nPixels = face_bb.width * face_bb.height;

	const int nFaceCoeffs = nShapeCoeffs + nExpressionCoeffs + nAlbedoCoeffs;
	const int n = nFeatures + nPixels + nFaceCoeffs;

	//TODO: Fine tune these configs according to TitanX in the end.
	const int threads = 128;
	const int block = (n + threads - 1) / threads;

	auto time = util::runKernelGetExecutionTime([&]() {cuComputeJacobianSparseDense << <block, threads >> > (
		//shared memory
		face_bb,
		nFeatures, imageWidth, imageHeight,
		nFaceCoeffs, nPixels, n,
		nShapeCoeffs, nExpressionCoeffs, nAlbedoCoeffs,
		nUnknowns, nResiduals,
		nVerticesTimes3, nShapeCoeffsTotal, nExpressionCoeffsTotal, nAlbedoCoeffsTotal,
		glm::sqrt(sparseWeight / nFeatures), glm::sqrt(denseWeight / face_bb.num_visible_pixels), glm::sqrt(regularizationWeight),

		image,

		face_pose, drx, dry, drz, projection, jacobian_local,

		//device memory input
		prior_local_ids, current_face, sparse_features,

		p_shape_basis,
		p_expression_basis,
		p_albedo_basis,

		p_coefficients_shape,
		p_coefficients_expression,
		p_coefficients_albedo,
		p_coefficients_sh,

		m_texture_rgb,
		m_texture_barycentrics,
		m_texture_vertex_ids,

		//device memory output
		p_jacobian, p_residuals
		);
	});

	std::cout << "Jacobian kernel time: " << time << std::endl;

	cudaDeviceSynchronize();
}

__global__ void cuComputeJTJDiagonals(const int nUnknowns, const int nCurrentResiduals, const int nResiduals, float* jacobian, float* preconditioner)
{
	int tid = threadIdx.x;
	int col = blockIdx.x;

	float sum = 0.0f;
	for (int row = tid; row < nCurrentResiduals; row += blockDim.x)
	{
		auto v = jacobian[col * nResiduals + row];
		sum += v * v;
	}

	atomicAdd(&preconditioner[col], sum);
}

__global__ void cuElementwiseMultiplication(float* v1, float* v2, float* out)
{
	int i = util::getThreadIndex1D();
	out[i] = v1[i] * v2[i];
}

__global__ void cuOneOverElement(float* preconditioner)
{
	int i = util::getThreadIndex1D();

	preconditioner[i] = 1.0f / glm::max(preconditioner[i], 1.0e-4f);
}

void GaussNewtonSolver::computeJacobiPreconditioner(const int nUnknowns, const int nCurrentResiduals, const int nResiduals, float* jacobian, float* preconditioner)
{
	cuComputeJTJDiagonals << <nUnknowns, 128 >> > (nUnknowns, nCurrentResiduals, nResiduals, jacobian, preconditioner);
	cudaDeviceSynchronize();

	cuOneOverElement << <1, nUnknowns >> > (preconditioner);
	cudaDeviceSynchronize();
}

void GaussNewtonSolver::elementwiseMultiplication(const int nElements, float* v1, float* v2, float* out)
{
	cuElementwiseMultiplication << <1, nElements >> > (v1, v2, out);
	cudaDeviceSynchronize();
}