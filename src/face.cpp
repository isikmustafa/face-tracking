#include "face.h"
#include "glsl_program.h"
#include "util.h"
#include "prior_sparse_features.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <glm/gtx/euler_angles.hpp>
#include <Eigen/Dense>

Face::Face(const std::string& morphable_model_directory)
	: m_sh_coefficients(9, 0.0f)
	, m_rotation_coefficients(0.0f, 0.0f, 0.0f)
	, m_translation_coefficients(0.0f, 0.0f, -0.4f)
{
	std::ifstream file(morphable_model_directory + "/nomouth.off");
	std::string str_dummy;
	file >> str_dummy;

	int number_of_faces;
	int int_dummy;
	file >> m_number_of_vertices;
	file >> number_of_faces;
	file >> int_dummy;

	std::vector<glm::vec3> positions(m_number_of_vertices);
	std::vector<glm::vec3> colors(m_number_of_vertices);

	m_sh_coefficients[0] = 0.5;

	m_number_of_indices = 3 * number_of_faces;
	std::vector<unsigned int> indices(m_number_of_indices);
	std::vector<glm::ivec3> faces(number_of_faces);
	constexpr float mesh_scale = 1 / 1000000.0f;
	for (int i = 0; i < m_number_of_vertices; ++i)
	{
		file >> positions[i].x >> positions[i].y >> positions[i].z;
		positions[i] *= mesh_scale;

		file >> colors[i].x >> colors[i].y >> colors[i].z;
		colors[i] *= 1.0f / 255.0f;
		file >> int_dummy;
	}

	for (int i = 0; i < number_of_faces; ++i)
	{
		file >> int_dummy;
		file >> indices[i * 3] >> indices[i * 3 + 1] >> indices[i * 3 + 2];
		faces[i].x = indices[i * 3];
		faces[i].y = indices[i * 3 + 1];
		faces[i].z = indices[i * 3 + 2];
	}
	file.close();

	//We will only update position, color and normals of vertices. In order not to copy the constant texture coordinates,
	//we dont allocate memory for them.
	m_average_face_gpu = util::DeviceArray<glm::vec3>(m_number_of_vertices * 3);
	m_current_face_gpu = util::DeviceArray<glm::vec3>(m_number_of_vertices * 3);

	m_average_face_gpu.memset(0); //Important thing is normals are initialized to 0. Whenever we copy m_average_face_gpu to m_current_face_gpu, normals will be reset.
	util::copy(m_average_face_gpu, positions, m_number_of_vertices);
	util::copy(m_average_face_gpu, colors, m_number_of_vertices, m_number_of_vertices, 0);

	m_faces_gpu = util::DeviceArray<glm::ivec3>(number_of_faces);
	util::copy(m_faces_gpu, faces, number_of_faces);

	glGenVertexArrays(1, &m_vertex_array);
	glGenBuffers(1, &m_vertex_buffer);
	glGenBuffers(1, &m_index_buffer);

	assert(m_vertex_array);
	assert(m_vertex_buffer);
	assert(m_index_buffer);

	glBindVertexArray(m_vertex_array);

	int positions_byte_size = m_number_of_vertices * sizeof(glm::vec3);
	int colors_byte_size = m_number_of_vertices * sizeof(glm::vec3);
	int normals_byte_size = m_number_of_vertices * sizeof(glm::vec3);
	int tex_coords_byte_size = m_number_of_vertices * sizeof(glm::vec2);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, positions_byte_size + colors_byte_size + normals_byte_size + tex_coords_byte_size, nullptr, GL_STATIC_DRAW);
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&m_resource, m_vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));

	updateVertexBuffer();

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)positions_byte_size);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)(positions_byte_size + colors_byte_size));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_number_of_indices * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	std::vector<float> shape_basis = loadModelData(morphable_model_directory + "/ShapeBasis_modified.matrix", true);
	auto shape_std_dev = loadModelData(morphable_model_directory + "/StandardDeviationShape.vec", false);
	m_shape_coefficients.resize(shape_std_dev.size(), 0.0f);
	Eigen::Map<Eigen::MatrixXf> shape_basis_eigen(shape_basis.data(), m_number_of_vertices * 3, m_shape_coefficients.size());
	Eigen::Map<Eigen::VectorXf> shape_std_dev_eigen(shape_std_dev.data(), shape_std_dev.size());
	shape_basis_eigen = shape_basis_eigen.array().rowwise() * shape_std_dev_eigen.transpose().array();

	m_shape_basis_gpu = util::DeviceArray<float>(shape_basis);
	m_shape_coefficients_gpu = util::DeviceArray<float>(m_shape_coefficients.size());

	std::vector<float> albedo_basis = loadModelData(morphable_model_directory + "/AlbedoBasis_modified.matrix", true);
	auto albedo_std_dev = loadModelData(morphable_model_directory + "/StandardDeviationAlbedo.vec", false);
	m_albedo_coefficients.resize(albedo_std_dev.size(), 0.0f);
	Eigen::Map<Eigen::MatrixXf> albedo_basis_eigen(albedo_basis.data(), m_number_of_vertices * 3, m_albedo_coefficients.size());
	Eigen::Map<Eigen::VectorXf> albedo_std_dev_dev_eigen(albedo_std_dev.data(), albedo_std_dev.size());
	albedo_basis_eigen = albedo_basis_eigen.array().rowwise() * albedo_std_dev_dev_eigen.transpose().array();

	m_albedo_basis_gpu = util::DeviceArray<float>(albedo_basis);
	m_albedo_coefficients_gpu = util::DeviceArray<float>(m_albedo_coefficients.size());

	std::vector<float> expression_basis = loadModelData(morphable_model_directory + "/ExpressionBasis_modified.matrix", true);
	auto expression_std_dev = loadModelData(morphable_model_directory + "/StandardDeviationExpression.vec", false);
	m_expression_coefficients.resize(expression_std_dev.size(), 0.0f);
	Eigen::Map<Eigen::MatrixXf> expression_basis_eigen(expression_basis.data(), m_number_of_vertices * 3, m_expression_coefficients.size());
	Eigen::Map<Eigen::VectorXf> expression_std_dev_dev_eigen(expression_std_dev.data(), expression_std_dev.size());
	expression_basis_eigen = expression_basis_eigen.array().rowwise() * expression_std_dev_dev_eigen.transpose().array();

	m_expression_basis_gpu = util::DeviceArray<float>(expression_basis);
	m_expression_coefficients_gpu = util::DeviceArray<float>(m_expression_coefficients.size());

	cublasCreate(&m_cublas);
}

Face::~Face()
{
	if (m_vertex_buffer)
	{
		glDeleteBuffers(1, &m_vertex_buffer);
		m_vertex_buffer = 0;
	}
	if (m_index_buffer)
	{
		glDeleteBuffers(1, &m_index_buffer);
		m_index_buffer = 0;
	}
	if (m_vertex_array)
	{
		glDeleteVertexArrays(1, &m_vertex_array);
		m_vertex_array = 0;
	}
	cublasDestroy(m_cublas);
}

void Face::computeFace()
{
	util::copy(m_shape_coefficients_gpu, m_shape_coefficients, m_shape_coefficients.size());
	util::copy(m_albedo_coefficients_gpu, m_albedo_coefficients, m_albedo_coefficients.size());
	util::copy(m_expression_coefficients_gpu, m_expression_coefficients, m_expression_coefficients.size());
	util::copy(m_current_face_gpu, m_average_face_gpu, m_average_face_gpu.getSize());

	float alpha = 1.0f;
	float beta = 1.0f;
	int m = 3 * m_number_of_vertices;
	int n = m_shape_coefficients.size();
	cublasSgemv(m_cublas, CUBLAS_OP_N, m, n, &alpha, m_shape_basis_gpu.getPtr(), m, m_shape_coefficients_gpu.getPtr(), 1, &beta,
		reinterpret_cast<float*>(m_current_face_gpu.getPtr()), 1);

	n = m_albedo_coefficients.size();
	cublasSgemv(m_cublas, CUBLAS_OP_N, m, n, &alpha, m_albedo_basis_gpu.getPtr(), m, m_albedo_coefficients_gpu.getPtr(), 1, &beta,
		reinterpret_cast<float*>(m_current_face_gpu.getPtr()) + m, 1);

	n = m_expression_coefficients.size();
	cublasSgemv(m_cublas, CUBLAS_OP_N, m, n, &alpha, m_expression_basis_gpu.getPtr(), m, m_expression_coefficients_gpu.getPtr(), 1, &beta,
		reinterpret_cast<float*>(m_current_face_gpu.getPtr()), 1);

	computeNormals();
}

glm::mat4 Face::computeModelMatrix() const
{
	auto model_matrix = glm::orientate4(m_rotation_coefficients);
	model_matrix[3] = glm::vec4(m_translation_coefficients, 1.0f);

	return model_matrix;
}

void Face::computeRotationDerivatives(glm::mat3& drx, glm::mat3& dry, glm::mat3& drz) const
{
	float cos_z = glm::cos(m_rotation_coefficients.z);
	float sin_z = glm::sin(m_rotation_coefficients.z);
	float cos_x = glm::cos(m_rotation_coefficients.x);
	float sin_x = glm::sin(m_rotation_coefficients.x);
	float cos_y = glm::cos(m_rotation_coefficients.y);
	float sin_y = glm::sin(m_rotation_coefficients.y);

	//Derivate with respecto to angle x
	drx[0][0] = sin_z * cos_x * sin_y;
	drx[1][0] = sin_z * cos_x * cos_y;
	drx[2][0] = -sin_z * sin_x;

	drx[0][1] = -sin_y * sin_x;
	drx[1][1] = -cos_y * sin_x;
	drx[2][1] = -cos_x;

	drx[0][2] = cos_z * cos_x * sin_y;
	drx[1][2] = cos_z * cos_x * cos_y;
	drx[2][2] = -cos_z * sin_x;

	//Derivate with respecto to angle y
	dry[0][0] = -cos_z * sin_y + sin_z * sin_x * cos_y;
	dry[1][0] = -cos_z * cos_y - sin_z * sin_x * sin_y;
	dry[2][0] = 0.0f;

	dry[0][1] = cos_y * cos_x;
	dry[1][1] = -sin_y * cos_x;
	dry[2][1] = 0.0f;

	dry[0][2] = sin_z * sin_y + cos_z * sin_x * cos_y;
	dry[1][2] = cos_y * sin_z - cos_z * sin_x * sin_y;
	dry[2][2] = 0.0f;

	//Derivate with respecto to angle z
	drz[0][0] = -sin_z * cos_y + cos_z * sin_x * sin_y;
	drz[1][0] = sin_z * sin_y + cos_z * sin_x * cos_y;
	drz[2][0] = cos_z * cos_x;

	drz[0][1] = 0.0f;
	drz[1][1] = 0.0f;
	drz[2][1] = 0.0f;

	drz[0][2] = -cos_z * cos_y - sin_z * sin_x * sin_y;
	drz[1][2] = sin_y * cos_z - sin_z * sin_x * cos_y;
	drz[2][2] = -sin_z * cos_x;
}

void Face::updateVertexBuffer()
{
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &m_resource, 0));
	void* vertex_buffer_ptr;
	size_t size;
	CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&vertex_buffer_ptr, &size, m_resource));
	CHECK_CUDA_ERROR(cudaMemcpy(vertex_buffer_ptr, m_current_face_gpu.getPtr(), m_number_of_vertices * sizeof(glm::vec3) * 3, cudaMemcpyDeviceToDevice));

	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_resource, 0));
}

void Face::draw() const
{
	if (m_graphics_settings.mapped_to_cuda)
	{
		throw std::runtime_error("Error: Draw is called while rts is mapped!");
	}

	// Render to face framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_graphics_settings.framebuffer);
	glViewport(0, 0, m_graphics_settings.texture_width, m_graphics_settings.texture_height);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	m_graphics_settings.shader->use();
	m_graphics_settings.shader->setMat4("model", computeModelMatrix());
	m_graphics_settings.shader->setUniformFVVar("sh_coefficients", getSHCoefficients());

	glBindVertexArray(m_vertex_array);
	glDrawElements(GL_TRIANGLES, m_number_of_indices, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

//Only load .matrix file with _modified suffix.
//You can use this for any .vec file.
std::vector<float> Face::loadModelData(const std::string& filename, bool is_basis)
{
	std::ifstream file(filename, std::ifstream::binary);

	unsigned int size;
	file.read((char*)&size, sizeof(unsigned int));
	std::vector<float> basis(size);
	file.read((char*)basis.data(), size * sizeof(float));
	file.close();

	return basis;
}
