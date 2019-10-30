#include "face.h"
#include "glsl_program.h"
#include "util.h"

#include <assert.h>
#include <fstream>
#include <iostream>

Face::Face(const std::string& morphable_model_directory)
{
	std::ifstream file(morphable_model_directory + "/averageMesh.off");
	std::string str_dummy;
	file >> str_dummy;

	int number_of_faces;
	int int_dummy;
	file >> m_number_of_vertices;
	file >> number_of_faces;
	file >> int_dummy;

	std::vector<glm::vec3> positions(m_number_of_vertices);
	std::vector<glm::vec3> colors(m_number_of_vertices);
	std::vector<glm::vec2> tex_coords(m_number_of_vertices);
	m_number_of_indices = 3 * number_of_faces;
	std::vector<unsigned int> indices(m_number_of_indices);
	constexpr float mesh_scale = 1 / 1000000.0f;
	for (int i = 0; i < m_number_of_vertices; ++i)
	{
		file >> positions[i].x >> positions[i].y >> positions[i].z;
		positions[i] *= mesh_scale;

		file >> colors[i].x >> colors[i].y >> colors[i].z;
		colors[i] *= (1.0f / 255.0f);
		file >> int_dummy;

		file >> tex_coords[i].x >> tex_coords[i].y;
	}
	for (int i = 0; i < number_of_faces; ++i)
	{
		file >> int_dummy;
		file >> indices[i * 3] >> indices[i * 3 + 1] >> indices[i * 3 + 2];
	}
	file.close();

	int positions_byte_size = m_number_of_vertices * sizeof(glm::vec3);
	int colors_byte_size = m_number_of_vertices * sizeof(glm::vec3);
	int tex_coords_byte_size = m_number_of_vertices * sizeof(glm::vec2);

	//We will only update position and color of vertices. In order not to copy the constant texture coordinates,
	//we dont allocate memory for them.
	CHECK_CUDA_ERROR(cudaMalloc(&m_average_face_gpu, positions_byte_size + colors_byte_size));
	CHECK_CUDA_ERROR(cudaMalloc(&m_current_face_gpu, positions_byte_size + colors_byte_size));
	CHECK_CUDA_ERROR(cudaMemcpy(m_average_face_gpu, positions.data(), positions_byte_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(m_average_face_gpu + m_number_of_vertices * 3, colors.data(), colors_byte_size, cudaMemcpyHostToDevice));

	glGenVertexArrays(1, &m_vertex_array);
	glGenBuffers(1, &m_vertex_buffer);
	glGenBuffers(1, &m_index_buffer);

	assert(m_vertex_array);
	assert(m_vertex_buffer);
	assert(m_index_buffer);

	glBindVertexArray(m_vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, positions_byte_size + colors_byte_size + tex_coords_byte_size, nullptr, GL_STATIC_DRAW);
	//Only copy texture coordinate information via glBufferSubData. Others will be updated via cuda-gl interop.
	glBufferSubData(GL_ARRAY_BUFFER, positions_byte_size + colors_byte_size, tex_coords_byte_size, tex_coords.data());
	updateVertexBuffer();

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)(positions_byte_size));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (GLvoid*)(positions_byte_size + colors_byte_size));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_number_of_indices * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	auto shape_basis = loadModelData(morphable_model_directory + "/ShapeBasis_modified.matrix", true);
	m_shape_std_dev = loadModelData(morphable_model_directory + "/StandardDeviationShape.vec", false);
	m_shape_coefficients.resize(m_shape_std_dev.size(), 0.0f);
	m_shape_coefficients_normalized.resize(m_shape_std_dev.size());

	CHECK_CUDA_ERROR(cudaMalloc(&m_shape_basis_gpu, shape_basis.size() * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&m_shape_coefficients_gpu, m_shape_coefficients.size() * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(m_shape_basis_gpu, shape_basis.data(), shape_basis.size() * sizeof(float), cudaMemcpyHostToDevice));

	auto albedo_basis = loadModelData(morphable_model_directory + "/AlbedoBasis_modified.matrix", true);
	m_albedo_std_dev = loadModelData(morphable_model_directory + "/StandardDeviationAlbedo.vec", false);
	m_albedo_coefficients.resize(m_albedo_std_dev.size(), 0.0f);
	m_albedo_coefficients_normalized.resize(m_albedo_std_dev.size());

	CHECK_CUDA_ERROR(cudaMalloc(&m_albedo_basis_gpu, albedo_basis.size() * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&m_albedo_coefficients_gpu, m_albedo_coefficients.size() * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(m_albedo_basis_gpu, albedo_basis.data(), albedo_basis.size() * sizeof(float), cudaMemcpyHostToDevice));

	auto expression_basis = loadModelData(morphable_model_directory + "/ExpressionBasis_modified.matrix", true);
	m_expression_std_dev = loadModelData(morphable_model_directory + "/StandardDeviationExpression.vec", false);
	m_expression_coefficients.resize(m_expression_std_dev.size(), 0.0f);
	m_expression_coefficients_normalized.resize(m_expression_std_dev.size());

	CHECK_CUDA_ERROR(cudaMalloc(&m_expression_basis_gpu, expression_basis.size() * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&m_expression_coefficients_gpu, m_expression_coefficients.size() * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(m_expression_basis_gpu, expression_basis.data(), expression_basis.size() * sizeof(float), cudaMemcpyHostToDevice));

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
	if (m_average_face_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_average_face_gpu));
	}
	if (m_current_face_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_current_face_gpu));
	}
	if (m_shape_basis_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_shape_basis_gpu));
	}
	if (m_shape_coefficients_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_shape_coefficients_gpu));
	}
	if (m_albedo_basis_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_albedo_basis_gpu));
	}
	if (m_albedo_coefficients_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_albedo_coefficients_gpu));
	}
	if (m_expression_basis_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_expression_basis_gpu));
	}
	if (m_expression_coefficients_gpu)
	{
		CHECK_CUDA_ERROR(cudaFree(m_expression_coefficients_gpu));
	}
	cublasDestroy(m_cublas);
}

void Face::computeFace()
{
	int shape_number_of_coefficients = m_shape_coefficients.size();
	for (int i = 0; i < shape_number_of_coefficients; ++i)
	{
		m_shape_coefficients_normalized[i] = m_shape_coefficients[i] * m_shape_std_dev[i];
	}

	int albedo_number_of_coefficients = m_albedo_coefficients.size();
	for (int i = 0; i < albedo_number_of_coefficients; ++i)
	{
		m_albedo_coefficients_normalized[i] = m_albedo_coefficients[i] * m_albedo_std_dev[i];
	}

	int expression_number_of_coefficients = m_expression_coefficients.size();
	for (int i = 0; i < expression_number_of_coefficients; ++i)
	{
		m_expression_coefficients_normalized[i] = m_expression_coefficients[i] * m_expression_std_dev[i];
	}

	CHECK_CUDA_ERROR(cudaMemcpy(m_shape_coefficients_gpu, m_shape_coefficients_normalized.data(), shape_number_of_coefficients * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(m_albedo_coefficients_gpu, m_albedo_coefficients_normalized.data(), albedo_number_of_coefficients * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(m_expression_coefficients_gpu, m_expression_coefficients_normalized.data(), expression_number_of_coefficients * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(m_current_face_gpu, m_average_face_gpu, m_number_of_vertices * sizeof(glm::vec3) * 2, cudaMemcpyHostToDevice));

	float alpha = 1.0f;
	float beta = 1.0f;
	int m = 3 * m_number_of_vertices;
	int n = shape_number_of_coefficients;
	cublasSgemv(m_cublas, CUBLAS_OP_N, m, n, &alpha, m_shape_basis_gpu, m, m_shape_coefficients_gpu, 1, &beta, m_current_face_gpu, 1);

	n = albedo_number_of_coefficients;
	cublasSgemv(m_cublas, CUBLAS_OP_N, m, n, &alpha, m_albedo_basis_gpu, m, m_albedo_coefficients_gpu, 1, &beta, m_current_face_gpu + m, 1);

	n = expression_number_of_coefficients;
	cublasSgemv(m_cublas, CUBLAS_OP_N, m, n, &alpha, m_expression_basis_gpu, m, m_expression_coefficients_gpu, 1, &beta, m_current_face_gpu, 1);
}

void Face::updateVertexBuffer()
{
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&m_resource, m_vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));

	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &m_resource, 0));
	void* vertex_buffer_ptr;
	size_t size;
	CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&vertex_buffer_ptr, &size, m_resource));
	CHECK_CUDA_ERROR(cudaMemcpy(vertex_buffer_ptr, m_current_face_gpu, m_number_of_vertices * sizeof(glm::vec3) * 2, cudaMemcpyDeviceToDevice));

	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_resource, 0));
}

void Face::draw(const GLSLProgram& program) const
{
	program.use();

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
	file.read((char*)(basis.data()), size * sizeof(float));
	file.close();

	return basis;
}
