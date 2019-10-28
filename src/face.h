#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <string>
#include <vector>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>

class GLSLProgram;

class Face
{
public:
	//Non-movable and non-copyable
	Face(const std::string& morphable_model_directory);
	Face(Face&) = delete;
	Face(Face&& rhs) = delete;
	Face& operator=(Face&) = delete;
	Face& operator=(Face&&) = delete;
	~Face();

	void computeFace();
	//Copies m_average_face_gpu to content of m_vertex_buffer.
	void updateVertexBuffer();
	void draw(const GLSLProgram& program) const;

	std::vector<float>& getShapeCoefficients() { return m_shape_coefficients; }
	const std::vector<float>& getShapeCoefficients() const { return m_shape_coefficients; }
	std::vector<float>& getAlbedoCoefficients() { return m_albedo_coefficients; }
	const std::vector<float>& getAlbedoCoefficients() const { return m_albedo_coefficients; }
	std::vector<float>& getExpressionCoefficients() { return m_expression_coefficients; }
	const std::vector<float>& getExpressionCoefficients() const { return m_expression_coefficients; }

private:
	GLuint m_vertex_array{ 0 };
	GLuint m_vertex_buffer{ 0 };
	GLuint m_index_buffer{ 0 };
	unsigned int m_number_of_vertices{ 0 };
	unsigned int m_number_of_indices{ 0 };

	//CUDA-GL interop for buffer object.
	cudaGraphicsResource* m_resource{ nullptr };
	float* m_average_face_gpu{ nullptr };
	float* m_current_face_gpu{ nullptr };
	int m_face_data_size{ 0 };

	//cuBLAS
	cublasHandle_t m_cublas;

	//Shape basis and standard deviation.
	std::vector<float> m_shape_std_dev;
	std::vector<float> m_shape_coefficients;
	std::vector<float> m_shape_coefficients_normalized;
	float* m_shape_basis_gpu{ nullptr };
	float* m_shape_coefficients_gpu{ nullptr };

	//Albedo basis and standard deviation.
	std::vector<float> m_albedo_std_dev;
	std::vector<float> m_albedo_coefficients;
	std::vector<float> m_albedo_coefficients_normalized;
	float* m_albedo_basis_gpu{ nullptr };
	float* m_albedo_coefficients_gpu{ nullptr };

	//Expression basis and standard deviation.
	std::vector<float> m_expression_std_dev;
	std::vector<float> m_expression_coefficients;
	std::vector<float> m_expression_coefficients_normalized;
	float* m_expression_basis_gpu{ nullptr };
	float* m_expression_coefficients_gpu{ nullptr };

private:
	std::vector<float> loadModelData(const std::string& filename, bool is_basis);
};