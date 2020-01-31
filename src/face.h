#pragma once

#include "device_array.h"

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <string>
#include <vector>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

class GLSLProgram;

class Face
{
public:
	struct GraphicsSettings
	{
		GLuint framebuffer{ 0 };
		GLuint rt_rgb{ 0 };
		cudaGraphicsResource_t rt_rgb_cuda_resource{ nullptr };
		cudaGraphicsResource_t rt_barycentrics_cuda_resource{ nullptr };
		cudaGraphicsResource_t rt_vertex_ids_cuda_resource{ nullptr };
		const GLSLProgram* shader{ nullptr };
		int texture_width{ 0 };
		int texture_height{ 0 };
		bool mapped_to_cuda{ false };
	};

public:
	//Non-movable and non-copyable
	Face(const std::string& morphable_model_directory);
	Face(Face&) = delete;
	Face(Face&& rhs) = delete;
	Face& operator=(Face&) = delete;
	Face& operator=(Face&&) = delete;
	~Face();

	void computeFace();
	void computeNormals();
	glm::mat4 computeModelMatrix() const;
	void computeRotationDerivatives(glm::mat3& dRx, glm::mat3& dRy, glm::mat3& dRz) const;

	//Copies m_average_face_gpu to content of m_vertex_buffer.
	void updateVertexBuffer();
	void draw() const;

	std::vector<float>& getShapeCoefficients() { return m_shape_coefficients; }
	const std::vector<float>& getShapeCoefficients() const { return m_shape_coefficients; }
	std::vector<float>& getAlbedoCoefficients() { return m_albedo_coefficients; }
	const std::vector<float>& getAlbedoCoefficients() const { return m_albedo_coefficients; }
	std::vector<float>& getExpressionCoefficients() { return m_expression_coefficients; }
	const std::vector<float>& getExpressionCoefficients() const { return m_expression_coefficients; }

	std::vector<float>& getSHCoefficients() { return m_sh_coefficients; }
	const std::vector<float>& getSHCoefficients() const { return m_sh_coefficients; }

	Face::GraphicsSettings& getGraphicsSettings() { return m_graphics_settings; }
	const Face::GraphicsSettings& getGraphicsSettings() const { return m_graphics_settings; }

	glm::vec3& getRotationCoefficients() { return m_rotation_coefficients; }
	const glm::vec3& getRotationCoefficients() const { return m_rotation_coefficients; }
	glm::vec3& getTranslationCoefficients() { return m_translation_coefficients; }
	const glm::vec3& getTranslationCoefficients() const { return m_translation_coefficients; }

	unsigned int getNumberOfVertices() const { return m_number_of_vertices; }

private:
	friend class GaussNewtonSolver;
	friend class Application; 
private:
	GraphicsSettings m_graphics_settings;

	GLuint m_vertex_array{ 0 };
	GLuint m_vertex_buffer{ 0 };
	GLuint m_index_buffer{ 0 };
	unsigned int m_number_of_vertices{ 0 };
	unsigned int m_number_of_indices{ 0 };
	cudaGraphicsResource* m_resource{ nullptr };

	std::vector<float> m_shape_basis;
	std::vector<float> m_albedo_basis;
	std::vector<float> m_expression_basis;

	//Face vertex and color data.
	util::DeviceArray<glm::vec3> m_average_face_gpu;
	util::DeviceArray<glm::vec3> m_current_face_gpu;
	util::DeviceArray<glm::ivec3> m_faces_gpu;

	//cuBLAS
	cublasHandle_t m_cublas;

	//Shape basis and standard deviation.
	std::vector<float> m_shape_coefficients;
	util::DeviceArray<float> m_shape_basis_gpu;
	util::DeviceArray<float> m_shape_coefficients_gpu;

	//Albedo basis and standard deviation.
	std::vector<float> m_albedo_coefficients;
	util::DeviceArray<float> m_albedo_basis_gpu;
	util::DeviceArray<float> m_albedo_coefficients_gpu;

	//Expression basis and standard deviation.
	std::vector<float> m_expression_coefficients;
	util::DeviceArray<float> m_expression_basis_gpu;
	util::DeviceArray<float> m_expression_coefficients_gpu;

	//SH parameters
	std::vector<float> m_sh_coefficients;

	//Pose parameters
	glm::vec3 m_rotation_coefficients;
	glm::vec3 m_translation_coefficients;

private:
	std::vector<float> loadModelData(const std::string& filename, bool is_basis);
};