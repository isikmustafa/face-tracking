#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <string>
#include <cuda_gl_interop.h>

class GLSLProgram;

class Face
{
public:
	//Movable but non-copyable
	Face(const std::string& filename);
	Face(Face&) = delete;
	Face(Face&& rhs);
	Face& operator=(Face&) = delete;
	Face& operator=(Face&&);
	~Face();

	//Copies m_face_data_gpu to content of m_vertex_buffer.
	void updateVertexBuffer();
	void draw(const GLSLProgram& program) const;

private:
	GLuint m_vertex_array{ 0 };
	GLuint m_vertex_buffer{ 0 };
	GLuint m_index_buffer{ 0 };
	unsigned int m_number_of_indices{ 0 };
	unsigned int m_number_of_vertices{ 0 };

	//CUDA-GL interop for buffer object.
	cudaGraphicsResource* m_resource{ nullptr };
	float* m_face_data_gpu{ nullptr };
	int m_face_data_size{ 0 };
};