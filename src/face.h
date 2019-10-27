#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <string>

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

	void draw(const GLSLProgram& program) const;

private:
	GLuint m_vertex_array{ 0 };
	GLuint m_vertex_buffer{ 0 };
	GLuint m_index_buffer{ 0 };
	unsigned int m_number_of_indices{ 0 };
};