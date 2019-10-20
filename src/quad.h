//https://github.com/isikmustafa/pathtracer/blob/master/gl/quad.h

#pragma once

#include <glad/glad.h>

class GLSLProgram;

class Quad
{
public:
	//Movable but non-copyable
	Quad() = default;
	Quad(Quad&) = delete;
	Quad(Quad&& rhs);
	Quad& operator=(Quad&) = delete;
	Quad& operator=(Quad&&);
	~Quad();

	void create();
	void draw(const GLSLProgram& program) const;

private:
	GLuint m_vertex_array{ 0 };
	GLuint m_vertex_buffer{ 0 };
};