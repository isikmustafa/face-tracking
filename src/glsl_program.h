//https://github.com/isikmustafa/pathtracer/blob/master/gl/glsl_program.h

#pragma once

#include <glad/glad.h>

#include <string>
#include <vector>

class GLSLProgram
{
public:
	//Movable but non-copyable
	GLSLProgram() = default;
	GLSLProgram(GLSLProgram&) = delete;
	GLSLProgram(GLSLProgram&& rhs);
	GLSLProgram& operator=(GLSLProgram&) = delete;
	GLSLProgram& operator=(GLSLProgram&&);
	~GLSLProgram();

	void attachShader(GLenum shader_type, const std::string& shader_path);
	void link();
	void use() const;

	void setUniformIVar(const std::string& name, std::initializer_list<GLint> values) const;
	void setUniformFVar(const std::string& name, std::initializer_list<GLfloat> values) const;

private:
	std::vector<GLuint> m_shaders;
	GLuint m_program{ 0 };
};