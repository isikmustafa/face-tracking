#pragma once

#include "glsl_program.h"
#include "window.h"
#include "face.h"

class Renderer
{
public:
	Renderer(const std::shared_ptr<Face>&);
	void drawFace() const;

private:
	GLSLProgram m_face_shader;
	std::shared_ptr<Face> m_face;
};
