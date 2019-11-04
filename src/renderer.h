#pragma once

#include "glsl_program.h"
#include "window.h"
#include "face.h"

class Renderer
{
public:
	Renderer(std::shared_ptr<Window>, std::shared_ptr<Face>);
	void start() const;

private:
	GLSLProgram m_face_shader;
	std::shared_ptr<Window> m_window;
	std::shared_ptr<Face> m_face;
};
