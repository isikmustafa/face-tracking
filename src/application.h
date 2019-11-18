#pragma once

#include "window.h"
#include "face.h"
#include "tracker.h"
#include "menu.h"
#include "gauss_newton_solver.h"

#include <GLFW/glfw3.h>

class Application
{
public:
	Application();

	void run();

private:
	Window m_window;
	Face m_face;
	GaussNewtonSolver m_solver;
	Tracker m_tracker;
	Menu m_menu;
	cv::VideoCapture m_camera;
	GLSLProgram m_face_shader;

private:
	void initMenuWidgets();
	void initFaceShader();
	void drawFace();
};