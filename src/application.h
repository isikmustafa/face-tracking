#pragma once

#include "window.h"
#include "face.h"
#include "tracker.h"
#include "menu.h"
#include "gauss_newton_solver.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

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
	GLSLProgram m_fullscreen_shader;
	glm::mat4 m_projection;
	double m_frame_time{ 0.0 };

	GLuint m_face_framebuffer; 
	GLuint m_rt_rgb; 
	GLuint m_rt_barycentrics; 
	GLuint m_rt_vertex_ids; 
	GLuint m_depth_buffer; 
	GLuint m_empty_vao; 
	GLuint m_camera_frame_texture; 

private:
	void initGraphics();
	void initMenuWidgets();
	void reloadShaders();
	void draw(cv::Mat& frame);
};