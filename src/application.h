#pragma once

#include "window.h"
#include "face.h"
#include "tracker.h"
#include "menu.h"
#include "gauss_newton_solver.h"
#include "quad.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Application
{
public:
	Application();
	Application(Application&) = delete;
	Application(Application&& rhs) = delete;
	Application& operator=(Application&) = delete;
	Application& operator=(Application&&) = delete;
	~Application();

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

	GLuint m_face_framebuffer{ 0 };
	GLuint m_rt_rgb{ 0 };
	GLuint m_rt_barycentrics{ 0 };
	GLuint m_rt_vertex_ids{ 0 };
	cudaGraphicsResource_t m_rt_rgb_cuda_resource{ nullptr };
	cudaGraphicsResource_t m_rt_barycentrics_cuda_resource{ nullptr };
	cudaGraphicsResource_t m_rt_vertex_ids_cuda_resource{ nullptr };
	GLuint m_depth_buffer{ 0 };
	GLuint m_empty_vao{ 0 };
	GLuint m_camera_frame_texture{ 0 };

private:
	void initGraphics();
	void initMenuWidgets();
	void reloadShaders();
	void draw(cv::Mat& frame);
};