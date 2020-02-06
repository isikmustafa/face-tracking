#pragma once

#include "window.h"
#include "face.h"
#include "tracker.h"
#include "menu.h"
#include "gauss_newton_solver.h"
#include "pyramid.h"

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

	void run();

private:
	cv::VideoCapture m_camera;
	int m_screen_width;
	int m_screen_height;
	glm::ivec2 m_gui_position;
	glm::ivec2 m_gui_size;
	GLSLProgram m_fullscreen_shader;
	glm::mat4 m_projection;
	Window m_window;
	Face m_face;
	GaussNewtonSolver m_solver;
	Tracker m_tracker;
	Menu m_menu;
	GLSLProgram m_face_shader;
	double m_frame_time{ 0.0 };
	Pyramid m_pyramid;
	GLuint m_empty_vao{ 0 };
	GLuint m_camera_frame_texture{ 0 };
	GLuint m_video_framebuffer;
	GLuint m_video_texture;
	cudaGraphicsResource_t m_video_texture_resource;
	GLSLProgram m_video_shader;
	int m_video_width;
	int m_video_height;
	cv::VideoWriter m_video_writer;

private:
	void initGraphics();
	void initMenuWidgets();
	void reloadShaders();
	void draw(cv::Mat& frame);
	void saveVideoFrame(cv::Mat& frame, std::vector<glm::vec2>& features = std::vector<glm::vec2>());

	//If you are calling this function. Make sure m_face uses "final.off".
	void printUniqueFaceVerticesSparse();
};
