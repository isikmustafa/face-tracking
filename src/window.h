#pragma once

#include "glsl_program.h"
#include "quad.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include <string>

struct GLFWwindow;
struct cudaGraphicsResource;

class Window
{
public:
	Window(int screen_width, int screen_height);
	Window(const Window&) = delete;
	Window(Window&&) = delete;
	Window& operator=(const Window&) = delete;
	Window& operator=(Window&&) = delete;
	~Window();

	void renderWindow();
	bool queryKey(int key, int condition) const;
	void setWindowTitle(const std::string& title) const;
	void getCursorPosition(double& x, double& y) const;

	//This surface object can used in CUDA kernels and be written into.
	cudaSurfaceObject_t getContent() const { return m_content; }
	GLFWwindow* getWindow() const { return m_window; }

private:
	//CUDA-GL Interop
	GLSLProgram m_window_shader;
	Quad m_window_quad;
	cudaGraphicsResource* m_resource{ nullptr };
	GLuint m_cuda_output_texture{ 0 };
	cudaSurfaceObject_t m_content{ 0 };
	cudaArray_t m_content_array{ nullptr };

	//Regular window variables
	GLFWwindow* m_window{ nullptr };
	int m_screen_width, m_screen_height;
};