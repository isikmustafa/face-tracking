//raytracer.mustafaisik.net//

#include "window.h"
#include "util.h"

#include <assert.h>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

Window::Window(int screen_width, int screen_height)
	: m_screen_width(screen_width)
	, m_screen_height(screen_height)
{
	//Initialize GLFW
	if (!glfwInit())
	{
		throw std::runtime_error("GLFW is not initialized properly");
	}

	//OpenGL 4.5
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	m_window = glfwCreateWindow(m_screen_width, m_screen_height, "Face2Face", nullptr, nullptr);
	if (!m_window)
	{
		throw std::runtime_error("GLFW could not create the window");
	}
	glfwMakeContextCurrent(m_window);

	//Initialize GLAD
	if (!gladLoadGL())
	{
		throw std::runtime_error("GLAD is not initialized properly");
	}

	//OpenGL settings
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glViewport(0, 0, m_screen_width, m_screen_height);

	//Compile the glsl program
	m_window_shader.attachShader(GL_VERTEX_SHADER, "../src/shader/window.vert");
	m_window_shader.attachShader(GL_FRAGMENT_SHADER, "../src/shader/window.frag");
	m_window_shader.link();

	//Create the quad into which we render the final image texture
	m_window_quad.create();

	//CUDA-GL Interop Initialization
	glGenTextures(1, &m_cuda_output_texture);
	assert(m_cuda_output_texture);

	glBindTexture(GL_TEXTURE_2D, m_cuda_output_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_screen_width, m_screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_resource, m_cuda_output_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
	glBindTexture(GL_TEXTURE_2D, 0);

	//Allocate CUDA array and create surface object.
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	CHECK_CUDA_ERROR(cudaMallocArray(&m_content_array, &channel_desc, m_screen_width, m_screen_height, cudaArraySurfaceLoadStore));

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	res_desc.res.array.array = m_content_array;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&m_content, &res_desc));
}

Window::~Window()
{
	CHECK_CUDA_ERROR(cudaDestroySurfaceObject(m_content));
	CHECK_CUDA_ERROR(cudaFreeArray(m_content_array));
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Window::renderWindow()
{
	cudaArray_t texture_ptr = nullptr;
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &m_resource, 0));
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_resource, 0, 0));
	CHECK_CUDA_ERROR(cudaMemcpyArrayToArray(texture_ptr, 0, 0, m_content_array, 0, 0, m_screen_width * m_screen_height * 4, cudaMemcpyDeviceToDevice));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_resource, 0));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_cuda_output_texture);
	m_window_shader.setUniformIVar("final_image", { 0 });
	m_window_quad.draw(m_window_shader);
	glBindTexture(GL_TEXTURE_2D, 0);
}

bool Window::queryKey(int key, int condition) const
{
	return glfwGetKey(m_window, key) == condition;
}

void Window::setWindowTitle(const std::string& title) const
{
	glfwSetWindowTitle(m_window, title.c_str());
}

void Window::getCursorPosition(double& x, double& y) const
{
	glfwGetCursorPos(m_window, &x, &y);
}