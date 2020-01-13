#include "application.h"

#include <imgui.h>
#include <glm/gtx/euler_angles.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <utility>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <chrono>

constexpr int kScreenWidth = 1200;
constexpr int kScreenHeight = 900;
//constexpr int kScreenWidth = 640;
//constexpr int kScreenHeight = 360;
constexpr int kTextureWidth = 640;
constexpr int kTextureHeight = 360;
constexpr glm::ivec2 kGuiPosition(0, 0);
constexpr glm::ivec2 kGuiSize(240, kScreenHeight);

static std::string kMorphableModelPath("../MorphableModel/");

Application::Application()
	: m_window(kGuiSize.x, kScreenWidth, kScreenHeight)
	, m_face(kMorphableModelPath)
	, m_solver()
	, m_tracker()
	, m_menu(kGuiPosition, kGuiSize)
	, m_projection(glm::perspectiveRH_NO(glm::radians(60.0f), static_cast<float>(kScreenWidth) / kScreenHeight, 0.01f, 10.0f))
{
	//m_camera = cv::VideoCapture(0);
	m_camera = cv::VideoCapture("./demo.mp4");
}

Application::~Application()
{
	if (m_rt_rgb_cuda_resource)
	{
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(m_rt_rgb_cuda_resource));
	}
	if (m_rt_barycentrics_cuda_resource)
	{
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(m_rt_barycentrics_cuda_resource));
	}
	if (m_rt_vertex_ids_cuda_resource)
	{
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(m_rt_vertex_ids_cuda_resource));
	}

	// TODO: Also destroy other OpenGL related stuff here.
}

void Application::run()
{
	initGraphics();
	initMenuWidgets();
	reloadShaders();

	while (!glfwWindowShouldClose(m_window.getGLFWWindow()))
	{
		auto start_frame = std::chrono::high_resolution_clock::now();

		glfwPollEvents();

		if (glfwGetKey(m_window.getGLFWWindow(), GLFW_KEY_F5) == GLFW_PRESS)
		{
			std::cout << "reload shaders" << std::endl;
			reloadShaders();
		}

		cv::Mat raw_frame;
		if (!m_camera.read(raw_frame))
		{
			continue;
		}
		cv::Mat frame;
		cv::pyrDown(raw_frame, frame);

		auto sparse_features = m_tracker.getSparseFeatures(frame);
		m_solver.solve(sparse_features, m_face, raw_frame, m_projection);
		//m_solver.solve_CPU(sparse_features, m_face, m_projection);


		m_face.computeFace();
		m_face.updateVertexBuffer();
		m_face.draw();

		draw(raw_frame);
		m_menu.draw();
		m_window.refresh();

		auto end_frame = std::chrono::high_resolution_clock::now();
		m_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(end_frame - start_frame).count() / 1000.0;
	}
}

void Application::initMenuWidgets()
{
	auto& shape_coefficients = m_face.getShapeCoefficients();
	auto shape_parameters_gui = [&shape_coefficients]()
	{
		ImGui::CollapsingHeader("Shape Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Shape1", &shape_coefficients[0], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape2", &shape_coefficients[1], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape3", &shape_coefficients[2], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape4", &shape_coefficients[3], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape5", &shape_coefficients[4], -5.0f, 5.0f);
	};
	m_menu.attach(std::move(shape_parameters_gui));

	auto& albedo_coefficients = m_face.getAlbedoCoefficients();
	auto albedo_parameters_gui = [&albedo_coefficients]()
	{
		ImGui::CollapsingHeader("Albedo Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Albedo1", &albedo_coefficients[0], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo2", &albedo_coefficients[1], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo3", &albedo_coefficients[2], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo4", &albedo_coefficients[3], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo5", &albedo_coefficients[4], -5.0f, 5.0f);
	};
	m_menu.attach(std::move(albedo_parameters_gui));

	auto& expression_coefficients = m_face.getExpressionCoefficients();
	auto expression_parameters_gui = [&expression_coefficients]()
	{
		ImGui::CollapsingHeader("Expression Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Expression1", &expression_coefficients[0], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression2", &expression_coefficients[1], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression3", &expression_coefficients[2], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression4", &expression_coefficients[3], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression5", &expression_coefficients[4], 0.0f, 1.0f);
	};
	m_menu.attach(std::move(expression_parameters_gui));

	auto& sh_coefficients = m_face.getSHCoefficients();
	auto sh_parameters_gui = [&sh_coefficients]()
	{
		ImGui::CollapsingHeader("Spherical Harmonics Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Ambient", &sh_coefficients[0], -1.0f, 1.0f);
		ImGui::SliderFloat("y", &sh_coefficients[1], -1.0f, 1.0f);
		ImGui::SliderFloat("z", &sh_coefficients[2], -1.0f, 1.0f);
		ImGui::SliderFloat("x", &sh_coefficients[3], -1.0f, 1.0f);
		ImGui::SliderFloat("xy", &sh_coefficients[4], -1.0f, 1.0f);
		ImGui::SliderFloat("yz", &sh_coefficients[5], -1.0f, 1.0f);
		ImGui::SliderFloat("3z2 - 1", &sh_coefficients[6], -1.0f, 1.0f);
		ImGui::SliderFloat("xz", &sh_coefficients[7], -1.0f, 1.0f);
		ImGui::SliderFloat("x2-y2", &sh_coefficients[8], -1.0f, 1.0f);
	};
	m_menu.attach(std::move(sh_parameters_gui));

	auto& solver_parameters = m_solver.getSolverParameters();
	auto opt_parameters = [&solver_parameters]()
	{
		ImGui::CollapsingHeader("Optimisation Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Sparse Weight exp", &solver_parameters.sparse_weight_exponent, -4.0f, 4.0f);
		ImGui::SliderFloat("Dense Weight exp", &solver_parameters.dense_weight_exponent, -4.0f, 4.0f);
		ImGui::SliderFloat("Regularizer exp", &solver_parameters.regularisation_weight_exponent, -8.0f, 4.0f);

		ImGui::SliderInt("Gauss Newton iterations", &solver_parameters.num_gn_iterations, 1, 15);
		ImGui::SliderInt("PCG iterations", &solver_parameters.num_pcg_iterations, 1, 500);

		ImGui::SliderInt("Shape Parameters", &solver_parameters.num_shape_coefficients, 0, 160);
		ImGui::SliderInt("Albedo Parameters", &solver_parameters.num_albedo_coefficients, 0, 160);
		ImGui::SliderInt("SH Parameters", &solver_parameters.num_sh_coefficients, 0, 9);

		ImGui::SliderInt("Expression Parameters", &solver_parameters.num_expression_coefficients, 0, 76);
	};
	m_menu.attach(std::move(opt_parameters));

	auto gpu_memory_info_gui = [this]()
	{
		ImGui::Separator();
		ImGui::Text("Frame Time: %.1f ms", m_frame_time);

		size_t free, total;
		CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));

		ImGui::Text("Free  GPU Memory: %.1f MB", free / (1024.0f * 1024.0f));
		ImGui::Text("Total GPU Memory: %.1f MB", total / (1024.0f * 1024.0f));

		ImGui::End();
	};
	m_menu.attach(std::move(gpu_memory_info_gui));
}

void Application::initGraphics()
{
	glGenFramebuffers(1, &m_face_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_face_framebuffer);

	// RGB render texture
	glGenTextures(1, &m_rt_rgb);
	glBindTexture(GL_TEXTURE_2D, m_rt_rgb);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kTextureWidth, kTextureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_rt_rgb, 0);
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_rt_rgb_cuda_resource, m_rt_rgb, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

	// barycentrics render texture
	glGenTextures(1, &m_rt_barycentrics);
	glBindTexture(GL_TEXTURE_2D, m_rt_barycentrics);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, kTextureWidth, kTextureHeight, 0, GL_RGBA, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, m_rt_barycentrics, 0);
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_rt_barycentrics_cuda_resource, m_rt_barycentrics, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

	// vertex ID render texture
	glGenTextures(1, &m_rt_vertex_ids);
	glBindTexture(GL_TEXTURE_2D, m_rt_vertex_ids);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32I, kTextureWidth, kTextureHeight, 0, GL_RGBA_INTEGER, GL_INT, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, m_rt_vertex_ids, 0);
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_rt_vertex_ids_cuda_resource, m_rt_vertex_ids, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

	GLenum draw_buffers[3] = { GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, draw_buffers); // "3" is the size of draw_buffers
	glGenRenderbuffers(1, &m_depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, m_depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, kTextureWidth, kTextureHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_buffer);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::runtime_error("Error: Failed to create the framebuffer!");
	}

	// empty vertex buffer used to draw fullscreen quad
	glGenVertexArrays(1, &m_empty_vao);
	// texture we upload the camera input to
	glGenTextures(1, &m_camera_frame_texture);
	glBindTexture(GL_TEXTURE_2D, m_camera_frame_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void Application::reloadShaders()
{
	m_face_shader = GLSLProgram();
	m_face_shader.attachShader(GL_VERTEX_SHADER, "../src/shader/face.vert");
	m_face_shader.attachShader(GL_GEOMETRY_SHADER, "../src/shader/face.geom");
	m_face_shader.attachShader(GL_FRAGMENT_SHADER, "../src/shader/face.frag");
	m_face_shader.link();

	m_face_shader.use();
	m_face_shader.setMat4("projection", m_projection);

	m_fullscreen_shader = GLSLProgram();
	m_fullscreen_shader.attachShader(GL_VERTEX_SHADER, "../src/shader/quad.vert");
	m_fullscreen_shader.attachShader(GL_FRAGMENT_SHADER, "../src/shader/quad.frag");
	m_fullscreen_shader.link();

	glFinish();

	auto& graphics_settings = m_face.getGraphicsSettings();
	graphics_settings.framebuffer = m_face_framebuffer;
	graphics_settings.rt_rgb_cuda_resource = m_rt_rgb_cuda_resource;
	graphics_settings.rt_barycentrics_cuda_resource = m_rt_barycentrics_cuda_resource;
	graphics_settings.rt_vertex_ids_cuda_resource = m_rt_vertex_ids_cuda_resource;
	graphics_settings.shader = &m_face_shader;
	graphics_settings.screen_width = kScreenWidth;
	graphics_settings.screen_height = kScreenHeight;
	graphics_settings.texture_width = kTextureWidth; 
	graphics_settings.texture_height = kTextureHeight; 
	graphics_settings.mapped_to_cuda = false;
}

void Application::draw(cv::Mat& frame)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(kGuiSize.x, 0, kScreenWidth, kScreenHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindVertexArray(m_empty_vao);
	glDisable(GL_DEPTH_TEST);

	m_fullscreen_shader.use();
	glActiveTexture(GL_TEXTURE0);
	m_fullscreen_shader.setUniformIVar("face", { 0 });
	glBindTexture(GL_TEXTURE_2D, m_rt_rgb);

	cv::Mat processed_frame;
	cv::resize(frame, processed_frame, cv::Size(kScreenWidth, kScreenHeight));
	cv::flip(processed_frame, processed_frame, 0);

	glActiveTexture(GL_TEXTURE1);
	m_fullscreen_shader.setUniformIVar("background", { 1 });
	glBindTexture(GL_TEXTURE_2D, m_camera_frame_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, kScreenWidth, kScreenHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, processed_frame.data);

	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glEnable(GL_DEPTH_TEST);
}
