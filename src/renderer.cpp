#include "renderer.h"

#include <glad/glad.h>
#include <glm/ext/matrix_clip_space.inl>
#include <glm/ext/matrix_transform.inl>
#include <utility>
#include <GLFW/glfw3.h>

Renderer::Renderer(std::shared_ptr<Window> window, std::shared_ptr<Face> face)
	: m_window(std::move(window)), m_face(std::move(face)) {

	m_face_shader.attachShader(GL_VERTEX_SHADER, "../src/shader/face.vert");
	m_face_shader.attachShader(GL_FRAGMENT_SHADER, "../src/shader/face.frag");
	m_face_shader.link();
}

void Renderer::start() const {
	const auto model = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f));
	const auto view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	const auto projection = glm::perspective(45.0f, static_cast<float>(Window::m_screen_width) / Window::m_screen_height, 0.01f, 100.0f);

	while (!glfwWindowShouldClose(m_window->getWindow()))
	{
		glfwPollEvents();

		m_face_shader.use();
		m_face_shader.setMat4("model", model);
		m_face_shader.setMat4("view", view);
		m_face_shader.setMat4("projection", projection);
		m_face_shader.setUniformFVVar("sh_coefficients", m_face->getSHCoefficients());

		m_face->computeFace();
		m_face->updateVertexBuffer();
		m_face->draw(m_face_shader);

		m_window->drawGui();
		m_window->refresh();
	}
}
