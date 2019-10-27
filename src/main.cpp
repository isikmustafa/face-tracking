#include "window.h"
#include "face.h"
#include "device_util.h"

#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

int main()
{
	constexpr int gui_width = 240;
	constexpr int screen_width = 720;
	constexpr int screen_height = 480;
	Window window(gui_width, screen_width, screen_height);

	GLSLProgram face_shader;
	face_shader.attachShader(GL_VERTEX_SHADER, "../src/shader/face.vert");
	face_shader.attachShader(GL_FRAGMENT_SHADER, "../src/shader/face.frag");
	face_shader.link();
	auto model = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f));
	auto view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	auto projection = glm::perspective(45.0f, static_cast<float>(screen_width) / screen_height, 0.01f, 100.0f);

	Face face("../MorphableModel/averageMesh.off");

	while (!glfwWindowShouldClose(window.getWindow()))
	{
		glfwPollEvents();

		face_shader.use();
		face_shader.setMat4("model", model);
		face_shader.setMat4("view", view);
		face_shader.setMat4("projection", projection);
		face.draw(face_shader);

		window.drawGui();
		window.refresh();
	}

	return 0;
}