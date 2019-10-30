#include "window.h"
#include "face.h"
#include "device_util.h"
#include "tracker.h"

#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>

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

	//const auto tracker = std::make_unique<Tracker>();
	//tracker->start();

	Face face("../MorphableModel/");

	auto& shape_coefficients = face.getShapeCoefficients();
	auto shape_parameters_gui = [&shape_coefficients]()
	{
		ImGui::CollapsingHeader("Shape Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Shape1", &shape_coefficients[0], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape2", &shape_coefficients[1], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape3", &shape_coefficients[2], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape4", &shape_coefficients[3], -5.0f, 5.0f);
		ImGui::SliderFloat("Shape5", &shape_coefficients[4], -5.0f, 5.0f);
	};
	window.attachToGui(std::move(shape_parameters_gui));

	auto& albedo_coefficients = face.getAlbedoCoefficients();
	auto albedo_parameters_gui = [&albedo_coefficients]()
	{
		ImGui::CollapsingHeader("Albedo Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Albedo1", &albedo_coefficients[0], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo2", &albedo_coefficients[1], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo3", &albedo_coefficients[2], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo4", &albedo_coefficients[3], -5.0f, 5.0f);
		ImGui::SliderFloat("Albedo5", &albedo_coefficients[4], -5.0f, 5.0f);
	};
	window.attachToGui(std::move(albedo_parameters_gui));

	auto& expression_coefficients = face.getExpressionCoefficients();
	auto expression_parameters_gui = [&expression_coefficients]()
	{
		ImGui::CollapsingHeader("Expression Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Expression1", &expression_coefficients[0], -5.0f, 5.0f);
		ImGui::SliderFloat("Expression2", &expression_coefficients[1], -5.0f, 5.0f);
		ImGui::SliderFloat("Expression3", &expression_coefficients[2], -5.0f, 5.0f);
		ImGui::SliderFloat("Expression4", &expression_coefficients[3], -5.0f, 5.0f);
		ImGui::SliderFloat("Expression5", &expression_coefficients[4], -5.0f, 5.0f);
	};
	window.attachToGui(std::move(expression_parameters_gui));

	while (!glfwWindowShouldClose(window.getWindow()))
	{
		glfwPollEvents();

		face_shader.use();
		face_shader.setMat4("model", model);
		face_shader.setMat4("view", view);
		face_shader.setMat4("projection", projection);

		face.computeFace();
		face.updateVertexBuffer();
		face.draw(face_shader);

		window.drawGui();
		window.refresh();
	}

	return 0;
}