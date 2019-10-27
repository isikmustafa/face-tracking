#include "window.h"
#include "face.h"
#include "device_util.h"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>
#include <GLFW/glfw3.h>

int main()
{
	constexpr int screen_width = 720;
	constexpr int screen_height = 480;
	Window window(screen_width, screen_height);

	GLSLProgram face_shader;
	face_shader.attachShader(GL_VERTEX_SHADER, "../src/shader/face.vert");
	face_shader.attachShader(GL_FRAGMENT_SHADER, "../src/shader/face.frag");
	face_shader.link();
	auto model = glm::scale(glm::mat4(1.0f), glm::vec3(10.0f));
	auto view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	auto projection = glm::perspective(45.0f, static_cast<float>(screen_width) / screen_height, 0.01f, 100.0f);

	Face face("../MorphableModel/averageMesh.off");

	//ImGui inits.
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window.getWindow(), true);
	ImGui_ImplOpenGL3_Init(nullptr);
	io.Fonts->AddFontDefault();

	glEnable(GL_DEPTH_TEST);
	while (!glfwWindowShouldClose(window.getWindow()))
	{
		glfwPollEvents();

		face_shader.use();
		face_shader.setMat4("model", model);
		face_shader.setMat4("view", view);
		face_shader.setMat4("projection", projection);
		face.draw(face_shader);

		//Start a new frame.
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		//Position and size of window.
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(240, screen_height), ImGuiCond_FirstUseEver);

		//Any application code here
		ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_MenuBar);
		ImGui::End();

		//Render.
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window.getWindow());
		glClear(GL_DEPTH_BUFFER_BIT);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	ImGui::DestroyContext();

	return 0;
}