#include "window.h"
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>
#include "device_util.h"
#include <GLFW/glfw3.h>

__global__ void dummyKernel(cudaSurfaceObject_t window_surface_content, float3 rgb)
{
	auto index = util::getThreadIndex2D();

	surf2Dwrite(util::rgbToUint(rgb), window_surface_content, index.x * 4, index.y);
}

int main()
{
	constexpr int screen_width = 720;
	constexpr int screen_height = 480;

	Window window(screen_width, screen_height);
	float3 screen_color{ 128.0f, 128.0f , 128.0f };

	//ImGui inits.
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window.getWindow(), true);
	ImGui_ImplOpenGL3_Init(nullptr);
	io.Fonts->AddFontDefault();

	while (!glfwWindowShouldClose(window.getWindow()))
	{
		glfwPollEvents();

		dim3 threads(16, 16);
		dim3 blocks(screen_width / threads.x, screen_height / threads.y);
		dummyKernel << <blocks, threads >> > (window.getContent(), screen_color);

		window.renderWindow();

		//Start a new frame.
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		//Position and size of window.
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(240, screen_height), ImGuiCond_FirstUseEver);

		//Any application code here
		ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_MenuBar);
		ImGui::SliderFloat("Red", &screen_color.x, 0.0f, 255.0f, "%.1f");
		ImGui::SliderFloat("Green", &screen_color.y, 0.0f, 255.0f, "%.1f");
		ImGui::SliderFloat("Blue", &screen_color.z, 0.0f, 255.0f, "%.1f");
		ImGui::End();

		//Render.
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window.getWindow());
		glClear(GL_COLOR_BUFFER_BIT);
	}

	ImGui::DestroyContext();

	return 0;
}