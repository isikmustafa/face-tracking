#include "menu.h"

#include <utility>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>

Menu::Menu(const glm::ivec2& position, const glm::ivec2& size)
	: m_position(position)
	, m_size(size)
{}

void Menu::attach(std::function<void()> func)
{
	m_funcs.push_back(std::move(func));
}

void Menu::draw() const
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//Start a new frame.
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	//Position and size of window.
	ImGui::SetNextWindowPos(ImVec2(m_position.x, m_position.y), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(m_size.x, m_size.y), ImGuiCond_FirstUseEver);

	//Any application code here
	ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_MenuBar);

	for (auto& func : m_funcs)
	{
		func();
	}

	ImGui::End();

	//Render.
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}