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

	ImGui::Separator();

	size_t free, total;
	CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
	ImGui::Text("Free  GPU Memory: %.1f MB", free / (1024.0f * 1024.0f));
	ImGui::Text("Total GPU Memory: %.1f MB", total / (1024.0f * 1024.0f));
	ImGui::End();

	//Render.
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void initializeMenuWidgets(Menu& menu, Face& face)
{
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
	menu.attach(std::move(shape_parameters_gui));

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
	menu.attach(std::move(albedo_parameters_gui));

	auto& expression_coefficients = face.getExpressionCoefficients();
	auto expression_parameters_gui = [&expression_coefficients]()
	{
		ImGui::CollapsingHeader("Expression Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Expression1", &expression_coefficients[0], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression2", &expression_coefficients[1], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression3", &expression_coefficients[2], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression4", &expression_coefficients[3], 0.0f, 1.0f);
		ImGui::SliderFloat("Expression5", &expression_coefficients[4], 0.0f, 1.0f);
	};
	menu.attach(std::move(expression_parameters_gui));

	auto& sh_coefficients = face.getSHCoefficients();
	auto sh_parameters_gui = [&sh_coefficients]()
	{
		ImGui::CollapsingHeader("Spherical Harmonics Parameters", ImGuiTreeNodeFlags_DefaultOpen);
		ImGui::SliderFloat("Ambient", &sh_coefficients[0], -1.0f, 1.0f);
		ImGui::SliderFloat("X", &sh_coefficients[1], -1.0f, 1.0f);
		ImGui::SliderFloat("Y", &sh_coefficients[2], -1.0f, 1.0f);
		ImGui::SliderFloat("Z", &sh_coefficients[3], -1.0f, 1.0f);
		ImGui::SliderFloat("xy", &sh_coefficients[4], -1.0f, 1.0f);
		ImGui::SliderFloat("yz", &sh_coefficients[5], -1.0f, 1.0f);
		ImGui::SliderFloat("-x2-y2+2z2", &sh_coefficients[6], -1.0f, 1.0f);
		ImGui::SliderFloat("zx", &sh_coefficients[7], -1.0f, 1.0f);
		ImGui::SliderFloat("x2-y2", &sh_coefficients[8], -1.0f, 1.0f);
	};
	menu.attach(std::move(sh_parameters_gui));
}