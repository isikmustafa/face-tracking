#pragma once

#include "glsl_program.h"

#include <string>
#include <vector>
#include <functional>

struct GLFWwindow;
struct cudaGraphicsResource;

class Window
{
public:
	Window();
	Window(const Window&) = delete;
	Window(Window&&) = delete;
	Window& operator=(const Window&) = delete;
	Window& operator=(Window&&) = delete;
	~Window();

	void attachToGui(std::function<void()> func);
	void drawGui();
	void refresh();
	bool queryKey(int key, int condition) const;
	void setWindowTitle(const std::string& title) const;
	void getCursorPosition(double& x, double& y) const;

	GLFWwindow* getWindow() const { return m_window; }

	const static int m_screen_width = 1440;
	const static int m_screen_height = 900;
	const static int m_gui_width= 240;

private:
	//Regular window variables
	GLFWwindow* m_window{ nullptr };
	std::vector<std::function<void()>> m_funcs;
};