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
	Window(int gui_width, int screen_width, int screen_height);
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

private:
	//Regular window variables
	GLFWwindow* m_window{ nullptr };
	int m_screen_width;
	int m_screen_height;
	int m_gui_width;
	std::vector<std::function<void()>> m_funcs;
};