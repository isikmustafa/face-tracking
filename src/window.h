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
	//Non-movable and non-copyable
	Window(int gui_width, int screen_width, int screen_height);
	Window(const Window&) = delete;
	Window(Window&&) = delete;
	Window& operator=(const Window&) = delete;
	Window& operator=(Window&&) = delete;
	~Window();

	void refresh();
	bool queryKey(int key, int condition) const;
	void setWindowTitle(const std::string& title) const;
	void getCursorPosition(double& x, double& y) const;

	GLFWwindow* getGLFWWindow() const { return m_window; }
	int getScreenWidth() const { return m_screen_width; }
	int getScreenHeight() const { return m_screen_height; }
	int getGuiWidth() const { return m_gui_width; }

private:
	//Regular window variables
	GLFWwindow* m_window{ nullptr };
	const int m_screen_width;
	const int m_screen_height;
	const int m_gui_width;
};