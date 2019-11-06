#pragma once

#include "window.h"
#include "face.h"

class Menu
{
public:
	explicit Menu(const std::shared_ptr<Window>& window, const std::shared_ptr<Face>& face);

	void initialize();
	void attach(std::function<void()> func);
	void draw() const;

private:
	std::vector<std::function<void()>> m_funcs;
	std::shared_ptr<Window> m_window;
	std::shared_ptr<Face> m_face;
};
