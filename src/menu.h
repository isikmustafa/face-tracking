#pragma once

#include "window.h"
#include "face.h"

class Menu
{
public:
	explicit Menu(std::shared_ptr<Window>, std::shared_ptr<Face>);
	void initialize() const;
private:
	std::shared_ptr<Window> m_window;
	std::shared_ptr<Face> m_face;
};
