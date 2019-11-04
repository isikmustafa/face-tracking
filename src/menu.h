#pragma once

#include "window.h"
#include "face.h"

class Menu
{
public:
	explicit Menu(const std::shared_ptr<Window>&, const std::shared_ptr<Face>&);
	void initialize() const;
private:
	std::shared_ptr<Window> m_window;
	std::shared_ptr<Face> m_face;
};
