#pragma once

#include "window.h"
#include "face.h"

class Menu
{
public:
	Menu(const glm::ivec2& position, const glm::ivec2& size);

	void initializeWidgets(const std::shared_ptr<Face>& face);
	void attach(std::function<void()> func);
	void draw() const;

private:
	glm::ivec2 m_position;
	glm::ivec2 m_size;
	std::vector<std::function<void()>> m_funcs;
};
