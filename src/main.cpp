#include "window.h"
#include "face.h"
#include "tracker.h"
#include "renderer.h"
#include "menu.h"
#include "gauss_newton.h"

#include <GLFW/glfw3.h>

int main()
{
	const std::string path = "../MorphableModel/";

	// Composition root
	const auto window = std::make_shared<Window>();
	const auto face = std::make_shared<Face>(path);
	const auto solver = std::make_shared<GaussNewton>(face);

	const auto tracker = std::make_unique<Tracker>();
	const auto menu = std::make_unique<Menu>(glm::ivec2(0, 0), glm::ivec2(Window::m_gui_width, Window::m_screen_height));
	const auto renderer = std::make_unique<Renderer>(face);

	menu->initializeWidgets(face);

	cv::VideoCapture camera(0);

	while (!glfwWindowShouldClose(window->getWindow()))
	{
		glfwPollEvents();

		face->computeFace();

		renderer->drawFace();

		menu->draw();
		window->refresh();

		cv::Mat frame;
		if (!camera.read(frame)) continue;

		auto correspondences = tracker->getCorrespondences(frame);

		solver->solve(correspondences);
	}

	return 0;
}
