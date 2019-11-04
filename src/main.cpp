#include "window.h"
#include "face.h"
#include "tracker.h"
#include "renderer.h"
#include "menu.h"
#include "gauss_newton.h"

int main()
{
	const std::string path = "../MorphableModel/";

	// Composition root
	const auto window = std::make_shared<Window>();
	const auto face = std::make_shared<Face>(path);
	const auto solver = std::make_shared<GaussNewton>();

	const auto tracker = std::make_unique<Tracker>(solver);
	const auto menu = std::make_unique<Menu>(window, face);
	const auto renderer = std::make_unique<Renderer>(window, face);

	menu->initialize();

	tracker->start();
	renderer->start();

	return 0;
}
