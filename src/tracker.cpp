#include "tracker.h"
#include <utility>

Tracker::Tracker(const std::shared_ptr<Solver>& solver) : m_solver(std::move(solver)) {}

void Tracker::start() const
{
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		std::cout << "Unable to connect to camera" << std::endl;
		return;
	}

	dlib::image_window win;
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	std::thread track([&cap, &win, &pose_model, &detector, this]() {
		try
		{
			while (!win.is_closed())
			{
				// Grab a frame
				cv::Mat temp;
				if (!cap.read(temp))
				{
					break;
				}

				dlib::cv_image<dlib::bgr_pixel> cimg(temp);
				std::vector<dlib::rectangle> faces = detector(cimg);

				std::vector<dlib::full_object_detection> shapes;
				for (const auto& face : faces)
				{
					shapes.push_back(pose_model(cimg, face));
				}

				win.clear_overlay();
				win.set_image(cimg);
				win.add_overlay(dlib::render_face_detections(shapes));

				Correspondences correspondences;
				m_solver->process(correspondences);
			}
		}
		catch (dlib::serialization_error& e)
		{
			std::cout << std::endl << e.what() << std::endl;
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	});
}
