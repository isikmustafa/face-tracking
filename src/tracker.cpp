#include "tracker.h"
#include <utility>

Tracker::Tracker()
{
	try
	{
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> m_pose_model;
	}
	catch (dlib::serialization_error& e)
	{
		std::cout << std::endl << e.what() << std::endl;
	}
}

std::vector<glm::vec2> Tracker::getSparseFeatures(const cv::Mat& frame)
{
	std::vector<glm::vec2> sparse_features;

	try
	{
		dlib::cv_image<dlib::bgr_pixel> cimg(frame);
		std::vector<dlib::rectangle> faces = m_detector(cimg);

		// Consider only one face for the processing
		if (faces.empty())
		{
			return sparse_features;
		}

		dlib::full_object_detection shape = m_pose_model(cimg, faces[0]);

		const dlib::rgb_pixel color = dlib::rgb_pixel(0, 255, 0);
		std::vector<dlib::image_window::overlay_circle> circles;

		for (unsigned long i = 1; i <= 59; ++i)
		{
			const dlib::point& point = shape.part(i);
			circles.emplace_back(point, 2, color);
			sparse_features.emplace_back(glm::vec2(point.x(), point.y()));
		}

		//m_window.clear_overlay();
		//m_window.set_image(cimg);

		//m_window.add_overlay(circles);
		//m_window.add_overlay(render_face_detections(shape));
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	return sparse_features;
}
