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

		auto frame_size = frame.size();
		auto two_over_width = 2.0f / static_cast<float>(frame_size.width);
		auto two_over_height = 2.0f / static_cast<float>(frame_size.height);
		for (unsigned long i = 0; i < 60; ++i)
		{
			const dlib::point& point = shape.part(i);
			//circles.emplace_back(point, 2, color);

			//Normalize sparse feature positions such that left-bottom corner is (-1, -1) and top-right corner is (+1, +1).
			//This is the OpenGL convention.
			sparse_features.emplace_back(point.x() * two_over_width - 1.0f, 1.0f - point.y() * two_over_height);
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
