#include "tracker.h"
#include <utility>

Tracker::Tracker() {
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> m_pose_model;
}

Correspondences Tracker::getCorrespondences()
{
	if (!m_camera.isOpened()) {
		return Correspondences();
	}

	try
	{
		cv::Mat temp;
		if (!m_camera.read(temp))
		{
			return Correspondences();
		}

		dlib::cv_image<dlib::bgr_pixel> cimg(temp);
		std::vector<dlib::rectangle> faces = m_detector(cimg);

		std::vector<dlib::full_object_detection> shapes;
		for (const auto& face : faces)
		{
			shapes.push_back(m_pose_model(cimg, face));
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

	return Correspondences();
}
