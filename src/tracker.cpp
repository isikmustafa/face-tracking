#include "tracker.h"
#include <utility>

Tracker::Tracker() {
	try
	{
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> m_pose_model;
	}
	catch (dlib::serialization_error& e)
	{
		std::cout << std::endl << e.what() << std::endl;
	}
}

Correspondences Tracker::getCorrespondences()
{
	if (!m_camera.isOpened())
		return Correspondences();

	try {
		cv::Mat temp;
		if (!m_camera.read(temp))
			return Correspondences();

		dlib::cv_image<dlib::bgr_pixel> cimg(temp);
		std::vector<dlib::rectangle> faces = m_detector(cimg);

		// Consider only one face for the processing
		if (faces.empty())
			return Correspondences();

		dlib::full_object_detection shape = m_pose_model(cimg, faces[0]);

		//TODO calculate Correspondences
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	return Correspondences();
}
