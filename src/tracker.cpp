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

Correspondences Tracker::getCorrespondences(cv::Mat& frame)
{
	try 
	{
		dlib::cv_image<dlib::bgr_pixel> cimg(frame);
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
