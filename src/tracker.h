#pragma once

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <glm/glm.hpp>
#include <vector>

class Tracker
{
public:
	Tracker();
	std::vector<glm::vec2> getSparseFeatures(const cv::Mat& frame);

private:
	dlib::frontal_face_detector m_detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor m_pose_model;
	dlib::image_window m_window;
};
