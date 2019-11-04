#pragma once

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include "correspondences.h"

class Tracker
{
public:
	Tracker();
	Correspondences getCorrespondences(cv::Mat&);
private:
	dlib::frontal_face_detector m_detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor m_pose_model;
};
