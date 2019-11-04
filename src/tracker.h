#pragma once

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include "solver.h"

class Tracker
{
public:
	Tracker(const std::shared_ptr<Solver>&);
	void start() const;
private:
	std::shared_ptr<Solver> m_solver;
};
