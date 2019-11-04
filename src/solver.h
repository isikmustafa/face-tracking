#pragma once

#include "face.h"
#include "correspondences.h"

class Solver
{
public:
	virtual ~Solver() = default;
	virtual void process(Correspondences&) = 0;
protected:
	std::shared_ptr<Face> m_face;
};
