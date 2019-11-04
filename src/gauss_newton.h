#pragma once

#include "solver.h"

class GaussNewton : public Solver
{
public:
	void process(Correspondences&) override;
};

inline void GaussNewton::process(Correspondences&){}