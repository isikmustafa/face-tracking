#pragma once

#include "solver.h"

class GaussNewton : public Solver
{
public:
	GaussNewton(const std::shared_ptr<Face>&face) : Solver(face) {}

	void process(Correspondences&) override;
};

inline void GaussNewton::process(Correspondences&) {}