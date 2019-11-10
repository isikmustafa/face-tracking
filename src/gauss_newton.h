#pragma once

#include "solver.h"

class GaussNewton : public Solver
{
public:
	GaussNewton(const std::shared_ptr<Face>&face) : Solver(face) {}

	void solve(std::vector<Point>) override;
};

inline void GaussNewton::solve(std::vector<Point> points){}