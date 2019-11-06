#pragma once

#include "face.h"
#include "correspondences.h"

class Solver
{
public:
	Solver(const std::shared_ptr<Face>&);

	virtual ~Solver() = default;
	virtual void solve(Correspondences&) = 0;

protected:
	std::shared_ptr<Face> m_face;
};

inline Solver::Solver(const std::shared_ptr<Face>&face) : m_face(face){}
