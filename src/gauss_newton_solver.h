#pragma once

#include "face.h"

class GaussNewtonSolver
{
public:
	void solve(const std::vector<glm::vec2>& sparse_features, Face& face);
};