#pragma once

#include <glm/glm.hpp>
#include <vector>

class PriorSparseFeatures
{
public:
	static PriorSparseFeatures& get();

	const std::vector<int>& getPriorIds() { return m_prior_ids; };

private:
	std::vector<int> m_prior_ids;

private:
	PriorSparseFeatures();
	PriorSparseFeatures(const PriorSparseFeatures&);
	PriorSparseFeatures(PriorSparseFeatures&&);
};
