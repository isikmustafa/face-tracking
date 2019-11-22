#pragma once

#include <glm/glm.hpp>
#include <utility>
#include <vector>

class PriorSparseFeatures
{
public:
	static PriorSparseFeatures& get();

	void addPriorPosition(const glm::vec3& position);

	const std::vector<int>& getPriorIds() { return m_prior_ids; };
	const std::vector<glm::vec3>& getPriorPositions() { return m_prior_positions; };

private:
	std::vector<int> m_prior_ids;
	std::vector<glm::vec3> m_prior_positions;

private:
	PriorSparseFeatures();
	PriorSparseFeatures(const PriorSparseFeatures&);
	PriorSparseFeatures(PriorSparseFeatures&&);
};
