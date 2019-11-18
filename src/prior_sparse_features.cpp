#include "prior_sparse_features.h"

std::vector<int> PriorSparseFeatures::m_prior_ids;
std::vector<glm::vec3> PriorSparseFeatures::m_prior_positions;
PriorSparseFeatures::CreateIds PriorSparseFeatures::m_initializer;

void PriorSparseFeatures::addPriorPosition(const glm::vec3& position)
{
	m_prior_positions.emplace_back(position);
}
