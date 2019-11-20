#include "prior_sparse_features.h"

PriorSparseFeatures::PriorSparseFeatures()
{
	// Chin (1-16)
	m_prior_ids.emplace_back(22428);
	m_prior_ids.emplace_back(21945);
	m_prior_ids.emplace_back(22225);
	m_prior_ids.emplace_back(43915);
	m_prior_ids.emplace_back(46042);
	m_prior_ids.emplace_back(47150);
	m_prior_ids.emplace_back(47666);
	m_prior_ids.emplace_back(48185);
	m_prior_ids.emplace_back(48639);
	m_prior_ids.emplace_back(49162);
	m_prior_ids.emplace_back(50497);
	m_prior_ids.emplace_back(51759);
	m_prior_ids.emplace_back(32934);
	m_prior_ids.emplace_back(33298);
	m_prior_ids.emplace_back(32234);
	m_prior_ids.emplace_back(31699);

	// Eye browes (17-26)
	m_prior_ids.emplace_back(38791);
	m_prior_ids.emplace_back(39632);
	m_prior_ids.emplace_back(40119);
	m_prior_ids.emplace_back(40348);
	m_prior_ids.emplace_back(6968);
	m_prior_ids.emplace_back(9548);
	m_prior_ids.emplace_back(41223);
	m_prior_ids.emplace_back(41513);
	m_prior_ids.emplace_back(41904);
	m_prior_ids.emplace_back(42556);

	// Nose (27-35)
	m_prior_ids.emplace_back(8160);
	m_prior_ids.emplace_back(8306);
	m_prior_ids.emplace_back(8315);
	m_prior_ids.emplace_back(8320);
	m_prior_ids.emplace_back(6393);
	m_prior_ids.emplace_back(7171);
	m_prior_ids.emplace_back(8329);
	m_prior_ids.emplace_back(9234);
	m_prior_ids.emplace_back(10003);

	// Eye (36-41)
	m_prior_ids.emplace_back(2086);
	m_prior_ids.emplace_back(3887);
	m_prior_ids.emplace_back(4791);
	m_prior_ids.emplace_back(5957);
	m_prior_ids.emplace_back(4673);
	m_prior_ids.emplace_back(3641);

	// Eye (42-47)
	m_prior_ids.emplace_back(10086);
	m_prior_ids.emplace_back(11112);
	m_prior_ids.emplace_back(12401);
	m_prior_ids.emplace_back(14082);
	m_prior_ids.emplace_back(12671);
	m_prior_ids.emplace_back(11641);

	// Mouth (48-59)
	m_prior_ids.emplace_back(5393);
	m_prior_ids.emplace_back(6285);
	m_prior_ids.emplace_back(7442);
	m_prior_ids.emplace_back(8347);
	m_prior_ids.emplace_back(8991);
	m_prior_ids.emplace_back(10026);
	m_prior_ids.emplace_back(11069);
	m_prior_ids.emplace_back(9534);
	m_prior_ids.emplace_back(8759);
	m_prior_ids.emplace_back(8373);
	m_prior_ids.emplace_back(7729);
	m_prior_ids.emplace_back(6698);
}

PriorSparseFeatures& PriorSparseFeatures::get()
{
	static PriorSparseFeatures obj;
	return obj;
}

void PriorSparseFeatures::addPriorPosition(const glm::vec3& position)
{
	m_prior_positions.emplace_back(position);
}
