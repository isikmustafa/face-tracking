#include "correspondences.h"

std::vector<int> Correspondences::m_prior_ids;
std::vector<glm::vec3> Correspondences::m_prior_positions;
Correspondences::CreateIds Correspondences::m_initializer;

std::vector<Point> Correspondences::getPoints()
{
	auto points = std::vector<Point>();

	for (int i = 0; i < m_feature_points.size(); i++)
	{
		points.emplace_back(m_feature_points[i], m_prior_positions[i]);
	}

	return points;
}

void Correspondences::addFeaturePoint(glm::vec2 point)
{
	m_feature_points.push_back(point);
}

void Correspondences::addPriorPosition(glm::vec3 position)
{
	m_prior_positions.emplace_back(position);
}
