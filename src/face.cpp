#include "face.h"
#include "glsl_program.h"

#include <assert.h>
#include <fstream>
#include <iostream>

Face::Face(const std::string& filename)
{
	std::ifstream file(filename);
	std::string str_dummy;
	file >> str_dummy;

	int number_of_vertices;
	int number_of_faces;
	int int_dummy;
	file >> number_of_vertices;
	file >> number_of_faces;
	file >> int_dummy;

	std::vector<glm::vec3> positions(number_of_vertices);
	std::vector<glm::vec3> colors(number_of_vertices);
	std::vector<glm::vec2> tex_coords(number_of_vertices);
	m_number_of_indices = 3 * number_of_faces;
	std::vector<unsigned int> indices(m_number_of_indices);
	constexpr float mesh_scale = 1 / 1000000.0f;
	for (int i = 0; i < number_of_vertices; ++i)
	{
		file >> positions[i].x >> positions[i].y >> positions[i].z;
		positions[i] *= mesh_scale;

		file >> colors[i].x >> colors[i].y >> colors[i].z;
		colors[i] *= (1.0f / 255.0f);
		file >> int_dummy;

		file >> tex_coords[i].x >> tex_coords[i].y;
	}
	for (int i = 0; i < number_of_faces; ++i)
	{
		file >> int_dummy;
		file >> indices[i * 3] >> indices[i * 3 + 1] >> indices[i * 3 + 2];
	}
	file.close();

	int positions_byte_size = number_of_vertices * sizeof(glm::vec3);
	int colors_byte_size = number_of_vertices * sizeof(glm::vec3);
	int tex_coords_byte_size = number_of_vertices * sizeof(glm::vec2);
	glGenVertexArrays(1, &m_vertex_array);
	glGenBuffers(1, &m_vertex_buffer);
	glGenBuffers(1, &m_index_buffer);

	assert(m_vertex_array);
	assert(m_vertex_buffer);
	assert(m_index_buffer);

	glBindVertexArray(m_vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, positions_byte_size + colors_byte_size + tex_coords_byte_size, nullptr, GL_STATIC_DRAW);

	glBufferSubData(GL_ARRAY_BUFFER, 0                                     , positions_byte_size , positions.data());
	glBufferSubData(GL_ARRAY_BUFFER, positions_byte_size                   , colors_byte_size    , colors.data());
	glBufferSubData(GL_ARRAY_BUFFER, positions_byte_size + colors_byte_size, tex_coords_byte_size, tex_coords.data());

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)(positions_byte_size));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (GLvoid*)(positions_byte_size + colors_byte_size));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_number_of_indices * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

Face::Face(Face&& rhs)
{
	m_vertex_array = rhs.m_vertex_array;
	m_vertex_buffer = rhs.m_vertex_buffer;
	m_index_buffer = rhs.m_index_buffer;
	m_number_of_indices = rhs.m_number_of_indices;
	rhs.m_vertex_array = 0;
	rhs.m_vertex_buffer = 0;
	rhs.m_index_buffer = 0;
	rhs.m_number_of_indices = 0;
}

Face& Face::operator=(Face&& rhs)
{
	m_vertex_array = rhs.m_vertex_array;
	m_vertex_buffer = rhs.m_vertex_buffer;
	m_index_buffer = rhs.m_index_buffer;
	m_number_of_indices = rhs.m_number_of_indices;
	rhs.m_vertex_array = 0;
	rhs.m_vertex_buffer = 0;
	rhs.m_index_buffer = 0;
	rhs.m_number_of_indices = 0;

	return *this;
}

Face::~Face()
{
	if (m_vertex_buffer)
	{
		glDeleteBuffers(1, &m_vertex_buffer);
		m_vertex_buffer = 0;
	}
	if (m_index_buffer)
	{
		glDeleteBuffers(1, &m_index_buffer);
		m_index_buffer = 0;
	}
	if (m_vertex_array)
	{
		glDeleteVertexArrays(1, &m_vertex_array);
		m_vertex_array = 0;
	}
}

void Face::draw(const GLSLProgram& program) const
{
	program.use();

	glBindVertexArray(m_vertex_array);
	glDrawElements(GL_TRIANGLES, m_number_of_indices, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}