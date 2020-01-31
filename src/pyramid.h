#pragma once

#include "face.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector>

class Pyramid
{
public:
	Pyramid(int number_of_levels, int top_width, int top_height);
	Pyramid(Pyramid&) = delete;
	Pyramid(Pyramid&& rhs) = delete;
	Pyramid& operator=(Pyramid&) = delete;
	Pyramid& operator=(Pyramid&&) = delete;

	//TODO: Don't be lazy and write the destructor
	//~Pyramid();

	void setGraphicsSettings(int pyramid_level, Face::GraphicsSettings& graphics_settings) const;

	int getNumberOfLevels() const { return m_face_framebuffer.size(); }

private:
	std::vector<GLuint> m_face_framebuffer;
	std::vector<GLuint> m_rt_rgb;
	std::vector<GLuint> m_rt_barycentrics;
	std::vector<GLuint> m_rt_vertex_ids;
	std::vector<cudaGraphicsResource_t> m_rt_rgb_cuda_resource;
	std::vector<cudaGraphicsResource_t> m_rt_barycentrics_cuda_resource;
	std::vector<cudaGraphicsResource_t> m_rt_vertex_ids_cuda_resource;
	std::vector<GLuint> m_depth_buffer;
	std::vector<int> m_widths;
	std::vector<int> m_heights;
};
