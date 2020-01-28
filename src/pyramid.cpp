#include "pyramid.h"

Pyramid::Pyramid(int number_of_levels, int top_width, int top_height, int highest_level)
	: m_face_framebuffer(number_of_levels, 0)
	, m_rt_rgb(number_of_levels, 0)
	, m_rt_barycentrics(number_of_levels, 0)
	, m_rt_vertex_ids(number_of_levels, 0)
	, m_rt_rgb_cuda_resource(number_of_levels, nullptr)
	, m_rt_barycentrics_cuda_resource(number_of_levels, nullptr)
	, m_rt_vertex_ids_cuda_resource(number_of_levels, nullptr)
	, m_depth_buffer(number_of_levels, 0)
	, m_widths(number_of_levels)
	, m_heights(number_of_levels)
	, m_highest_level(highest_level)
{
	m_widths[0] = top_width;
	m_heights[0] = top_height;
	for (int i = 1; i < number_of_levels; ++i)
	{
		m_widths[i] = m_widths[i - 1] / 2;
		m_heights[i] = m_heights[i - 1] / 2;
	}

	for (int i = highest_level; i < number_of_levels; ++i)
	{
		glGenFramebuffers(1, &m_face_framebuffer[i]);
		glBindFramebuffer(GL_FRAMEBUFFER, m_face_framebuffer[i]);

		// RGB render texture
		glGenTextures(1, &m_rt_rgb[i]);
		glBindTexture(GL_TEXTURE_2D, m_rt_rgb[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_widths[i], m_heights[i], 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_rt_rgb[i], 0);
		CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_rt_rgb_cuda_resource[i], m_rt_rgb[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

		// barycentrics render texture
		glGenTextures(1, &m_rt_barycentrics[i]);
		glBindTexture(GL_TEXTURE_2D, m_rt_barycentrics[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_widths[i], m_heights[i], 0, GL_RGBA, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, m_rt_barycentrics[i], 0);
		CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_rt_barycentrics_cuda_resource[i], m_rt_barycentrics[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

		// vertex ID render texture
		glGenTextures(1, &m_rt_vertex_ids[i]);
		glBindTexture(GL_TEXTURE_2D, m_rt_vertex_ids[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32I, m_widths[i], m_heights[i], 0, GL_RGBA_INTEGER, GL_INT, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, m_rt_vertex_ids[i], 0);
		CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_rt_vertex_ids_cuda_resource[i], m_rt_vertex_ids[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

		GLenum draw_buffers[3] = { GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2 };
		glDrawBuffers(3, draw_buffers); // "3" is the size of draw_buffers
		glGenRenderbuffers(1, &m_depth_buffer[i]);
		glBindRenderbuffer(GL_RENDERBUFFER, m_depth_buffer[i]);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_widths[i], m_heights[i]);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_buffer[i]);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			throw std::runtime_error("Error: Failed to create the framebuffer!");
		}
	}
}

void Pyramid::setGraphicsSettings(int pyramid_level, Face::GraphicsSettings& graphics_settings) const
{
	if (pyramid_level >= m_face_framebuffer.size())
	{
		throw std::runtime_error("Error: Invalid pyramid_level index!");
	}

	graphics_settings.framebuffer = m_face_framebuffer[pyramid_level];
	graphics_settings.rt_rgb = m_rt_rgb[pyramid_level];
	graphics_settings.rt_rgb_cuda_resource = m_rt_rgb_cuda_resource[pyramid_level];
	graphics_settings.rt_barycentrics_cuda_resource = m_rt_barycentrics_cuda_resource[pyramid_level];
	graphics_settings.rt_vertex_ids_cuda_resource = m_rt_vertex_ids_cuda_resource[pyramid_level];
	graphics_settings.texture_width = m_widths[pyramid_level];
	graphics_settings.texture_height = m_heights[pyramid_level];
}
