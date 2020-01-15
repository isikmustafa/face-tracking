#pragma once 

#include "gauss_newton_solver.h"
#include "util.h"
#include "device_util.h"
#include "device_array.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

__global__ void textureRgbTestKernel(cudaTextureObject_t texture, uchar* frame, float* arr, int width, int height)
{
	auto index = util::getThreadIndex2D();
	if (index.x >= width || index.y >= height)
	{
		return;
	}
	int y = height - 1 - index.y; // "height - 1 - index.y" is used since OpenGL uses left-bottom corner as texture origin.
	float4 color = tex2D<float4>(texture, index.x, y);
	auto idx = (index.x + index.y * width) * 3;

	if (color.w > 0)
	{
		arr[idx] = color.x;
		arr[idx + 1] = color.y;
		arr[idx + 2] = color.z;
	}
	else
	{
		arr[idx] = frame[idx] / 255.0;
		arr[idx + 1] = frame[idx + 1] / 255.0;
		arr[idx + 2] = frame[idx + 2] / 255.0;
	}
}

__global__ void textureBarycentricsVertexIdsTestKernel(cudaTextureObject_t texture_barycentrics, cudaTextureObject_t texture_vertex_ids, glm::vec3* albedos,
	float* arr, int width, int height)
{
	auto index = util::getThreadIndex2D();

	if (index.x >= width || index.y >= height)
	{
		return;
	}

	int y = height - 1 - index.y; // "height - 1 - index.y" is used since OpenGL uses left-bottom corner as texture origin.
	float4 barycentrics_light = tex2D<float4>(texture_barycentrics, index.x, y); // barycentrics_light.w is light.
	int4 vertex_ids = tex2D<int4>(texture_vertex_ids, index.x, y);

	auto albedo_v0 = albedos[vertex_ids.x];
	auto albedo_v1 = albedos[vertex_ids.y];
	auto albedo_v2 = albedos[vertex_ids.z];

	auto albedo = barycentrics_light.x * albedo_v0 + barycentrics_light.y * albedo_v1 + barycentrics_light.z * albedo_v2;
	auto color = albedo * barycentrics_light.w;

	auto idx = (index.x + index.y * width) * 3;
	arr[idx] = color.x;
	arr[idx + 1] = color.y;
	arr[idx + 2] = color.z;
}

void GaussNewtonSolver::debugFrameBufferTextures(Face& face, uchar* frame, const std::string& rgb_filepath, const std::string& deferred_filepath)
{
	int img_width = face.m_graphics_settings.texture_width;
	int img_height = face.m_graphics_settings.texture_height;
	util::DeviceArray<float> temp_memory(img_width * img_height * 3);

	dim3 threads(16, 16);
	dim3 blocks(img_width / threads.x + 1, img_height / threads.y + 1);

	textureRgbTestKernel << <blocks, threads >> > (m_texture_rgb, frame, temp_memory.getPtr(), img_width, img_height);

	std::vector<float> temp_memory_host(img_width * img_height * 3);
	util::copy(temp_memory_host, temp_memory, temp_memory.getSize());

	cv::Mat image(cv::Size(img_width, img_height), CV_8UC3);
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			auto idx = (x + y * img_width) * 3;

			// OpenCV expects it to be an BGRA image.
			image.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255.0f * cv::Vec3f(temp_memory_host[idx + 2], temp_memory_host[idx + 1], temp_memory_host[idx]));
		}
	}
	cv::imwrite(rgb_filepath, image);

	textureBarycentricsVertexIdsTestKernel << <blocks, threads >> > (m_texture_barycentrics, m_texture_vertex_ids, face.m_current_face_gpu.getPtr() + face.m_number_of_vertices,
		temp_memory.getPtr(), img_width, img_height);

	util::copy(temp_memory_host, temp_memory, temp_memory.getSize());
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			auto idx = (x + y * img_width) * 3;

			// OpenCV expects it to be an BGRA image.
			image.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255.0f * cv::Vec3f(temp_memory_host[idx + 2], temp_memory_host[idx + 1], temp_memory_host[idx]));
		}
	}
	cv::imwrite(deferred_filepath, image);
}