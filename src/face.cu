#include "device_util.h"
#include "face.h"

__global__ void computeNormalsKernel(int number_of_vertices, int number_of_faces, glm::vec3* current_face, glm::ivec3* faces)
{
	int index = util::getThreadIndex1D();
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < number_of_faces; i += stride)
	{
		const auto face = faces[index];

		glm::vec3 v0 = current_face[face.x];
		glm::vec3 v1 = current_face[face.y];
		glm::vec3 v2 = current_face[face.z];

		// Not normalizing face_normal is actually a way to use weighted average of normals of neighbouring triangles
		// where weights are the areas of the triangles.
		glm::vec3 face_normal = glm::cross((v1 - v0), (v2 - v0));
		auto normals = &current_face[2 * number_of_vertices];

		for (int i : { face.x, face.y, face.z })
		{
			auto& vertex_normal = normals[i];

			atomicAdd(&vertex_normal.x, face_normal.x);
			atomicAdd(&vertex_normal.y, face_normal.y);
			atomicAdd(&vertex_normal.z, face_normal.z);
		}
	}
}

void Face::computeNormals()
{
	const int number_of_faces = m_number_of_indices / 3;
	int block_size = 256;
	int num_blocks = (number_of_faces + block_size - 1) / block_size;
	computeNormalsKernel <<<num_blocks, block_size>>>(
		m_number_of_vertices,
		number_of_faces,
		m_current_face_gpu.getPtr(),
		m_faces_gpu.getPtr());
}
