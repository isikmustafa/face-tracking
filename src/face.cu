#include "device_util.h"
#include "face.h"

__global__ void compute(int number_of_vertices, int number_of_faces, glm::vec3* current_face, glm::vec3* faces)
{
	int index = util::getThreadIndex1D();
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < number_of_faces; i += stride)
	{
		const glm::vec3 face = faces[index];

		glm::vec3 v0 = current_face[int(face.x)];
		glm::vec3 v1 = current_face[int(face.y)];
		glm::vec3 v2 = current_face[int(face.z)];

		glm::vec3 n = normalize(cross((v1 - v0), (v2 - v0)));

		float* ptr = reinterpret_cast<float*>(current_face);

		for (int i : { face.x, face.y, face.z })
		{
			const int position = (i + 2 * number_of_vertices) * 3;

			atomicAdd(&ptr[position + 0], n.x);
			atomicAdd(&ptr[position + 1], n.y);
			atomicAdd(&ptr[position + 2], n.z);
		}
	}
}

void Face::computeNormals() {
	const int number_of_faces = m_number_of_indices / 3;
	int blockSize = 256;
	int numBlocks = (number_of_faces + blockSize - 1) / blockSize;
	compute<<<numBlocks, blockSize>>>(
		m_number_of_vertices,
		number_of_faces,
		reinterpret_cast<glm::vec3*>(m_current_face_gpu.getPtr()),
		reinterpret_cast<glm::vec3*>(m_faces_gpu.getPtr()));
}
