#version 330 core

in vec3 albedo;
in vec3 vertex_normal;
out vec4 fragment_color;

uniform float sh_coefficients[9];

float computeSH(vec3 normal);

void main()
{
	float light = computeSH(vertex_normal);
	fragment_color = light * vec4(albedo, 1.0f);
}

float computeSH(vec3 normal)
{
	//band 0 aka ambient
	float light = sh_coefficients[0];

	//band 1
	light += sh_coefficients[1] * normal.x;
	light += sh_coefficients[2] * normal.y;
	light += sh_coefficients[3] * normal.z;

	//band 2
	vec3 normalSq = normal*normal; 
	light += sh_coefficients[4] * normal.x * normal.y; 
	light += sh_coefficients[5] * normal.y * normal.z; 
	light += sh_coefficients[6] * (2 * normalSq.z - normalSq.x - normalSq.y);
	light += sh_coefficients[7] * normal.z * normal.x; 
	light += sh_coefficients[8] * (normalSq.x - normalSq.z);

	return light;
}