#version 330 core




in G2P
{
	vec3 normal;
	vec3 albedo;
	vec3 barycentrics; 
	flat ivec3 ids; 
} frag;

layout(location = 0) out vec4 fragment_color;
layout(location = 1) out vec4 barycentrics;
layout(location = 2) out vec4 vertex_indices;

uniform float sh_coefficients[9];

float computeSH(vec3 dir);

void main()
{
	float light = computeSH(normalize(frag.normal));
	fragment_color = vec4(light * frag_albedo, 1.0f);
	barycentrics = vec4(frag.barycentrics,light); 
	vertex_indices = vec4(frag.ids, 0); 

}

float computeSH(vec3 dir)
{
	//band 0 aka ambient
	float light = sh_coefficients[0];

	//band 1
	light += sh_coefficients[1] * dir.y;
	light += sh_coefficients[2] * dir.z;
	light += sh_coefficients[3] * dir.x;

	//band 2
	light += sh_coefficients[4] * dir.x * dir.y; 
	light += sh_coefficients[5] * dir.y * dir.z; 
	light += sh_coefficients[6] * (3.0f * dir.z * dir.z - 1.0f);
	light += sh_coefficients[7] * dir.x * dir.z; 
	light += sh_coefficients[8] * (dir.x * dir.x - dir.y * dir.y);

	return light;
}