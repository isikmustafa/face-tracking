#version 330 core

in G2P
{
	vec3 normal;
	vec3 albedo;
	vec3 barycentrics; 
} frag;


out vec4 fragment_color;

uniform float sh_coefficients[9];

float computeSH(vec3 dir);

void main()
{
	float light = computeSH(normalize(frag.normal));
	fragment_color = light * vec4(frag.albedo, 1.0f);

	fragment_color = vec4(frag.barycentrics,1.0f); 

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