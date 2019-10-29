#version 330 core

in vec3 albedo;
out vec4 fragment_color;

void main()
{
	fragment_color = vec4(albedo, 1.0f);
}