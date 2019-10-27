#version 330 core

in vec3 Color;
out vec4 FragmentColor;

void main()
{
	FragmentColor = vec4(Color, 1.0f);
}