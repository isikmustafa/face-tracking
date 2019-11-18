#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 tex_coord;

out vec3 frag_albedo;
out vec3 frag_normal;

uniform mat4 model;
uniform mat4 projection;

void main()
{
    frag_albedo = color;
	frag_normal = normalize(normal);

    gl_Position = projection * model * vec4(position, 1.0f);
}