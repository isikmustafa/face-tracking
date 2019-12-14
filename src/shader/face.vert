#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 tex_coord;

out V2G
{
 vec4 position;
 vec3 normal;
 vec3 albedo;
 int id; 
} vertex;

uniform mat4 model;
uniform mat4 projection;

void main()
{
	vertex.position = projection * model * vec4(position, 1.0f);
    vertex.albedo = color;
	vertex.normal = normalize(normal);
	vertex.id = gl_VertexID; 
//    gl_Position = projection * model * vec4(position, 1.0f);
}