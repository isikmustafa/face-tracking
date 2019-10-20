#version 330 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texture_coordinate;

out vec2 tex_coord;

void main()
{
    gl_Position = vec4(position, 0.0f, 1.0f);
    tex_coord = texture_coordinate;
}