#version 330 core

in vec2 tex_coord;
out vec4 color;

uniform sampler2D face;
uniform sampler2D background; 

void main()
{
	vec4 face_color = texture(face, tex_coord);
	if(face_color.w > 0.0f)
	{
		color = face_color; 
	}
	else
	{
		color = texture(background, tex_coord);
	}
}