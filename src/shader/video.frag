#version 330 core

in vec2 tex_coord;
layout(location = 0) out vec4 color;

uniform sampler2D face;
uniform sampler2D background; 

void main()
{
	vec2 tex_coord_tmp = tex_coord;
	if (tex_coord_tmp.x < 0.5)
	{
		tex_coord_tmp.x *= 2.0f;
		vec4 face_color = texture(face, vec2(tex_coord_tmp.x, 1.0f - tex_coord_tmp.y));
		if(face_color.w > 0.0f)
		{
			color = face_color;
		}
		else
		{
			color = texture(background, tex_coord_tmp);
		}
	}
	else
	{
		tex_coord_tmp.x -= 0.5f;
		tex_coord_tmp.x *= 2.0f;
		color = texture(background, tex_coord_tmp);
	}
}
