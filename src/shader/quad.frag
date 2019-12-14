#version 330 core

in vec2 tex_coord;
out vec4 color;

uniform sampler2D face;
uniform sampler2D background; 

void main()
{
	
	color = texture(background, tex_coord);
	vec4 face_color = texture(face, tex_coord);
	if(face_color.w >0)
	{
		color = face_color; 
	}
	//color = vec4(1,0,0,1); 
	//color = vec4(tex_coord,0,1);
}