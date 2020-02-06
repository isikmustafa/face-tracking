#version 330 core

layout(triangles) in;
layout (triangle_strip, max_vertices=3) out;

in V2G
{
	vec4 position; 
	vec3 normal;
	vec3 albedo;
	int id; 
} vertices[3];

out G2P
{
	noperspective vec3 normal;
	noperspective vec3 albedo;
	noperspective vec3 barycentrics; 
	flat ivec3 ids; 
} v;

void main()
{
	v.ids = ivec3(vertices[0].id, vertices[1].id, vertices[2].id); 
	
	gl_Position = vertices[0].position; 
	v.normal = vertices[0].normal;
	v.albedo = vertices[0].albedo; 
	v.barycentrics = vec3(1.0f, 0.0f, 0.0f); 
	EmitVertex();

	gl_Position = vertices[1].position;
	v.normal = vertices[1].normal;
	v.albedo = vertices[1].albedo; 
	v.barycentrics = vec3(0.0f, 1.0f, 0.0f); 
	EmitVertex();

	gl_Position = vertices[2].position;
	v.normal = vertices[2].normal;
	v.albedo = vertices[2].albedo; 
	v.barycentrics = vec3(0.0f, 0.0f, 1.0f); 
	EmitVertex();
}