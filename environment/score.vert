#version 120

uniform mat4 u_mv;
uniform mat4 u_mvp;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;
varying float v_eye_depth;
varying vec3 v_eye_normal;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;

    vec3 v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz;
    v_eye_depth = -v_eye_pos.z;
    v_eye_normal = normalize((u_mv * vec4(a_normal, 0.0)).xyz);
}