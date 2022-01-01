#version 120

const float TAU = 0.02; // in meters
const float ALPHA = 0.7; // largest allowed angular error

uniform sampler2D u_texture;
uniform sampler2D u_observation;
uniform int u_mode;

varying vec2 v_texcoord;
varying float v_eye_depth;
varying vec3 v_eye_normal;

void main() {
    // STAGE 1) \hat{d}_T and \hat{n}_T: per-pixel depth d and normal n under the estimated pose T
    if(u_mode == 0) {
        vec3 eye_normal = normalize(v_eye_normal);
        gl_FragColor = vec4(eye_normal.x, eye_normal.y, eye_normal.z, v_eye_depth);
    }

    // STAGE 2) f_d(T) and f_n(T): compute verification sub-scores for per-pixel depth and normal fit
    else if(u_mode == 1) {
        vec3 ren_normal = texture2D(u_texture, v_texcoord).xyz;
        float ren_depth = float(texture2D(u_texture, v_texcoord).w);
        vec3 obs_normal = texture2D(u_observation, v_texcoord).xyz;
        float obs_depth = float(texture2D(u_observation, v_texcoord).w);

        float valid = 0.0;
        float delta_d = 0.0;
        float delta_n = 0.0;

        if (ren_depth > 0.0 || obs_depth > 0.0) {
            valid = 1.0;
            float dist = abs(ren_depth - obs_depth);
            delta_d = 1.0 - min(1.0, dist/TAU);

            float ang = clamp(dot(obs_normal, ren_normal), 0.0, 1.0);  // 1 if same, 0 if orthogonal (and <0 beyond)
            delta_n = 1.0 - min(1.0, (1.0-ang)/ALPHA);
        }

        gl_FragColor = vec4(valid, delta_d, delta_n, 1.0);
    }
}
