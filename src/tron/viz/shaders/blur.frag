#version 330
// Flou gaussien separable a 9 echantillons ; direction = pas d'un texel (horizontal ou vertical).
uniform sampler2D source;
uniform vec2 direction;
in vec2 v_uv;
out vec4 fragColor;

const float WEIGHTS[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec3 acc = texture(source, v_uv).rgb * WEIGHTS[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = direction * float(i) * 1.5;
        acc += texture(source, v_uv + offset).rgb * WEIGHTS[i];
        acc += texture(source, v_uv - offset).rgb * WEIGHTS[i];
    }
    fragColor = vec4(acc, 1.0);
}
