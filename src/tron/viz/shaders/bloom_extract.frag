#version 330
// Extraction des pixels lumineux : seuil progressif sur la composante la plus forte.
uniform sampler2D source;
uniform float threshold;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 color = texture(source, v_uv).rgb;
    float peak = max(color.r, max(color.g, color.b));
    float keep = smoothstep(threshold, 1.0, peak);
    fragColor = vec4(color * keep, 1.0);
}
