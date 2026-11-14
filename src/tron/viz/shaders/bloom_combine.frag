#version 330
// Composition : scene + halo floute (additif), sans depasser le blanc.
uniform sampler2D scene;
uniform sampler2D bloom;
uniform float intensity;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 color = texture(scene, v_uv).rgb + texture(bloom, v_uv).rgb * intensity;
    fragColor = vec4(min(color, vec3(1.0)), 1.0);
}
