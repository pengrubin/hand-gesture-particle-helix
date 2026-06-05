// Extract pixels brighter than uThreshold. Uses the same Rec. 601 luma
// weights as the TouchDesigner original (W = vec3(0.2989, 0.5870, 0.1140)).
// smoothstep gives a soft knee so the bloom doesn't pop on/off at the edge.

uniform sampler2D tInput;
uniform float uThreshold;

varying vec2 vUv;

const vec3 W = vec3(0.2989, 0.5870, 0.1140);

void main() {
  vec3 color = texture2D(tInput, vUv).rgb;
  float luma = dot(color, W);
  float gate = smoothstep(uThreshold, uThreshold + 0.15, luma);
  gl_FragColor = vec4(color * gate, 1.0);
}
