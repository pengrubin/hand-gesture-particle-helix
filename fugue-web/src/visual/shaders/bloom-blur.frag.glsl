// Single-pass Kawase blur step. Samples four taps at +/-offset in both
// diagonals, averages them. Chain 3 passes with increasing offsets and you
// get a quality-equivalent of a 13×13 Gaussian at a fraction of the cost.

uniform sampler2D tInput;
uniform vec2 uResolution;
uniform float uOffset;

varying vec2 vUv;

void main() {
  vec2 texel = 1.0 / uResolution;
  vec2 d = texel * uOffset;
  vec3 sum =
      texture2D(tInput, vUv + vec2( d.x,  d.y)).rgb
    + texture2D(tInput, vUv + vec2( d.x, -d.y)).rgb
    + texture2D(tInput, vUv + vec2(-d.x,  d.y)).rgb
    + texture2D(tInput, vUv + vec2(-d.x, -d.y)).rgb;
  gl_FragColor = vec4(sum * 0.25, 1.0);
}
