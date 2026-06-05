// Fragment shader: dark base + cold rim Fresnel + warm mid-band accent
// where the surface is bulging + high-band glints on the crests.
//
// Bloom will be layered on top in M4; for M3 this looks coherent already.

uniform float uMid;
uniform float uHigh;

varying vec3 vNormal;
varying vec3 vViewPos;
varying float vDisp;

void main() {
  vec3 n = normalize(vNormal);
  vec3 viewDir = normalize(-vViewPos);

  float ndv = max(0.0, dot(n, viewDir));
  float fresnel = pow(1.0 - ndv, 2.5);

  vec3 baseColor = vec3(0.04, 0.05, 0.10);
  vec3 rimColor  = vec3(0.78, 0.86, 1.00);

  vec3 color = mix(baseColor, rimColor, fresnel);

  // Mid-band drives a warm accent that scales with where the surface bulges
  // outward. smoothstep keeps the inner regions quiet so the rim still reads.
  color += vec3(0.55, 0.30, 0.12) * uMid * smoothstep(0.0, 0.25, vDisp);

  // High-band drives extra brightness on the noise peaks — fast attack.
  color += vec3(uHigh * 0.75) * smoothstep(0.05, 0.22, vDisp);

  gl_FragColor = vec4(color, 1.0);
}
