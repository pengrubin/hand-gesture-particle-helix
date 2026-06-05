// Shared vertex shader for all full-screen post-process passes. The
// geometry is a clip-space quad covering [-1, +1]^2, so we ignore any
// transform matrices and just pass uv through.

varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
