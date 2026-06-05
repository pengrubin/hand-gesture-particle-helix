// Passthrough copy. Used to (a) snapshot the current frame into the
// feedback history buffer and (b) blit the final framebuffer to canvas.

uniform sampler2D tInput;
varying vec2 vUv;

void main() {
  gl_FragColor = texture2D(tInput, vUv);
}
