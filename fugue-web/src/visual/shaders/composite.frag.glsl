// Final composite. Direct port of the TouchDesigner bloom shader, plus
// an extra max() against a decayed previous frame for the RGBA-delay trail.
//
// TD original (sTD2DInputs[0] = scene, [1] = pre-blurred bloom):
//   bloom *= uBloomIntensity;
//   bloom += uGlowColor * luma * uGlowIntensity;
//   fragColor = scene * uInputLevel + vec4(bloom, luma);
//
// Here uInputLevel is fixed at 1.0 — the user-facing dial is bloomIntensity
// + glowIntensity. Both are driven from audio bands in main.ts.

uniform sampler2D tScene;
uniform sampler2D tBloom;
uniform sampler2D tPrev;
uniform float uBloomIntensity;
uniform float uGlowIntensity;
uniform vec3  uGlowColor;
uniform float uFeedbackDecay;

varying vec2 vUv;

const vec3 W = vec3(0.2989, 0.5870, 0.1140);

void main() {
  vec3 scene    = texture2D(tScene, vUv).rgb;
  vec3 bloomRaw = texture2D(tBloom, vUv).rgb;
  vec3 prev     = texture2D(tPrev,  vUv).rgb;

  float luma = dot(bloomRaw, W);
  vec3 bloom = bloomRaw * uBloomIntensity + uGlowColor * luma * uGlowIntensity;

  vec3 lit = scene + bloom;

  // Trail: take max with the decayed previous frame. max() (instead of add)
  // keeps the trail behind the bright current image without darkening it
  // and avoids the runaway brightness positive-feedback loops would cause.
  vec3 trailed = max(lit, prev * uFeedbackDecay);

  gl_FragColor = vec4(trailed, 1.0);
}
