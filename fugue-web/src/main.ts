// Entry point. Wires the Start button, boots audio inside the user-gesture
// handler (iOS Safari requirement), then builds renderer + post-process,
// attaches gesture + HUD + keyboard, and runs a single rAF loop that drives
// everything.
//
// If camera or MediaPipe fail, the page still works via the Shift-gated
// keyboard fallback — we surface the error in the HUD.

import { applyActive, getAnalyser, start } from './audio/audioEngine.ts';
import { createSampler } from './audio/analyser.ts';
import { startCamera } from './gesture/cameraSource.ts';
import { countFingers, gestureFromCount } from './gesture/fingerCounter.ts';
import { mapHandsToTracks } from './gesture/gestureMapper.ts';
import { createHandLandmarker, detectFrame } from './gesture/handLandmarker.ts';
import { mountCameraPreview } from './hud/cameraPreview.ts';
import { mountHud } from './hud/statusPanel.ts';
import { getState, setState } from './state.ts';
import type { HandState, TrackId } from './types.ts';
import { attachKeyboardFallback } from './util/keyboardFallback.ts';
import { PostProcessor } from './visual/postprocess.ts';
import {
  detectTier,
  getSettings,
  rememberTier,
  type QualitySettings,
} from './visual/qualityTier.ts';
import { createRenderer } from './visual/renderer.ts';
import { createSphere, type SphereUniforms } from './visual/sphere.ts';

interface LoopContext {
  post: PostProcessor;
  sphereUniforms: SphereUniforms;
  sample: () => { bass: number; mid: number; high: number };
}

interface GestureContext {
  landmarker: Awaited<ReturnType<typeof createHandLandmarker>>;
  video: HTMLVideoElement;
  lastVideoTime: number;
}

let gestureCtx: GestureContext | null = null;

async function boot(): Promise<void> {
  const btn = document.getElementById('start-button') as HTMLButtonElement | null;
  if (!btn) return;

  btn.disabled = true;
  btn.textContent = 'Loading audio…';

  try {
    await start();
  } catch (err) {
    handleFatal(btn, 'Audio start failed', err);
    return;
  }

  document.getElementById('app')?.remove();

  // Pick the GPU tier first so every system below knows how heavy it can go.
  const tier = detectTier();
  rememberTier(tier);
  const settings = getSettings(tier);
  setState({ qualityTier: tier });

  mountHud();
  attachKeyboardFallback();
  attachVisibilityGuard();

  const renderer = createRenderer(settings.pixelRatioCap);
  const sphere = createSphere(settings.sphereSegments);
  renderer.scene.add(sphere.mesh);

  const post = new PostProcessor(renderer.renderer, renderer.scene, renderer.camera, {
    blurOffsets: settings.bloomBlurOffsets,
  });

  const analyser = getAnalyser();
  if (!analyser) {
    handleFatal(btn, 'Analyser unavailable', new Error('no master analyser'));
    return;
  }
  const sample = createSampler(analyser);

  startMainLoop({ post, sphereUniforms: sphere.uniforms, sample });

  void bootGesture(settings);
}

async function bootGesture(settings: QualitySettings): Promise<void> {
  try {
    setState({ error: null });
    const cam = await startCamera(settings.cameraWidth, settings.cameraHeight);
    const landmarker = await createHandLandmarker();
    mountCameraPreview(cam.video);
    gestureCtx = { landmarker, video: cam.video, lastVideoTime: -1 };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setState({ error: `Gesture disabled: ${msg}` });
    console.error('Gesture boot failed:', err);
  }
}

function startMainLoop(ctx: LoopContext): void {
  let last = performance.now();
  let accumMs = 0;
  let frames = 0;
  // Visual "active time" — only advances while the AudioContext is running.
  // Freezing this on suspend means the sphere's noise field stops evolving
  // and it holds its current pose.
  let activeTime = 0;

  function tick(now: number): void {
    const dt = now - last;
    accumMs += dt;
    last = now;
    frames += 1;
    if (accumMs >= 500) {
      setState({ fps: (frames * 1000) / accumMs });
      accumMs = 0;
      frames = 0;
    }

    if (gestureCtx) processGestureFrame(gestureCtx, now);

    const isSuspended = getState().audioStatus === 'suspended';

    let bands;
    if (isSuspended) {
      bands = { bass: 0, mid: 0, high: 0 };
    } else {
      bands = ctx.sample();
      activeTime += dt;
    }
    setState({ audioBands: bands });

    // Sphere uniforms (vertex displacement / colour accents).
    ctx.sphereUniforms.uTime.value = activeTime / 1000;
    ctx.sphereUniforms.uBass.value = bands.bass;
    ctx.sphereUniforms.uMid.value = bands.mid;
    ctx.sphereUniforms.uHigh.value = bands.high;

    // Post uniforms (bloom and feedback knobs ride the audio).
    ctx.post.uniforms.bloomIntensity.value = 0.7 + bands.high * 1.5;
    ctx.post.uniforms.glowIntensity.value = 0.25 + bands.mid * 0.55;
    ctx.post.uniforms.feedbackDecay.value = 0.86 + bands.bass * 0.08;

    ctx.post.render();

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

function processGestureFrame(g: GestureContext, now: number): void {
  if (g.video.readyState < 2) return;
  if (g.video.currentTime === g.lastVideoTime) return;
  g.lastVideoTime = g.video.currentTime;

  try {
    const detections = detectFrame(g.landmarker, g.video, now);
    const hands: { left?: HandState; right?: HandState } = {};
    for (const det of detections) {
      const r = countFingers(det.landmarks);
      const gesture = gestureFromCount(r.count);
      const hs: HandState = {
        handedness: det.handedness,
        fingerCount: r.count,
        gesture,
        landmarks: det.landmarks,
      };
      if (det.handedness === 'Left') hands.left = hs;
      else hands.right = hs;
    }
    setState({ hands });
    const active = mapHandsToTracks(hands.left?.gesture, hands.right?.gesture);
    applyActive(active);
  } catch (err) {
    console.error('detect frame error:', err);
  }
}

/**
 * When the tab becomes hidden, force-mute everything so the audio doesn't
 * keep playing in the background. Resuming is left to the next gesture so
 * the user is in control.
 */
function attachVisibilityGuard(): void {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      applyActive(new Set<TrackId>());
    }
  });
}

function handleFatal(btn: HTMLButtonElement, label: string, err: unknown): void {
  const msg = err instanceof Error ? err.message : String(err);
  setState({ audioStatus: 'error', error: `${label}: ${msg}` });
  btn.disabled = false;
  btn.textContent = `${label} — click to retry`;
  console.error(label, err);
}

document.getElementById('start-button')?.addEventListener('click', () => {
  void boot();
});
