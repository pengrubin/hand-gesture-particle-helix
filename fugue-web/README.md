# Fugue Hands

Bach's three-voice fugue, played by your hands — open the page, raise fingers,
hear the counterpoint emerge. A pure browser port of a gesture-controlled
audiovisual installation: MediaPipe → Web Audio → Three.js + custom bloom.

## Quick start (local)

Requires Node ≥ 20.11 and pnpm.

```bash
cd fugue-web
pnpm install
pnpm dev            # → http://localhost:5173
```

`pnpm dev` first syncs the three Bach stems from `../data/audio/` into
`public/audio/`, and downloads MediaPipe's `hand_landmarker.task` (~7MB)
into `public/mediapipe/`. Both are idempotent.

## Use

Click **Start**, allow camera access, raise fingers:

| Gesture | Effect |
|---|---|
| 1 finger | violin |
| 2 fingers | lute |
| 3 fingers | organ |
| open hand (5) | all three voices |
| fist / no hand / 4 fingers | mute (suspends AudioContext) |

Both hands compose: left=1 + right=2 → violin + lute. The three tracks
share one sample-locked AudioContext timeline; "no hands" suspends the
whole context so hand-in resumes from the exact pause point, sample-accurate
indefinitely.

The full-screen background is a Three.js sphere driven by the live mix —
bass inflates the surface, mid adds a warm accent, high glints on the
crests. Bloom + feedback (ported from the original TouchDesigner GLSL)
layer on top of the scene each frame.

## URL flags

| Query | Effect |
|---|---|
| `?tier=low` | Force the lowest quality tier (sphere 128², camera 480×360, 2 bloom passes, DPR cap 1.0) |
| `?tier=mid` | Force the middle tier (256², 640×480, 3 bloom passes, DPR cap 1.5) |
| `?tier=high` | Force the highest tier (384², 640×480, 3 bloom passes, DPR cap 2.0) |

Without an override, the app probes `hardwareConcurrency` + `deviceMemory`
+ user-agent and persists the choice in `localStorage`. Clear it to
re-probe (DevTools → Application → Local Storage).

## Debug fallback (no camera)

`Shift + 1 / 2 / 3 / 5 / 0` simulates per-hand gesture sets. Useful when
camera permission is denied or you're testing audio sync in isolation.

## Roadmap

- [x] **M1** — audio engine + keyboard fallback
- [x] **M2** — MediaPipe hand gestures replace keyboard
- [x] **M3** — Three.js sphere driven by audio analyser
- [x] **M4** — bloom + feedback post (ported from TouchDesigner GLSL)
- [x] **M5** — performance tiers + visibility guard + deploy ready

## Deploy

See [`DEPLOY.md`](./DEPLOY.md) for Vercel (recommended) and GitHub Pages
walkthroughs.

## Architecture

See [`PLAN.md`](./PLAN.md) for the full module-by-module breakdown.

```
fugue-web/
  public/
    audio/                # ../data/audio synced here (git-ignored)
    mediapipe/            # hand_landmarker.task downloaded (git-ignored)
  scripts/
    sync-audio.mjs        # ../data/audio → public/audio
    download-model.mjs    # fetch MediaPipe model
  src/
    main.ts               # entry: Start → tier → audio → renderer → post → loop
    state.ts              # central pub/sub
    types.ts              # shared types
    config.ts             # constants
    vite-env.d.ts
    audio/
      audioEngine.ts      # 3-track sync, suspend/resume, hard 0/1 gains
      analyser.ts         # AnalyserNode → bass/mid/high bands
    gesture/
      cameraSource.ts     # getUserMedia wrapper
      handLandmarker.ts   # MediaPipe Tasks Vision wrapper (handedness flip)
      fingerCounter.ts    # pure: 21 landmarks → finger count (Python port)
      gestureMapper.ts    # pure: (left, right) → Set<TrackId>
    visual/
      renderer.ts         # Three.js renderer + scene + camera
      sphere.ts           # high-res SphereGeometry + ShaderMaterial
      postprocess.ts      # threshold → blur → composite → feedback → canvas
      qualityTier.ts      # boot-time GPU probe + URL override + persist
      shaders/
        sphere.vert.glsl
        sphere.frag.glsl
        fullscreen.vert.glsl
        bloom-threshold.frag.glsl
        bloom-blur.frag.glsl
        composite.frag.glsl
        copy.frag.glsl
    hud/
      statusPanel.ts      # audio/hands/active/lights/fps/tier/error
      cameraPreview.ts    # mirrored video + landmark overlay
    util/
      keyboardFallback.ts # Shift+number debug input
```
