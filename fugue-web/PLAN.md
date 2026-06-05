# PLAN.md — Fugue Hands web port

Browser implementation of the gesture-controlled Bach three-voice fugue
visualization. Replaces the TouchDesigner + Python + BlackHole stack with a
single static web app.

## 1. Project metadata

- **Location**: `/Users/hongweipeng/hand-gesture-particle-helix/fugue-web/`
- **npm package**: `fugue-hands`
- **Node**: `>=20.11.0`
- **Package manager**: `pnpm@9`
- **Tagline**: Bach's three-voice fugue, played by your hands.

## 2. Directory tree

```
fugue-web/
  index.html                       # single page entry, mount + Start button
  package.json
  tsconfig.json                    # strict + ES2022 + bundler resolution
  vite.config.ts
  .nvmrc / .gitignore
  README.md
  public/
    audio/                         # synced from ../data/audio/ (git-ignored)
    mediapipe/hand_landmarker.task # (M2) self-hosted model weights ~7MB
  src/
    main.ts                        # entry: DOM mount, Start button, boot
    state.ts                       # central AppState singleton + pub/sub
    types.ts                       # global types
    config.ts                      # constants: audio paths, thresholds, tiers
    audio/
      audioEngine.ts               # 3-track sync, suspend/resume, gain switching
      analyser.ts                  # (M3) AnalyserNode → bass/mid/high
    gesture/
      cameraSource.ts              # (M2) getUserMedia 640x480
      handLandmarker.ts            # (M2) MediaPipe Tasks Vision wrapper
      fingerCounter.ts             # (M2) pure: landmarks → fingerCount + handedness
      gestureMapper.ts             # (M2) pure: (left,right) → Set<TrackId>
    visual/
      renderer.ts                  # (M3) Three.js setup
      sphere.ts                    # (M3) high-res sphere + vertex shader
      shaders/
        sphere.vert.glsl
        sphere.frag.glsl
        bloom.frag.glsl            # (M4) port of TD luminance + glow
        feedback.frag.glsl         # (M4) RGBA delay trail
      postprocess.ts               # (M4) EffectComposer chain
      qualityTier.ts               # (M5) GPU probing → low/mid/high
    hud/
      hudRoot.ts                   # bottom-right PIP container
      cameraPreview.ts             # (M2) mirrored <video> + landmark overlay
      statusPanel.ts               # gesture, active tracks, lights, fps
    util/
      fpsMeter.ts
      keyboardFallback.ts          # 1/2/3/5/0 simulate gestures (no camera)
      perfOverlay.ts               # (M5) Ctrl+P frametime / triangle count
  scripts/
    sync-audio.mjs                 # copy ../data/audio/*.mp3 → public/audio/
```

## 3. End-to-end data flow

```
[Camera 640×480 @30fps]                    (M2+)
   │ ImageBitmap / VideoFrame
   ▼
[handLandmarker.detectForVideo()]          (M2+)
   │ HandLandmarkerResult
   ▼
[fingerCounter] each hand → fingerCount, gesture
   │
   ▼
[gestureMapper] (left, right) → activeTracks: Set<TrackId>
   │
   ├─► [AppState.activeTracks] (write, triggers subscribers)
   │
   ▼
[audioEngine.applyActive(activeTracks)]
   ├─► each track.gainNode.gain.value = active ? 1 : 0   (hard, no ramp)
   └─► size>0 && ctx.suspended → ctx.resume()
       size===0 && ctx.running → ctx.suspend()

[masterAnalyser] ← combined gains                          (M3+)
   │ getByteFrequencyData() → bass/mid/high
   ▼
[AppState.audioBands]
   │
   ▼ each frame
[sphere shader uniforms]                                   (M3+)
   ▼
[Scene render → bloom pass → feedback pass → canvas]       (M4+)

[HUD] subscribes to AppState, updates DOM ~50ms throttle
```

The only writer for `hands`/`activeTracks` is the gesture pipeline. The only
writer for `audioStatus`/`trackGains`/`audioBands` is the audio engine. The
visual and HUD layers are read-only consumers.

## 4. Five milestones

### M1 — skeleton + 3-track sync audio (keyboard driven)
- `pnpm dev` opens a black page with a Start button. Click it.
- Keyboard 1/2/3/5/0 drives the three voices through suspend/resume.
- HUD shows audio status, active set, three voice lights, fps.
- **Estimate**: 1 day.

### M2 — MediaPipe hands + finger counter + camera HUD
- getUserMedia 640×480, MediaPipe Tasks Vision in browser.
- Port the Python algorithm verbatim: `tipY < pipY && dist(tip,wrist)/dist(mcp,wrist) > 1.1`.
- Thumb uses ip ratio.
- Camera preview + landmark overlay in the bottom-right PIP.
- Keyboard fallback gated behind Shift to keep it as a debug aid.
- **Estimate**: 1 day.

### M3 — Three.js sphere + audio analyser
- Sphere geometry, vertex shader noise displacement, uniforms driven by
  AnalyserNode bass/mid/high.
- Full-screen canvas, PIP HUD on top.
- **Estimate**: 1 day.

### M4 — bloom + feedback post
- Port the TouchDesigner bloom GLSL (luminance + threshold + glow color +
  ramp lookup) to a `postprocessing` EffectPass.
- Add a feedback pass (double-buffered FBO).
- **Estimate**: 1–2 days.

### M5 — performance tiers + deploy
- Quality probe at boot, three tiers, persist in localStorage.
- iOS / mobile fallback (low tier forced).
- `vercel.json` + GitHub Action for Pages backup.
- **Estimate**: 1–2 days.

## 5. Audio engine design (M1 core)

```ts
// Single AudioContext, three sources, all started at the same currentTime.
// Loop forever. gain is the only mutable knob.

ctx = new AudioContext({ sampleRate: 44100 });
const buffers = await Promise.all(tracks.map(load));
const t0 = ctx.currentTime + 0.1;
for each track t:
  source = ctx.createBufferSource()
  source.buffer = buffers[t]; source.loop = true
  gain = ctx.createGain(); gain.gain.value = 0
  source.connect(gain).connect(ctx.destination)
  source.start(t0)

applyActive(active: Set<TrackId>):
  for each track t:
    track.gain.gain.value = active.has(t) ? 1 : 0    // hard, no ramp
  if active.size === 0 && ctx.state === 'running': ctx.suspend()
  if active.size > 0  && ctx.state === 'suspended': ctx.resume()
```

Why this works:
- All three sources live on the same sample clock. Once started together
  they cannot drift relative to each other; `suspend()` freezes the entire
  context, `resume()` continues from the same sample.
- `gain.value` is the immediate setter (no scheduling), so 0/1 transitions
  are sample-aligned to the next render quantum (≤128 samples ≈ 3ms).
- We never `stop()` the sources during normal operation, so we never need
  to recreate them and never face start-time alignment issues.

## 6. Gesture rule (locked)

```
per-hand mapping:
  one       → {violin}
  two       → {lute}
  three     → {organ}
  open_hand → {violin, lute, organ}
  fist/other → {}

union(left, right) → activeTracks
```

Volume is hard 0/1 — no fade. No-hands → suspend ctx, hand reappears →
resume from exact pause point.

## 7. Risks & mitigations (top 5)

| Risk | Mitigation |
|---|---|
| iOS Safari blocks ctx.start without user gesture | Start button in main user click path; resume on first input |
| MediaPipe fps too low on weak hardware | low tier: frame skip every other frame; camera 320×240 |
| HTTPS required for camera + MediaPipe model fetch | localhost is exempt; Vercel/Pages provide HTTPS |
| Hidden tab → rAF stops but ctx still draws power | `visibilitychange` listener: suspend on hide |
| Three-track drift bug | suspend/resume + same start time = sample-locked by design; add 5s telemetry as belt-and-braces |

## 8. Local dev

```bash
cd fugue-web
pnpm install
pnpm dev             # http://localhost:5173
```

Optional flags (URL query):
- `?tier=low|mid|high` — force quality tier (M5)
- `?debug=hands` — show finger ratio overlay (M2)

Keyboard always works as fallback for sanity testing.

## 9. Deploy

### Vercel
- Project settings: Root Directory = `fugue-web`, Framework = Vite.
- No `vercel.json` needed.

### GitHub Pages (backup)
- `vite.config.ts` with `base: '/hand-gesture-particle-helix/'` under
  `GH_PAGES=1`.
- Action: setup pnpm → `cd fugue-web && pnpm i && GH_PAGES=1 pnpm build` →
  publish `fugue-web/dist`.

Both yield HTTPS out of the box.
