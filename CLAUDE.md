# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Three generations of the same idea, a Bach "Fugue in G Trio" (violin / lute / organ stems) played by hand gestures, live side by side:

| Layer | Where | Status |
|---|---|---|
| **fugue-web** browser port (TypeScript, Vite, Three.js, MediaPipe Tasks Vision, Web Audio) | `fugue-web/` | Active. All recent commits touch this. Live at https://hand.pengrubin.com/ (Cloudflare Workers). Intentionally minimal: gesture → voice switching only, no heavy GPU patterns, no EEG. |
| Python desktop app (OpenCV + MediaPipe + pygame + PyOpenGL) | repo root `*.py`, `data/` | Maintained but older. `python run.py`. |
| E_Major full-body pose → 11-track orchestra | `projects/e_major/` | Self-contained sub-project with its own `run.py`, requirements and `hand_gesture_detector.py`. Do not confuse with the root modules of the same name. |

TouchDesigner project files (`*.toe`, `*.tox`, `*.toe.dir/`) exist locally but are gitignored. Docs for that integration are in `docs/touchdesigner/`.

Commit messages are conventional-commit style with a `(fugue-web)` scope where relevant, written in Chinese. Comments in the codebase are mixed Chinese and English.

## Commands

### fugue-web (Node >= 20.11, pnpm 9)

```bash
cd fugue-web
pnpm install
pnpm dev          # prepare-assets, then Vite on http://localhost:5173
pnpm build        # prepare-assets, tsc --noEmit, vite build → dist/
pnpm typecheck    # tsc --noEmit only
pnpm preview      # serve dist/
```

`prepare-assets` runs two idempotent scripts: `scripts/sync-audio.mjs` copies the three stems from `../data/audio/` into `public/audio/` (falls back to the committed copies when `../data/audio` is absent, as in CI), and `scripts/download-model.mjs` fetches `hand_landmarker.task` (~7 MB) into `public/mediapipe/`.

There is no test runner and no linter. `tsc --noEmit` under strict mode (with `noUncheckedIndexedAccess`, `noUnusedLocals`, `noUnusedParameters`) is the only gate, and `pnpm build` runs it.

Deploy targets, all static:
- Cloudflare Workers via `wrangler.jsonc` (worker name `hand-gesture-particle-helix`, assets from `./dist`, SPA fallback). Production builds trigger from `main`; bumping `version` in `package.json` is the established way to force a rebuild.
- Vercel: import the repo with root directory `fugue-web`.
- GitHub Pages: set `VITE_BASE_PATH=/<repo>/` at build time. See `fugue-web/DEPLOY.md`.

Runtime flags: `?tier=low|mid|high` forces a quality tier for one visit. `Shift+1/2/3/5/0` simulates gestures without a camera.

### Python desktop app

```bash
pip install -r requirements.txt              # or requirements_cross_platform.txt on Intel Mac / Windows
python run.py                                # dependency + camera check, then main_app.py
python main_app.py                           # skip the checks
python main_prototype.py [mapping|auto]      # parametric-curve prototype, see docs/guides/PROTOTYPE_USAGE.md
python projects/e_major/run.py               # E_Major sub-project
```

There is no pytest or flake8 configuration.

Audio stems live in `data/audio/` with fixed filenames (`Fugue in G Trio violin-Violin.mp3`, `Fugue in G Trio-Tenor_Lute.mp3`, `Fugue in G Trio Organ-Organ.mp3`). `*.mp3` is gitignored globally; the only committed copies are `fugue-web/public/audio/*.mp3`, re-allowed by `fugue-web/.gitignore` so CI builds succeed.

## Architecture: fugue-web

Single-page app, single `requestAnimationFrame` loop, no framework. Read `fugue-web/PLAN.md` for the module-by-module rationale.

Data flow per frame (`src/main.ts`):

```
camera video → handLandmarker.detectFrame → fingerCounter.countFingers → gestureFromCount
            → gestureMapper.mapHandsToTracks(left, right) → audioEngine.applyActive(Set<TrackId>)
masterAnalyser → analyser.createSampler → {bass, mid, high} → sphere shader uniforms + post uniforms
            → PostProcessor.render (threshold → blur → composite → feedback → canvas)
```

- `src/state.ts` is the one pub/sub store. Ownership is strict: the gesture loop (or keyboard fallback) writes `hands`/`activeTracks`; the audio engine writes `audioStatus`/`trackGains`/`audioBands`; HUD and visuals only read.
- `src/audio/audioEngine.ts` contract: three looping `AudioBufferSourceNode`s started at the same `currentTime`, gains set to a hard 0 or 1 with no ramp, and an empty active set suspends the whole `AudioContext` while a non-empty one resumes it. Sources are never stopped or recreated, which is what keeps the three voices sample-locked across pauses. `start()` must run inside a user click for iOS Safari.
- Gesture rule is locked: 1 finger = violin, 2 = lute, 3 = organ, open hand = all three, anything else (fist, 4, no hand) = nothing. Two hands are a plain union.
- `src/gesture/fingerCounter.ts` is a verbatim port of the finger test in `gesture_detector.py` (tip above PIP and tip-to-wrist > 1.1 × MCP-to-wrist; thumb uses the IP joint). The 1.1 ratios live in `src/config.ts`. If you change the algorithm on one side, change the other.
- `src/gesture/handLandmarker.ts` passes MediaPipe's handedness label through unchanged. A selfie flip was removed because it inverted left/right on iOS WebKit. Do not reintroduce it. The WASM runtime URL pins `@mediapipe/tasks-vision@0.10.18` on jsdelivr and must match the version in `package.json`.
- `src/visual/qualityTier.ts` picks low/mid/high from URL override, then `localStorage` (`fugue-tier`), then hardware hints; mobile user agents always get low. The tier sets sphere segments, camera resolution, DPR cap and bloom pass count.
- `src/hud/cameraPreview.ts` sizes the PIP to the real video aspect ratio so the landmark overlay lines up on phone front cameras.
- `visibilitychange` mutes everything when the tab hides; resuming waits for the next gesture.

## Architecture: Python desktop app

Flat modules at the repo root.

- `main_app.py` wires `GestureDetector` (`gesture_detector.py`), `RenderEngine` (`render_engine.py`), `ParticleSphereSystem` (`particle_sphere_system.py`, the helix and wave shapes), `RealisticAudioManager` (`realistic_audio_manager.py`) and `AudioSpectrumAnalyzer`. It installs SIGINT/SIGTERM and `atexit` handlers to force camera release; keep that when touching startup or shutdown.
- `gesture_detector.py` is the unified detector (stabilizer, cache, conductor analysis, region bounds) and ends with module-level functions (`initialize_detector`, `process_camera_frame`, `get_gesture_data`, ...) that are the TouchDesigner entry points.
- `realistic_audio_manager.py` prefers `data/audio/` and falls back to the script directory. Pause/resume is virtual-position based because pygame cannot seek precisely; `docs/development/BUGFIX_SUMMARY.md` explains the strategy.
- The parametric-curve layer (`gesture_parametric_bridge.py`, `gesture_radius_mapper.py`, `parametric_equation_renderer.py`, `td_parametric_integration.py`, `td_gpu_*.py`, `parametric_curves.glsl`) and the trajectory layer (`trajectory_*.py`) are separate experiments driven by `main_prototype.py` and `live_gesture_parametric.py`, not by `main_app.py`.
- `tools/` holds standalone scripts: `midi/` (pretty_midi/trimesh-based MIDI → 3D model generators, outputs in `data/models/`), `osc/` (Python → TouchDesigner OSC demo) and `debug/` (camera diagnostics).

## Docs worth opening first

- `fugue-web/README.md`, `fugue-web/PLAN.md`, `fugue-web/DEPLOY.md`
- `docs/README.md` is the index for the Python side; `docs/reference/API_REFERENCE.md` and `docs/reference/PARAMETER_REFERENCE.md` for parameter meanings.
- `projects/e_major/README.md` and `projects/e_major/TECHNICAL_DESIGN.md` for the pose-driven sub-project.
