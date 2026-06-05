// Three-track sample-locked audio engine.
//
// Design contract:
//   - All three AudioBufferSourceNodes are created and start()-ed at the
//     same currentTime + AUDIO_START_OFFSET_S. They loop forever.
//   - Volume is hard 0/1 — gain.value assigned directly, no ramp.
//   - "No active tracks" -> suspend the entire AudioContext (freezes the
//     sample clock for every track simultaneously).
//   - "Any active track" while suspended -> resume the context. Sample
//     alignment is preserved by design; we never stop/restart sources.
//
// M3+: a masterGain sums the per-track gains; both `destination` and an
// AnalyserNode tap off the master, so the visualiser sees post-gain audio.

import {
  AUDIO_FILES,
  AUDIO_SAMPLE_RATE,
  AUDIO_START_OFFSET_S,
} from '../config.ts';
import { setState } from '../state.ts';
import { ALL_TRACKS, type TrackId } from '../types.ts';

interface TrackNode {
  source: AudioBufferSourceNode;
  gain: GainNode;
}

let ctx: AudioContext | null = null;
let tracks: Record<TrackId, TrackNode> | null = null;
let masterAnalyser: AnalyserNode | null = null;
let started = false;

async function loadBuffer(audioCtx: AudioContext, url: string): Promise<AudioBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const arrayBuffer = await res.arrayBuffer();
  return audioCtx.decodeAudioData(arrayBuffer);
}

/**
 * Initialize the engine. Must be called from within a user-gesture handler
 * (click / touch) so iOS Safari permits the AudioContext to start.
 *
 * Throws on fetch/decode failure; caller is responsible for surfacing.
 */
export async function start(): Promise<void> {
  if (started) return;

  setState({ audioStatus: 'loading' });

  ctx = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });

  // Decode all three in parallel so first paint of audio is as snappy as
  // possible.
  const loaded = await Promise.all(
    ALL_TRACKS.map(async (t) => {
      const buf = await loadBuffer(ctx!, AUDIO_FILES[t]);
      return [t, buf] as const;
    }),
  );

  // Master sum + analyser tap. Both destination and analyser hear the same
  // post-gain mix, so the visualiser can react to whatever's currently
  // audible.
  const master = ctx.createGain();
  master.gain.value = 1;

  masterAnalyser = ctx.createAnalyser();
  masterAnalyser.fftSize = 1024;
  masterAnalyser.smoothingTimeConstant = 0.7;

  master.connect(ctx.destination);
  master.connect(masterAnalyser);

  // Schedule all three sources to start at the same currentTime. From this
  // point on the sources share a sample clock and cannot drift.
  const startAt = ctx.currentTime + AUDIO_START_OFFSET_S;
  const nodes = {} as Record<TrackId, TrackNode>;

  for (const [t, buffer] of loaded) {
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.loop = true;

    const gain = ctx.createGain();
    gain.gain.value = 0;

    source.connect(gain).connect(master);
    source.start(startAt);

    nodes[t] = { source, gain };
  }

  tracks = nodes;
  started = true;

  setState({ audioStatus: 'running' });
}

/**
 * Apply a desired active set. Hard 0/1 gains, suspend on empty, resume on
 * non-empty. Safe to call at any time post-start().
 */
export function applyActive(activeTracks: ReadonlySet<TrackId>): void {
  if (!ctx || !tracks) return;

  const gains: Record<TrackId, 0 | 1> = { violin: 0, lute: 0, organ: 0 };
  for (const t of ALL_TRACKS) {
    const v: 0 | 1 = activeTracks.has(t) ? 1 : 0;
    tracks[t].gain.gain.value = v;
    gains[t] = v;
  }

  const wantRunning = activeTracks.size > 0;

  if (!wantRunning && ctx.state === 'running') {
    void ctx.suspend().then(() => {
      setState({
        audioStatus: 'suspended',
        activeTracks: new Set(activeTracks),
        trackGains: gains,
      });
    });
    return;
  }

  if (wantRunning && ctx.state === 'suspended') {
    void ctx.resume().then(() => {
      setState({
        audioStatus: 'running',
        activeTracks: new Set(activeTracks),
        trackGains: gains,
      });
    });
    return;
  }

  setState({
    activeTracks: new Set(activeTracks),
    trackGains: gains,
  });
}

/**
 * Returns the master AnalyserNode, or null if start() hasn't completed yet.
 * The node lives on the same AudioContext as everything else; suspending
 * the context freezes the analyser too, so during 'suspended' the visualiser
 * will see frozen-but-valid spectrum data from the last running frame.
 */
export function getAnalyser(): AnalyserNode | null {
  return masterAnalyser;
}
