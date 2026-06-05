// Minimal central state with a pub/sub. Everything reads from here.
// Writers: gesture (or keyboard), audioEngine, main. Readers: HUD, visual.

import type { AppState, TrackId } from './types.ts';

type Listener = (s: AppState) => void;

const initial: AppState = {
  hands: {},
  activeTracks: new Set<TrackId>(),
  audioStatus: 'idle',
  trackGains: { violin: 0, lute: 0, organ: 0 },
  audioBands: { bass: 0, mid: 0, high: 0 },
  fps: 0,
  qualityTier: 'mid',
  error: null,
};

let state: AppState = initial;
const listeners = new Set<Listener>();

export function getState(): AppState {
  return state;
}

export function setState(partial: Partial<AppState>): void {
  state = { ...state, ...partial };
  for (const l of listeners) l(state);
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  listener(state);
  return () => {
    listeners.delete(listener);
  };
}
