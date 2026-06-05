// Shared types. Kept tiny so every module pulls from one source of truth.

export type TrackId = 'violin' | 'lute' | 'organ';

export const ALL_TRACKS: readonly TrackId[] = ['violin', 'lute', 'organ'] as const;

export type GestureName = 'one' | 'two' | 'three' | 'open_hand' | 'other';

export type Handedness = 'Left' | 'Right';

export type AudioStatus = 'idle' | 'loading' | 'running' | 'suspended' | 'error';

export type QualityTier = 'low' | 'mid' | 'high';

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface HandState {
  handedness: Handedness;
  fingerCount: number;
  gesture: GestureName;
  landmarks: ReadonlyArray<Landmark>;
}

export interface AudioBands {
  bass: number;
  mid: number;
  high: number;
}

export interface AppState {
  // input — written by gesture loop (or keyboard fallback)
  hands: { left?: HandState; right?: HandState };

  // derived — written by gestureMapper
  activeTracks: ReadonlySet<TrackId>;

  // audio — written by audioEngine and the analyser sampler
  audioStatus: AudioStatus;
  trackGains: Record<TrackId, 0 | 1>;
  audioBands: AudioBands;

  // system
  fps: number;
  qualityTier: QualityTier;
  error: string | null;
}
