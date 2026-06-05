// Static configuration. All "magic numbers" live here, exported as consts.

import type { TrackId } from './types.ts';

// Audio file URLs served from public/audio/.
export const AUDIO_FILES: Record<TrackId, string> = {
  violin: '/audio/violin.mp3',
  lute: '/audio/lute.mp3',
  organ: '/audio/organ.mp3',
};

// Finger-extension thresholds (ported from gesture_detector.py).
// Kept here so they're easy to tweak without editing the algorithm.
export const THUMB_EXTENSION_RATIO = 1.1;
export const FINGER_EXTENSION_RATIO = 1.1;

// Audio engine
export const AUDIO_SAMPLE_RATE = 44100;
// Tiny lead time so all three buffer sources line up on the first sample
// boundary after currentTime; 100 ms is comfortable for decoding overhead.
export const AUDIO_START_OFFSET_S = 0.1;
