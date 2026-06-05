// Debug fallback: Shift + 1/2/3/5/0/O simulates the gesture mapper output.
//
// Gated behind Shift so it can't clash with text input or accidentally fire
// while the user moves their hands. Uses `e.code` so it works regardless of
// keyboard layout or modifier-induced character substitution.

import { applyActive } from '../audio/audioEngine.ts';
import { ALL_TRACKS, type TrackId } from '../types.ts';

const CODE_MAP: Record<string, ReadonlySet<TrackId>> = {
  Digit1: new Set<TrackId>(['violin']),
  Digit2: new Set<TrackId>(['lute']),
  Digit3: new Set<TrackId>(['organ']),
  Digit5: new Set<TrackId>(ALL_TRACKS),
  KeyO: new Set<TrackId>(ALL_TRACKS),
  Digit0: new Set<TrackId>(),
};

export function attachKeyboardFallback(): void {
  document.addEventListener('keydown', (e) => {
    if (!e.shiftKey || e.repeat) return;
    const target = CODE_MAP[e.code];
    if (target === undefined) return;
    e.preventDefault();
    applyActive(target);
  });
}
