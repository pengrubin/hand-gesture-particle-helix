// Pure function: per-hand gesture names → union set of active tracks.
//
// Per-hand mapping (locked from the original Python design):
//   one       → {violin}
//   two       → {lute}
//   three     → {organ}
//   open_hand → {violin, lute, organ}        (full ensemble shortcut)
//   other     → {}                           (fist / unknown / 4 fingers)
//
// Both hands: simple union. No precedence, no left-vs-right semantics.

import { ALL_TRACKS, type GestureName, type TrackId } from '../types.ts';

const GESTURE_TO_TRACKS: Record<GestureName, ReadonlyArray<TrackId>> = {
  one: ['violin'],
  two: ['lute'],
  three: ['organ'],
  open_hand: ALL_TRACKS,
  other: [],
};

export function mapHandsToTracks(
  left: GestureName | undefined,
  right: GestureName | undefined,
): Set<TrackId> {
  const out = new Set<TrackId>();
  for (const g of [left, right]) {
    if (!g) continue;
    for (const t of GESTURE_TO_TRACKS[g]) out.add(t);
  }
  return out;
}
