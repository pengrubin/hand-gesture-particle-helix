// Frequency-band sampler on top of an AnalyserNode. Returns three numbers
// in [0, 1] — bass, mid, high — corresponding to the same loose buckets the
// TouchDesigner audioAnalysis used to drive the visual.
//
// With fftSize = 1024 and 44.1 kHz sample rate, each bin is ~43 Hz wide.
// Buckets:
//   bass:  bins   0..8      → ~0   .. 344 Hz
//   mid:   bins   8..64     → ~344 .. 2.8 kHz
//   high:  bins  64..256    → ~2.8 .. 11 kHz
//
// Anything above ~11 kHz is mostly air/noise for this material so we ignore
// the upper third.

import type { AudioBands } from '../types.ts';

const BASS_END = 8;
const MID_END = 64;
const HIGH_END = 256;

export function createSampler(analyser: AnalyserNode): () => AudioBands {
  const bins = new Uint8Array(analyser.frequencyBinCount);

  return function sample(): AudioBands {
    analyser.getByteFrequencyData(bins);

    let bass = 0;
    let mid = 0;
    let high = 0;

    for (let i = 0; i < BASS_END; i++) bass += bins[i] ?? 0;
    for (let i = BASS_END; i < MID_END; i++) mid += bins[i] ?? 0;
    for (let i = MID_END; i < HIGH_END; i++) high += bins[i] ?? 0;

    return {
      bass: bass / (BASS_END * 255),
      mid: mid / ((MID_END - BASS_END) * 255),
      high: high / ((HIGH_END - MID_END) * 255),
    };
  };
}
