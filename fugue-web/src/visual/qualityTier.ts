// Quality tier detection + per-tier settings.
//
// The tier picks: sphere mesh density, camera resolution, renderer DPR cap,
// and how many bloom-blur passes the post-process runs. Detected at boot
// from a small set of signals — URL override > localStorage > UA/hardware
// hints — and cached so repeat visits don't re-probe.

import type { QualityTier } from '../types.ts';

export interface QualitySettings {
  sphereSegments: number;
  cameraWidth: number;
  cameraHeight: number;
  pixelRatioCap: number;
  bloomBlurOffsets: number[];
}

const SETTINGS: Record<QualityTier, QualitySettings> = {
  low: {
    sphereSegments: 128,
    cameraWidth: 480,
    cameraHeight: 360,
    pixelRatioCap: 1.0,
    bloomBlurOffsets: [2.0, 4.0], // 2 passes instead of 3
  },
  mid: {
    sphereSegments: 256,
    cameraWidth: 640,
    cameraHeight: 480,
    pixelRatioCap: 1.5,
    bloomBlurOffsets: [1.5, 3.0, 5.0],
  },
  high: {
    sphereSegments: 384,
    cameraWidth: 640,
    cameraHeight: 480,
    pixelRatioCap: 2.0,
    bloomBlurOffsets: [1.5, 3.0, 5.0],
  },
};

const STORAGE_KEY = 'fugue-tier';
const ALL: ReadonlyArray<QualityTier> = ['low', 'mid', 'high'];

function isTier(v: string | null): v is QualityTier {
  return v !== null && (ALL as ReadonlyArray<string>).includes(v);
}

/**
 * Pick a tier. Precedence:
 *   1. ?tier=low|mid|high in the URL (always wins; for testing)
 *   2. Previously persisted value in localStorage
 *   3. Heuristic from navigator hints
 *
 * URL choice doesn't get persisted — it's intentionally per-visit so a
 * tester can force a tier without affecting normal users.
 */
export function detectTier(): QualityTier {
  const url = new URLSearchParams(window.location.search).get('tier');
  if (isTier(url)) return url;

  try {
    const cached = localStorage.getItem(STORAGE_KEY);
    if (isTier(cached)) return cached;
  } catch {
    // localStorage may be blocked (Safari private mode etc.) — ignore.
  }

  const ua = navigator.userAgent;
  const isMobile = /iPhone|iPad|iPod|Android/i.test(ua);
  if (isMobile) return 'low';

  const cores = navigator.hardwareConcurrency ?? 4;
  // deviceMemory is non-standard but supported in Chromium / many browsers.
  const memGb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;

  if (cores >= 8 && memGb >= 8) return 'high';
  if (cores <= 4 || memGb <= 2) return 'low';
  return 'mid';
}

export function getSettings(tier: QualityTier): QualitySettings {
  return SETTINGS[tier];
}

export function rememberTier(tier: QualityTier): void {
  try {
    localStorage.setItem(STORAGE_KEY, tier);
  } catch {
    // see detectTier()
  }
}
