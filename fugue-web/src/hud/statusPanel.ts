// HUD status panel. Subscribes to AppState and updates the text rows + the
// three voice lights + the left/right gesture indicators + the tier badge.

import { subscribe } from '../state.ts';
import type { TrackId } from '../types.ts';

const TRACK_LIGHT_IDS: Record<TrackId, string> = {
  violin: 'light-violin',
  lute: 'light-lute',
  organ: 'light-organ',
};

export function mountHud(): void {
  const hud = document.getElementById('hud');
  const statusEl = document.getElementById('hud-status');
  const leftEl = document.getElementById('hud-left');
  const rightEl = document.getElementById('hud-right');
  const activeEl = document.getElementById('hud-active');
  const fpsEl = document.getElementById('hud-fps');
  const tierEl = document.getElementById('hud-tier');
  const errorEl = document.getElementById('hud-error');
  if (!hud || !statusEl || !leftEl || !rightEl || !activeEl || !fpsEl || !tierEl || !errorEl) {
    return;
  }

  const lights: Partial<Record<TrackId, HTMLElement>> = {};
  for (const [t, id] of Object.entries(TRACK_LIGHT_IDS) as [TrackId, string][]) {
    const el = document.getElementById(id);
    if (el) lights[t] = el;
  }

  hud.classList.add('visible');

  subscribe((s) => {
    statusEl.textContent = s.audioStatus;

    leftEl.textContent = s.hands.left
      ? `${s.hands.left.gesture} (${s.hands.left.fingerCount})`
      : '—';
    rightEl.textContent = s.hands.right
      ? `${s.hands.right.gesture} (${s.hands.right.fingerCount})`
      : '—';

    const list = Array.from(s.activeTracks);
    activeEl.textContent = list.length > 0 ? list.join(' + ') : '—';

    fpsEl.textContent = s.fps > 0 ? s.fps.toFixed(0) : '—';
    tierEl.textContent = s.qualityTier;

    for (const t of Object.keys(TRACK_LIGHT_IDS) as TrackId[]) {
      const el = lights[t];
      if (el) el.classList.toggle('on', s.trackGains[t] === 1);
    }

    if (s.error) {
      errorEl.textContent = s.error;
      errorEl.classList.add('visible');
    } else {
      errorEl.classList.remove('visible');
    }
  });
}
