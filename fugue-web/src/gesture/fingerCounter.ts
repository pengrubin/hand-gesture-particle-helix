// Pure-function finger counter, ported verbatim from gesture_detector.py
// (commit 3b86e2a clean version).
//
// Rules (all four non-thumb fingers):
//   tipY < pipY                                       (tip is above pip in image)
//   dist2D(tip, wrist) > dist2D(mcp, wrist) * 1.1     (tip extended past mcp)
//
// Rule (thumb):
//   dist2D(tip, wrist) > dist2D(ip, wrist) * 1.1
//
// No temporal smoothing — the clean Python version is purely per-frame,
// and we keep that behaviour. If jitter becomes a problem we can add a
// 3-frame majority vote later.

import { FINGER_EXTENSION_RATIO, THUMB_EXTENSION_RATIO } from '../config.ts';
import type { GestureName, Landmark } from '../types.ts';

const WRIST = 0;
const THUMB_IP = 3;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_PIP = 6;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_PIP = 10;
const MIDDLE_TIP = 12;
const RING_MCP = 13;
const RING_PIP = 14;
const RING_TIP = 16;
const PINKY_MCP = 17;
const PINKY_PIP = 18;
const PINKY_TIP = 20;

function dist2D(a: Landmark, b: Landmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export interface CountResult {
  count: number;
  thumbUp: boolean;
  indexUp: boolean;
  middleUp: boolean;
  ringUp: boolean;
  pinkyUp: boolean;
}

export function countFingers(landmarks: ReadonlyArray<Landmark>): CountResult {
  const zero: CountResult = {
    count: 0,
    thumbUp: false,
    indexUp: false,
    middleUp: false,
    ringUp: false,
    pinkyUp: false,
  };
  if (landmarks.length < 21) return zero;

  const wrist = landmarks[WRIST];
  const thumbIp = landmarks[THUMB_IP];
  const thumbTip = landmarks[THUMB_TIP];
  if (!wrist || !thumbIp || !thumbTip) return zero;

  const thumbUp =
    dist2D(thumbTip, wrist) > dist2D(thumbIp, wrist) * THUMB_EXTENSION_RATIO;

  const fingerCheck = (tipIdx: number, pipIdx: number, mcpIdx: number): boolean => {
    const tip = landmarks[tipIdx];
    const pip = landmarks[pipIdx];
    const mcp = landmarks[mcpIdx];
    if (!tip || !pip || !mcp) return false;
    const basicUp = tip.y < pip.y;
    const distUp = dist2D(tip, wrist) > dist2D(mcp, wrist) * FINGER_EXTENSION_RATIO;
    return basicUp && distUp;
  };

  const indexUp = fingerCheck(INDEX_TIP, INDEX_PIP, INDEX_MCP);
  const middleUp = fingerCheck(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP);
  const ringUp = fingerCheck(RING_TIP, RING_PIP, RING_MCP);
  const pinkyUp = fingerCheck(PINKY_TIP, PINKY_PIP, PINKY_MCP);

  const count =
    (thumbUp ? 1 : 0) +
    (indexUp ? 1 : 0) +
    (middleUp ? 1 : 0) +
    (ringUp ? 1 : 0) +
    (pinkyUp ? 1 : 0);

  return { count, thumbUp, indexUp, middleUp, ringUp, pinkyUp };
}

export function gestureFromCount(count: number): GestureName {
  switch (count) {
    case 1:
      return 'one';
    case 2:
      return 'two';
    case 3:
      return 'three';
    case 5:
      return 'open_hand';
    default:
      return 'other';
  }
}
