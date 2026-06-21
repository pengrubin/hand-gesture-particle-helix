// MediaPipe Tasks Vision wrapper. Loads the model from public/mediapipe/
// (self-hosted) and the WASM runtime from jsdelivr (small, stable).

import {
  FilesetResolver,
  HandLandmarker,
} from '@mediapipe/tasks-vision';
import type { Handedness, Landmark } from '../types.ts';

const MODEL_URL = '/mediapipe/hand_landmarker.task';
const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

export interface Detection {
  handedness: Handedness;
  landmarks: ReadonlyArray<Landmark>;
}

export async function createHandLandmarker(): Promise<HandLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

/**
 * Run one detection on a video frame. Returns at most 2 detections.
 *
 * NOTE on handedness: we pass MediaPipe's label through unchanged. A previous
 * unconditional selfie flip behaved inconsistently across platforms (it
 * reversed Left/Right on iOS WebKit), so it was removed.
 */
export function detectFrame(
  lm: HandLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): Detection[] {
  const result = lm.detectForVideo(video, timestampMs);
  const out: Detection[] = [];
  const n = Math.min(result.landmarks.length, result.handedness.length);
  for (let i = 0; i < n; i++) {
    const lms = result.landmarks[i];
    const handCategories = result.handedness[i];
    if (!lms || !handCategories || handCategories.length === 0) continue;
    const rawLabel = handCategories[0]?.categoryName;
    if (rawLabel !== 'Left' && rawLabel !== 'Right') continue;
    // Use MediaPipe's label directly. An earlier "selfie flip" assumed the
    // raw front-camera frame was un-mirrored, but that varies by platform
    // (desktop vs iOS WebKit), which flipped Left/Right on mobile.
    const handedness: Handedness = rawLabel;
    const landmarks: Landmark[] = lms.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    out.push({ handedness, landmarks });
  }
  return out;
}
