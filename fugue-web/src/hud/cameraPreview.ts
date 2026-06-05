// Mirrored camera preview inside the right-bottom PIP, with a transparent
// canvas overlaying landmark dots + connections. Subscribes to AppState so
// the overlay redraws whenever hands change.

import { subscribe } from '../state.ts';
import type { HandState } from '../types.ts';

// MediaPipe Hands connection table (21 landmarks).
const CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  // thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  // palm closure
  [0, 17],
];

const LEFT_COLOR = '#5fff8c'; // mint green = user's left hand
const RIGHT_COLOR = '#5fb8ff'; // sky blue   = user's right hand

export function mountCameraPreview(video: HTMLVideoElement): void {
  const container = document.getElementById('camera-pip');
  if (!container) throw new Error('No #camera-pip element in DOM');

  // Mount the video, mirrored.
  video.style.width = '100%';
  video.style.height = '100%';
  video.style.objectFit = 'cover';
  video.style.transform = 'scaleX(-1)';
  video.style.display = 'block';
  container.appendChild(video);

  // Landmark overlay canvas, also mirrored.
  const canvas = document.createElement('canvas');
  canvas.style.position = 'absolute';
  canvas.style.inset = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.transform = 'scaleX(-1)';
  canvas.style.pointerEvents = 'none';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2d context unavailable');

  const resize = (): void => {
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
  };
  video.addEventListener('loadedmetadata', resize);
  resize();

  const drawHand = (hand: HandState, color: string): void => {
    const w = canvas.width;
    const h = canvas.height;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';

    for (const [a, b] of CONNECTIONS) {
      const pa = hand.landmarks[a];
      const pb = hand.landmarks[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x * w, pa.y * h);
      ctx.lineTo(pb.x * w, pb.y * h);
      ctx.stroke();
    }
    for (const p of hand.landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  subscribe((state) => {
    if (canvas.width === 0 || canvas.height === 0) resize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.hands.left) drawHand(state.hands.left, LEFT_COLOR);
    if (state.hands.right) drawHand(state.hands.right, RIGHT_COLOR);
  });
}
