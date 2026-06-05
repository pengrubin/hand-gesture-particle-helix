// Download the MediaPipe hand_landmarker.task model if missing.
// Self-hosted (rather than CDN) so first-paint is predictable and offline
// dev still works after the first run.

import { existsSync, mkdirSync, createWriteStream, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { get } from 'node:https';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DST = resolve(
  __dirname,
  '..',
  'public',
  'mediapipe',
  'hand_landmarker.task',
);
const URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

if (existsSync(DST) && statSync(DST).size > 1_000_000) {
  console.log('hand_landmarker.task already present.');
  process.exit(0);
}

mkdirSync(dirname(DST), { recursive: true });

console.log('downloading hand_landmarker.task ...');

const file = createWriteStream(DST);
get(URL, (res) => {
  if (res.statusCode !== 200) {
    console.error(`download failed: HTTP ${res.statusCode}`);
    file.close();
    process.exit(1);
  }
  res.pipe(file);
  file.on('finish', () => {
    file.close(() => {
      const size = statSync(DST).size;
      console.log(`  ✓ hand_landmarker.task  (${(size / 1024 / 1024).toFixed(1)} MB)`);
    });
  });
}).on('error', (err) => {
  console.error('download error:', err.message);
  process.exit(1);
});
