// Copy the three Bach stems from ../data/audio/ into public/audio/.
// Runs before `vite dev` and on demand. Idempotent.

import { mkdirSync, copyFileSync, existsSync, statSync } from 'node:fs';
import { dirname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC_DIR = resolve(__dirname, '..', '..', 'data', 'audio');
const DST_DIR = resolve(__dirname, '..', 'public', 'audio');

const FILES = {
  'Fugue in G Trio violin-Violin.mp3': 'violin.mp3',
  'Fugue in G Trio-Tenor_Lute.mp3': 'lute.mp3',
  'Fugue in G Trio Organ-Organ.mp3': 'organ.mp3',
};

mkdirSync(DST_DIR, { recursive: true });

let copied = 0;
let skipped = 0;
let missing = [];

for (const [srcName, dstName] of Object.entries(FILES)) {
  const srcPath = resolve(SRC_DIR, srcName);
  const dstPath = resolve(DST_DIR, dstName);

  if (!existsSync(srcPath)) {
    missing.push(srcName);
    continue;
  }

  if (existsSync(dstPath) && statSync(dstPath).size === statSync(srcPath).size) {
    skipped += 1;
    continue;
  }

  copyFileSync(srcPath, dstPath);
  copied += 1;
  console.log(`  ✓ ${dstName}  (${basename(srcName)})`);
}

if (missing.length > 0) {
  console.error(`\n✗ Missing source files in ${SRC_DIR}:`);
  for (const m of missing) console.error(`    - ${m}`);
  process.exit(1);
}

console.log(`audio sync: ${copied} copied, ${skipped} already up to date.`);
