import { defineConfig } from 'vite';

// Base path override for GitHub Pages deploys. Set VITE_BASE_PATH in the
// CI workflow, e.g. VITE_BASE_PATH=/hand-gesture-particle-helix/. Vercel
// and `pnpm dev` leave it unset and serve from `/`.
const basePath = process.env.VITE_BASE_PATH ?? '/';

export default defineConfig({
  base: basePath,
  server: {
    host: true,
    port: 5173,
    open: false,
  },
  build: {
    target: 'es2022',
    sourcemap: false,
  },
});
