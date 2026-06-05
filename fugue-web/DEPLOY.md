# Deployment

Two paths. Vercel is the recommended one — zero config, HTTPS, automatic
preview URLs per branch. GitHub Pages is the backup if you'd rather keep
everything on GitHub.

## Vercel (recommended)

1. Push `fugue-web/` to GitHub (any repo).
2. https://vercel.com/new → import the repo.
3. In project settings, set **Root Directory** to `fugue-web`.
4. Framework preset should auto-detect Vite. If not, choose "Vite".
5. Click Deploy.

That's it. `vercel.json` in this folder pins the install/build/output
settings explicitly so Vercel doesn't second-guess `pnpm`.

### Notes

- HTTPS is automatic, which the camera + MediaPipe both require.
- Custom domain: project settings → Domains.
- Build will fail if `../data/audio/*.mp3` is missing from the repo, since
  `scripts/sync-audio.mjs` runs as part of `pnpm prepare-assets`. Make sure
  the three stems are committed (or relax the `public/audio/*.mp3` rule in
  `.gitignore` and commit them directly into `fugue-web/public/audio/`).
- MediaPipe model (`hand_landmarker.task`, ~7MB) is fetched at build time
  by `scripts/download-model.mjs`. Vercel will pull it on each build.

## GitHub Pages (backup)

1. Repo settings → Pages → Source = GitHub Actions.
2. Drop the workflow below at `.github/workflows/fugue-web-pages.yml` in
   the repo root (not inside `fugue-web/`). Adjust `VITE_BASE_PATH` if your
   repo name differs.
3. Push to `main`. The action builds `fugue-web/dist/` and publishes.

```yaml
name: Deploy fugue-web to Pages

on:
  push:
    branches: [main]
    paths:
      - 'fugue-web/**'
      - 'data/audio/**'
      - '.github/workflows/fugue-web-pages.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: fugue-web
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
          cache-dependency-path: fugue-web/pnpm-lock.yaml
      - run: pnpm install --no-frozen-lockfile
      - run: pnpm build
        env:
          # Adjust to match your repo name.
          VITE_BASE_PATH: /hand-gesture-particle-helix/
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: fugue-web/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

### Notes

- `VITE_BASE_PATH` must end with `/`. Vite uses it as the public URL prefix
  so `/audio/violin.mp3` becomes `/hand-gesture-particle-helix/audio/violin.mp3`.
- Pages serves over HTTPS. Camera + MediaPipe work fine.
- First deploy after enabling Pages may take 1-2 minutes to propagate.

## Asset budget

| File | Size | Path |
|---|---|---|
| `violin.mp3` | 4.8 MB | `public/audio/` |
| `lute.mp3` | 4.8 MB | `public/audio/` |
| `organ.mp3` | 4.8 MB | `public/audio/` |
| `hand_landmarker.task` | 7.5 MB | `public/mediapipe/` |
| App JS + CSS | ~600 KB gzipped | `dist/assets/` |
| **Total** | **~22 MB** | |

Static-host friendly. No server-side anything.

## Verifying a deployed build

After deploy:

1. Open the URL in Chrome / Safari.
2. Confirm HTTPS (camera + MediaPipe both refuse mixed/HTTP).
3. Click Start → check the HUD shows `Audio: running`.
4. Allow camera → check the right-bottom PIP shows your mirrored selfie.
5. Try `?tier=low` in the URL to verify the tier override works.
6. Lighthouse Performance score should be > 80.
