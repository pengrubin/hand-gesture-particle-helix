// Three.js WebGL renderer + scene + perspective camera. Single full-screen
// canvas, inserted behind everything else in the DOM (the HUD overlays on
// top).

import {
  Color,
  PerspectiveCamera,
  Scene,
  WebGLRenderer,
} from 'three';

export interface RendererBundle {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
}

export function createRenderer(pixelRatioCap = 2): RendererBundle {
  const canvas = document.createElement('canvas');
  canvas.id = 'scene-canvas';
  canvas.style.position = 'fixed';
  canvas.style.inset = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  canvas.style.zIndex = '0';
  document.body.insertBefore(canvas, document.body.firstChild);

  const renderer = new WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, pixelRatioCap));
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  const scene = new Scene();
  scene.background = new Color(0x050507);

  const camera = new PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    100,
  );
  camera.position.set(0, 0, 3.2);
  camera.lookAt(0, 0, 0);

  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });

  return { renderer, scene, camera };
}
