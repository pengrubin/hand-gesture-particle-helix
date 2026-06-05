// High-res sphere with a vertex shader that displaces along the normal
// using 3D simplex noise. Uniforms are driven each frame from the audio
// analyser bands.

import { Mesh, ShaderMaterial, SphereGeometry } from 'three';
import sphereVert from './shaders/sphere.vert.glsl?raw';
import sphereFrag from './shaders/sphere.frag.glsl?raw';

export interface SphereUniforms {
  uTime: { value: number };
  uBass: { value: number };
  uMid: { value: number };
  uHigh: { value: number };
}

export interface SphereBundle {
  mesh: Mesh;
  uniforms: SphereUniforms;
}

export function createSphere(segments = 256): SphereBundle {
  const geometry = new SphereGeometry(1, segments, segments);

  const uniforms: SphereUniforms = {
    uTime: { value: 0 },
    uBass: { value: 0 },
    uMid: { value: 0 },
    uHigh: { value: 0 },
  };

  const material = new ShaderMaterial({
    uniforms,
    vertexShader: sphereVert,
    fragmentShader: sphereFrag,
  });

  const mesh = new Mesh(geometry, material);
  return { mesh, uniforms };
}
