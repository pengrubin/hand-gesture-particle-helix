// Hand-rolled post-process pipeline.
//
//   sphere scene
//        │ render
//        ▼
//   sceneRT (full res)
//        │ threshold extract (luma > uThreshold)
//        ▼
//   brightRT (half res)
//        │ kawase blur, 3 passes ping-pong  ┐
//        ▼                                   │ offsets 1.5, 3.0, 5.0
//   blur ping-pong (half res) ───────────────┘
//        │ final blur output → tBloom
//        │
//   sceneRT, blur, prevRT ─► composite ─► currRT
//                                   │
//                                   ├─► copy ─► prevRT (next frame's tail)
//                                   │
//                                   └─► copy ─► canvas
//
// Bloom + feedback fall straight out of the original TouchDesigner GLSL.
// Bloom kernel is half-res for perf; final image is full-res.

import {
  BufferAttribute,
  BufferGeometry,
  Color,
  LinearFilter,
  Mesh,
  OrthographicCamera,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Vector2,
  WebGLRenderTarget,
  type Camera,
  type WebGLRenderer,
} from 'three';

import fullscreenVert from './shaders/fullscreen.vert.glsl?raw';
import bloomBlurFrag from './shaders/bloom-blur.frag.glsl?raw';
import bloomThresholdFrag from './shaders/bloom-threshold.frag.glsl?raw';
import compositeFrag from './shaders/composite.frag.glsl?raw';
import copyFrag from './shaders/copy.frag.glsl?raw';

export interface PostUniforms {
  bloomIntensity: { value: number };
  bloomThreshold: { value: number };
  glowIntensity: { value: number };
  glowColor: { value: Color };
  feedbackDecay: { value: number };
}

export interface PostProcessorOptions {
  /** Kawase blur offsets, one per ping-pong pass. Default: [1.5, 3.0, 5.0]. */
  blurOffsets?: number[];
}

export class PostProcessor {
  readonly uniforms: PostUniforms;

  private readonly renderer: WebGLRenderer;
  private readonly scene: Scene;
  private readonly sceneCam: Camera;
  private readonly blurOffsets: number[];

  private readonly sceneRT: WebGLRenderTarget;
  private readonly brightRT: WebGLRenderTarget;
  private readonly blurPing: WebGLRenderTarget;
  private readonly blurPong: WebGLRenderTarget;
  private readonly prevRT: WebGLRenderTarget;
  private readonly currRT: WebGLRenderTarget;

  private readonly thresholdMat: ShaderMaterial;
  private readonly blurMat: ShaderMaterial;
  private readonly compositeMat: ShaderMaterial;
  private readonly copyMat: ShaderMaterial;

  private readonly fsScene: Scene;
  private readonly fsCam: OrthographicCamera;
  private readonly fsQuad: Mesh;

  constructor(
    renderer: WebGLRenderer,
    scene: Scene,
    sceneCam: Camera,
    opts: PostProcessorOptions = {},
  ) {
    this.renderer = renderer;
    this.scene = scene;
    this.sceneCam = sceneCam;
    this.blurOffsets = opts.blurOffsets ?? [1.5, 3.0, 5.0];

    this.uniforms = {
      bloomIntensity: { value: 1.0 },
      bloomThreshold: { value: 0.30 },
      glowIntensity: { value: 0.35 },
      glowColor: { value: new Color(0x6a9aff) },
      feedbackDecay: { value: 0.88 },
    };

    const size = new Vector2();
    renderer.getDrawingBufferSize(size);

    const rtOpts = {
      format: RGBAFormat,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    };

    this.sceneRT = makeRT(size.x, size.y, { ...rtOpts, depthBuffer: true });
    this.brightRT = makeRT(size.x / 2, size.y / 2, rtOpts);
    this.blurPing = makeRT(size.x / 2, size.y / 2, rtOpts);
    this.blurPong = makeRT(size.x / 2, size.y / 2, rtOpts);
    this.prevRT = makeRT(size.x, size.y, rtOpts);
    this.currRT = makeRT(size.x, size.y, rtOpts);

    this.thresholdMat = new ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: bloomThresholdFrag,
      uniforms: {
        tInput: { value: null },
        uThreshold: this.uniforms.bloomThreshold,
      },
    });

    this.blurMat = new ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: bloomBlurFrag,
      uniforms: {
        tInput: { value: null },
        uResolution: { value: new Vector2(size.x / 2, size.y / 2) },
        uOffset: { value: 1.5 },
      },
    });

    this.compositeMat = new ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: compositeFrag,
      uniforms: {
        tScene: { value: null },
        tBloom: { value: null },
        tPrev: { value: null },
        uBloomIntensity: this.uniforms.bloomIntensity,
        uGlowIntensity: this.uniforms.glowIntensity,
        uGlowColor: this.uniforms.glowColor,
        uFeedbackDecay: this.uniforms.feedbackDecay,
      },
    });

    this.copyMat = new ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: copyFrag,
      uniforms: { tInput: { value: null } },
    });

    // Full-screen clip-space quad (two triangles).
    const geom = new BufferGeometry();
    geom.setAttribute(
      'position',
      new BufferAttribute(
        new Float32Array([
          -1, -1, 0,
           1, -1, 0,
          -1,  1, 0,
           1,  1, 0,
        ]),
        3,
      ),
    );
    geom.setAttribute(
      'uv',
      new BufferAttribute(
        new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
        2,
      ),
    );
    geom.setIndex([0, 1, 2, 1, 3, 2]);

    this.fsScene = new Scene();
    this.fsQuad = new Mesh(geom);
    this.fsScene.add(this.fsQuad);
    this.fsCam = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

    window.addEventListener('resize', () => this.handleResize());
  }

  private handleResize(): void {
    const size = new Vector2();
    this.renderer.getDrawingBufferSize(size);
    this.sceneRT.setSize(size.x, size.y);
    this.brightRT.setSize(size.x / 2, size.y / 2);
    this.blurPing.setSize(size.x / 2, size.y / 2);
    this.blurPong.setSize(size.x / 2, size.y / 2);
    this.prevRT.setSize(size.x, size.y);
    this.currRT.setSize(size.x, size.y);
    (this.blurMat.uniforms.uResolution!.value as Vector2).set(size.x / 2, size.y / 2);
  }

  private blit(target: WebGLRenderTarget | null, material: ShaderMaterial): void {
    this.fsQuad.material = material;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.fsScene, this.fsCam);
  }

  render(): void {
    // 1. Scene to its own render target.
    this.renderer.setRenderTarget(this.sceneRT);
    this.renderer.clear();
    this.renderer.render(this.scene, this.sceneCam);

    // 2. Threshold extract bright pixels.
    this.thresholdMat.uniforms.tInput!.value = this.sceneRT.texture;
    this.blit(this.brightRT, this.thresholdMat);

    // 3. Kawase blur ping-pong with growing offsets.
    let src = this.brightRT;
    let dst = this.blurPing;
    for (const off of this.blurOffsets) {
      this.blurMat.uniforms.tInput!.value = src.texture;
      this.blurMat.uniforms.uOffset!.value = off;
      this.blit(dst, this.blurMat);
      const swap = src === this.brightRT ? this.blurPong : src;
      src = dst;
      dst = swap === src ? this.blurPing : swap;
    }
    // After the loop, `src` is the final blurred bloom.

    // 4. Composite scene + bloom + feedback into currRT.
    this.compositeMat.uniforms.tScene!.value = this.sceneRT.texture;
    this.compositeMat.uniforms.tBloom!.value = src.texture;
    this.compositeMat.uniforms.tPrev!.value = this.prevRT.texture;
    this.blit(this.currRT, this.compositeMat);

    // 5. Snapshot currRT into prevRT for next frame's feedback.
    this.copyMat.uniforms.tInput!.value = this.currRT.texture;
    this.blit(this.prevRT, this.copyMat);

    // 6. Blit currRT to the canvas.
    this.copyMat.uniforms.tInput!.value = this.currRT.texture;
    this.blit(null, this.copyMat);
  }
}

function makeRT(
  w: number,
  h: number,
  opts: ConstructorParameters<typeof WebGLRenderTarget>[2],
): WebGLRenderTarget {
  return new WebGLRenderTarget(Math.max(1, Math.floor(w)), Math.max(1, Math.floor(h)), opts);
}
