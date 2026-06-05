/// <reference types="vite/client" />

// Allow `import shader from './foo.glsl?raw'` as a string.
declare module '*.glsl?raw' {
  const content: string;
  export default content;
}
