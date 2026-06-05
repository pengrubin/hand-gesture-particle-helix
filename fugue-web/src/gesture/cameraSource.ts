// getUserMedia wrapper. Always front-facing (selfie) at the requested
// resolution. Returns a ready-to-render <video> element + actual dimensions
// (the device may not honour the request exactly).

export interface CameraResult {
  video: HTMLVideoElement;
  width: number;
  height: number;
  stream: MediaStream;
}

export async function startCamera(
  requestedWidth = 640,
  requestedHeight = 480,
): Promise<CameraResult> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia not available in this browser');
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: requestedWidth },
      height: { ideal: requestedHeight },
      frameRate: { ideal: 30 },
      facingMode: 'user',
    },
  });

  const video = document.createElement('video');
  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;

  await new Promise<void>((resolve, reject) => {
    const onReady = (): void => {
      video.removeEventListener('loadedmetadata', onReady);
      resolve();
    };
    video.addEventListener('loadedmetadata', onReady);
    video.addEventListener('error', () => reject(new Error('video error')));
  });
  await video.play();

  return {
    video,
    width: video.videoWidth || requestedWidth,
    height: video.videoHeight || requestedHeight,
    stream,
  };
}
