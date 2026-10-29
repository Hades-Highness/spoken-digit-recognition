import { SAMPLE_RATE, WINDOW_SAMPLES } from "./onnxInference";

/** Decodes a browser recording and resamples it to the model's mono rate. */
export async function decodeRecording(blob: Blob): Promise<Float32Array> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const outputLength = Math.ceil(decoded.duration * SAMPLE_RATE);
    const offlineContext = new OfflineAudioContext(
      1,
      outputLength,
      SAMPLE_RATE,
    );
    const source = offlineContext.createBufferSource();
    source.buffer = decoded;
    source.connect(offlineContext.destination);
    source.start();

    return (await offlineContext.startRendering()).getChannelData(0).slice();
  } finally {
    await context.close();
  }
}

/**
 * Frames raw audio into the fixed one-second model input without speech
 * detection, trimming, or feature extraction.
 */
export function prepareInferenceWindow(audio: Float32Array): Float32Array {
  if (audio.length === 0) {
    throw new Error("The audio recording is empty.");
  }

  const window = new Float32Array(WINDOW_SAMPLES);
  if (audio.length >= WINDOW_SAMPLES) {
    const start = Math.floor((audio.length - WINDOW_SAMPLES) / 2);
    window.set(audio.subarray(start, start + WINDOW_SAMPLES));
  } else {
    const destinationStart = Math.floor((WINDOW_SAMPLES - audio.length) / 2);
    window.set(audio, destinationStart);
  }

  return window;
}

/** Downsamples audio into min/max sample pairs for an SVG waveform trace. */
export function getWaveformEnvelope(
  audio: Float32Array,
  pointCount = 800,
): Array<{ min: number; max: number }> {
  const envelope: Array<{ min: number; max: number }> = [];
  for (let point = 0; point < pointCount; point += 1) {
    const start = Math.floor((point * audio.length) / pointCount);
    const end = Math.max(
      start + 1,
      Math.floor(((point + 1) * audio.length) / pointCount),
    );
    let min = 0;
    let max = 0;
    for (let index = start; index < Math.min(end, audio.length); index += 1) {
      min = Math.min(min, audio[index]);
      max = Math.max(max, audio[index]);
    }
    envelope.push({ min, max });
  }
  return envelope;
}
