import {
  HOP_SAMPLES,
  SAMPLE_RATE,
  WINDOW_SAMPLES,
  runInference,
  type CalibrationSettings,
  type InferenceResult,
} from "./onnxInference";

/**
 * Microphone capture and 16 kHz resampling for the listen mode.
 *
 * Browsers hand us audio at the hardware rate (44.1 or 48 kHz) and will not
 * decimate by an integer factor, so the graph is:
 *
 *   MediaStreamSource -> AudioWorkletNode (FIR low-pass + linear interpolation
 *   decimation) -> zero-gain sink
 *
 * The worklet keeps the hot loop off the main thread; the main thread only
 * receives 100 ms raw mono chunks, buffers them into 16000-sample windows and
 * runs an inference every 8000 samples (0.5 s hop, so 50 percent overlap).
 */

/** Output rate expected by the ONNX graph. */
const TARGET_SAMPLE_RATE = SAMPLE_RATE;

/** Samples handed to the main thread per message: 100 ms at 16 kHz. */
const OUTPUT_CHUNK_SAMPLES = 1600;

/** Tap count of the anti-aliasing FIR. Odd so it has an integer group delay. */
const FIR_TAPS = 129;

/** Cutoff used for the anti-aliasing filter, just under the 8 kHz Nyquist. */
const FIR_CUTOFF_HZ = 7200;

const PROCESSOR_NAME = "digitsense-capture";

/**
 * Runs on the audio thread. Loaded from a Blob so the app needs no extra
 * bundler configuration for AudioWorklet modules.
 *
 * For every render quantum it low-passes the input with a windowed-sinc FIR,
 * then resamples the filtered signal by evaluating it at fractional positions
 * spaced by inputRate / 16000. Linear interpolation is accurate enough once
 * the signal is band-limited, and it costs one multiply-add per output sample.
 */
const CAPTURE_PROCESSOR_SOURCE = `
class DigitSenseCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const settings = options.processorOptions;
    this.ratio = settings.inputSampleRate / settings.targetSampleRate;
    this.fir = settings.firCoefficients;
    this.history = new Float32Array(this.fir.length);
    this.writeIndex = 0;
    this.phase = 1;
    this.previousFiltered = 0;
    this.chunk = new Float32Array(settings.chunkSamples);
    this.chunkLength = 0;
    this.filtered = new Float32Array(2048);
    this.resampled = new Float32Array(Math.ceil(4096 / this.ratio) + 8);
  }

  process(inputs) {
    const channels = inputs[0];
    const input = channels && channels[0];
    if (!input || input.length === 0) {
      return true;
    }

    const length = input.length;
    const fir = this.fir;
    const taps = fir.length;
    const history = this.history;
    const filtered = this.filtered;

    let writeIndex = this.writeIndex;
    for (let index = 0; index < length; index += 1) {
      history[writeIndex] = input[index];

      let accumulator = 0;
      let cursor = writeIndex;
      for (let tap = 0; tap < taps; tap += 1) {
        accumulator += fir[tap] * history[cursor];
        cursor = cursor === 0 ? taps - 1 : cursor - 1;
      }

      filtered[index] = accumulator;
      writeIndex = writeIndex + 1 === taps ? 0 : writeIndex + 1;
    }
    this.writeIndex = writeIndex;

    const resampled = this.resampled;
    let produced = 0;
    while (this.phase < length) {
      const position = Math.floor(this.phase);
      const fraction = this.phase - position;
      const start = position === 0 ? this.previousFiltered : filtered[position - 1];
      const end = filtered[position];
      resampled[produced] = start + (end - start) * fraction;
      produced += 1;
      this.phase += this.ratio;
    }
    this.phase -= length;
    this.previousFiltered = filtered[length - 1];

    let offset = 0;
    while (offset < produced) {
      const room = this.chunk.length - this.chunkLength;
      const take = Math.min(room, produced - offset);
      this.chunk.set(resampled.subarray(offset, offset + take), this.chunkLength);
      this.chunkLength += take;
      offset += take;

      if (this.chunkLength === this.chunk.length) {
        const message = this.chunk.slice();
        this.port.postMessage(message.buffer, [message.buffer]);
        this.chunkLength = 0;
      }
    }

    return true;
  }
}

registerProcessor("${PROCESSOR_NAME}", DigitSenseCaptureProcessor);
`;

export type AudioProcessorOptions = {
  /** Selected input device, or "default"/undefined for the system default. */
  deviceId?: string;
  /** Called once per inference window while the session is running. */
  onResult: (result: InferenceResult) => void;
  /** Called for each captured audio chunk with its normalized signal level. */
  onAudioLevel?: (level: number) => void;
  calibration?: CalibrationSettings;
  /** Called after the session stops because of an unrecoverable error. */
  onError?: (error: Error) => void;
};

/** Hamming-windowed sinc low-pass with unity DC gain. */
function designLowPass(
  cutoffHz: number,
  sampleRate: number,
  taps: number,
): Float32Array {
  const coefficients = new Float32Array(taps);
  const normalized = cutoffHz / sampleRate;
  const center = (taps - 1) / 2;
  let sum = 0;

  for (let index = 0; index < taps; index += 1) {
    const offset = index - center;
    const sinc =
      offset === 0
        ? 2 * normalized
        : Math.sin(2 * Math.PI * normalized * offset) /
          (Math.PI * offset);
    const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * index) / (taps - 1));
    const value = sinc * window;
    coefficients[index] = value;
    sum += value;
  }

  for (let index = 0; index < taps; index += 1) {
    coefficients[index] /= sum;
  }

  return coefficients;
}

/** Turns the browser's DOMException names into actionable UI copy. */
function describeCaptureError(error: unknown): string {
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    switch (error.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Microphone access was blocked. Allow microphone permission for this site and try again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No usable microphone was found. Choose another input in Settings.";
      case "NotReadableError":
        return "The microphone is already in use by another application.";
      default:
        return `Audio capture failed: ${error.message}`;
    }
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Unknown audio capture error.";
}

/**
 * Streams raw microphone audio to the DigitSense model after sample-rate
 * conversion.
 *
 * Lifecycle: start() opens the device, builds the worklet graph and begins
 * emitting one InferenceResult per 0.5 s; stop() tears everything down. Any
 * unrecoverable failure tears the graph down first and then reports via
 * onError, so the UI never shows "listening" with a dead pipeline.
 */
export class AudioProcessor {
  private readonly deviceId?: string;

  private readonly onResult: (result: InferenceResult) => void;

  private readonly onAudioLevel?: (level: number) => void;

  private calibration?: CalibrationSettings;

  private readonly onError?: (error: Error) => void;

  private context: AudioContext | null = null;

  private stream: MediaStream | null = null;

  private source: MediaStreamAudioSourceNode | null = null;

  private worklet: AudioWorkletNode | null = null;

  private sink: GainNode | null = null;

  /** Holds up to two windows so a 100 ms chunk can complete several hops. */
  private readonly buffer = new Float32Array(WINDOW_SAMPLES * 2);

  private buffered = 0;

  private running = false;

  private inFlight = false;

  private failure: Error | null = null;

  private generation = 0;

  constructor(options: AudioProcessorOptions) {
    this.deviceId = options.deviceId;
    this.onResult = options.onResult;
    this.onAudioLevel = options.onAudioLevel;
    this.calibration = options.calibration;
    this.onError = options.onError;
  }

  get isRunning(): boolean {
    return this.running;
  }

  setCalibration(calibration: CalibrationSettings): void {
    this.calibration = calibration;
  }

  /** Requests the microphone and starts the 0.5 s classification loop. */
  async start(): Promise<void> {
    if (this.running) {
      return;
    }

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      throw new Error(
        "This browser does not expose microphone capture (navigator.mediaDevices.getUserMedia).",
      );
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(
        this.buildConstraints(),
      );

      // Keep the hardware rate: forcing 16000 Hz makes some drivers resample
      // poorly (or reject the constraint), and the worklet needs the native
      // rate to plan its decimation phase anyway.
      const context = new AudioContext();
      this.context = context;
      await context.resume();

      const moduleUrl = URL.createObjectURL(
        new Blob([CAPTURE_PROCESSOR_SOURCE], { type: "text/javascript" }),
      );
      try {
        await context.audioWorklet.addModule(moduleUrl);
      } finally {
        URL.revokeObjectURL(moduleUrl);
      }

      this.source = context.createMediaStreamSource(this.stream);

      const cutoff = Math.min(
        FIR_CUTOFF_HZ,
        context.sampleRate / 2 - 1,
      );

      this.worklet = new AudioWorkletNode(context, PROCESSOR_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit",
        processorOptions: {
          inputSampleRate: context.sampleRate,
          targetSampleRate: TARGET_SAMPLE_RATE,
          chunkSamples: OUTPUT_CHUNK_SAMPLES,
          firCoefficients: designLowPass(cutoff, context.sampleRate, FIR_TAPS),
        },
      });

      this.worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        this.append(new Float32Array(event.data));
      };
      this.worklet.onprocessorerror = () => {
        this.fail(new Error("The audio worklet stopped unexpectedly."));
      };

      this.source.connect(this.worklet);

      // The worklet has to be pulled by the destination; a muted gain node
      // keeps the graph alive without echoing the microphone.
      const sink = context.createGain();
      sink.gain.value = 0;
      this.worklet.connect(sink);
      sink.connect(context.destination);
      this.sink = sink;

      this.buffered = 0;
      this.failure = null;
      this.running = true;
    } catch (error) {
      await this.teardown();
      throw new Error(describeCaptureError(error));
    }
  }

  /** Stops capture and releases the microphone. Safe to call twice. */
  async stop(): Promise<void> {
    await this.teardown();
  }

  private async teardown(): Promise<void> {
    this.generation += 1;
    this.running = false;
    this.buffered = 0;
    this.inFlight = false;

    if (this.worklet) {
      this.worklet.port.onmessage = null;
      this.worklet.onprocessorerror = null;
      this.worklet.port.close();
      this.worklet.disconnect();
      this.worklet = null;
    }

    this.source?.disconnect();
    this.source = null;

    this.sink?.disconnect();
    this.sink = null;

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.context) {
      const context = this.context;
      this.context = null;
      if (context.state !== "closed") {
        try {
          await context.close();
        } catch {
          // Already closed by the browser; nothing left to release.
        }
      }
    }

    this.onAudioLevel?.(0);
  }

  /** Buffers a downsampled chunk and classifies every completed window. */
  private append(samples: Float32Array): void {
    if (!this.running) {
      return;
    }

    let squaredTotal = 0;
    for (const sample of samples) {
      squaredTotal += sample * sample;
    }
    const rms = Math.sqrt(squaredTotal / samples.length);
    this.onAudioLevel?.(Math.min(1, Math.max(0, (rms - 0.004) * 12)));

    let offset = 0;
    while (offset < samples.length) {
      const room = this.buffer.length - this.buffered;
      const take = Math.min(room, samples.length - offset);
      this.buffer.set(samples.subarray(offset, offset + take), this.buffered);
      this.buffered += take;
      offset += take;

      // 100 ms chunks usually complete zero or one hop, but the loop keeps the
      // invariant even if a worklet message is delayed.
      while (this.buffered >= WINDOW_SAMPLES) {
        const window = this.buffer.slice(0, WINDOW_SAMPLES);
        this.buffer.copyWithin(0, HOP_SAMPLES, this.buffered);
        this.buffered -= HOP_SAMPLES;
        void this.classify(window);
      }
    }
  }

  private async classify(window: Float32Array): Promise<void> {
    if (this.inFlight || !this.running || this.failure) {
      return;
    }

    const generation = this.generation;
    this.inFlight = true;
    try {
      const result = await runInference(window, this.calibration);
      // Ignore results that finish after stop()/start() so a stale window can
      // never land in a new session's history.
      if (this.running && this.generation === generation) {
        this.onResult(result);
      }
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
    } finally {
      this.inFlight = false;
    }
  }

  /** Tears the graph down, then reports the failure exactly once. */
  private fail(error: Error): void {
    if (this.failure) {
      return;
    }

    this.failure = error;
    void this.teardown().then(() => this.onError?.(error));
  }

  private buildConstraints(): MediaStreamConstraints {
    const audio: MediaTrackConstraints = {
      channelCount: 1,
      // The model was trained on raw speech; browser DSP changes the spectrum
      // and the calibration was fitted without it.
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    };

    if (this.deviceId && this.deviceId !== "default") {
      audio.deviceId = { exact: this.deviceId };
    }

    return { audio, video: false };
  }
}
