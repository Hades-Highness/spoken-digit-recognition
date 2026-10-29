import type * as OrtTypes from "onnxruntime-web";

/** Native sample rate expected by the v4.1 ONNX graph. */
export const SAMPLE_RATE = 16000;

/** The current web pipeline sends one second of raw mono audio per inference. */
export const WINDOW_SAMPLES = SAMPLE_RATE;

/** Consecutive listen-mode windows overlap by 50 percent. */
export const HOP_SAMPLES = WINDOW_SAMPLES / 2;

/** Release artifact served from the public folder. */
export const MODEL_URL = "/models/DigitSense_v4.1.onnx";

/** Default minimum softmax confidence required to accept a digit. */
export const CONFIDENCE_THRESHOLD = 0.6;

export const DIGIT_LABELS: readonly number[] = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

export type CalibrationSettings = {
  confidenceThreshold: number;
};

export type DigitPrediction = number | "Unknown";

export type InferenceResult = {
  digit: DigitPrediction;
  confidence: number;
};

type OrtModule = typeof import("onnxruntime-web");

declare global {
  interface Window {
    __digitSenseOrt?: OrtModule;
  }
}

let ortModulePromise: Promise<OrtModule> | null = null;

async function loadOrtRuntime(): Promise<OrtModule> {
  if (typeof window === "undefined" || typeof document === "undefined") {
    throw new Error("ONNX Runtime Web can only be loaded in a browser.");
  }

  if (!ortModulePromise) {
    ortModulePromise = new Promise<OrtModule>((resolve, reject) => {
      const script = document.createElement("script");
      script.type = "module";
      script.src = "/ort/ort-runtime-loader.mjs";
      script.onload = () => {
        try {
          const ort = window.__digitSenseOrt;
          if (!ort) {
            throw new Error("The ONNX Runtime module did not initialize.");
          }

          const isMobile =
            /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
              navigator.userAgent,
            ) || window.innerWidth < 768;

          ort.env.wasm.wasmPaths = "/ort/";
          ort.env.logLevel = "error";
          if (isMobile) {
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.proxy = false;
          }
          resolve(ort);
        } catch (error) {
          reject(error);
        }
      };
      script.onerror = () => {
        script.remove();
        reject(
          new Error(
            "Unable to load /ort/ort-runtime-loader.mjs. Run the project predev or prebuild script to copy the ONNX Runtime assets.",
          ),
        );
      };
      document.head.append(script);
    }).catch((error: unknown) => {
      ortModulePromise = null;
      throw error;
    });
  }

  return ortModulePromise;
}

/** Owns the WASM session and serializes overlapping inference requests. */
export class DigitSenseInference {
  private session: OrtTypes.InferenceSession | null = null;

  private sessionPromise: Promise<OrtTypes.InferenceSession> | null = null;

  private inputName = "raw_audio";

  private outputName = "logits";

  private queue: Promise<unknown> = Promise.resolve();

  get isReady(): boolean {
    return this.session !== null;
  }

  async load(): Promise<void> {
    await this.ensureSession();
  }

  /** Classifies exactly 16000 raw mono float samples at 16 kHz. */
  runInference(
    audioData: Float32Array,
    calibration: CalibrationSettings = {
      confidenceThreshold: CONFIDENCE_THRESHOLD,
    },
  ): Promise<InferenceResult> {
    if (audioData.length !== WINDOW_SAMPLES) {
      return Promise.reject(
        new Error(
          `runInference expects exactly ${WINDOW_SAMPLES} samples, received ${audioData.length}.`,
        ),
      );
    }

    const task = this.queue.then(() => this.execute(audioData, calibration));
    this.queue = task.then(
      () => undefined,
      () => undefined,
    );
    return task;
  }

  async dispose(): Promise<void> {
    const session = this.session;
    this.session = null;
    this.sessionPromise = null;
    if (session) {
      await session.release();
    }
  }

  private async execute(
    audioData: Float32Array,
    calibration: CalibrationSettings,
  ): Promise<InferenceResult> {
    const ort = await loadOrtRuntime();
    const session = await this.ensureSession();
    const input = new ort.Tensor("float32", audioData, [1, WINDOW_SAMPLES]);
    const outputs = await session.run({ [this.inputName]: input });
    const output = outputs[this.outputName] ?? Object.values(outputs)[0];

    if (!output) {
      throw new Error("The DigitSense model returned no output tensor.");
    }
    if (!(output.data instanceof Float32Array)) {
      throw new Error("The DigitSense model returned non-float32 logits.");
    }
    if (output.data.length !== DIGIT_LABELS.length) {
      throw new Error(
        `The model returned ${output.data.length} logits but ${DIGIT_LABELS.length} digit labels are mapped. The model and label order do not match.`,
      );
    }

    return classifyLogits(output.data, calibration.confidenceThreshold);
  }

  private ensureSession(): Promise<OrtTypes.InferenceSession> {
    if (this.session) {
      return Promise.resolve(this.session);
    }
    if (!this.sessionPromise) {
      this.sessionPromise = this.createSession().catch((error: unknown) => {
        this.sessionPromise = null;
        throw error;
      });
    }
    return this.sessionPromise;
  }

  private async createSession(): Promise<OrtTypes.InferenceSession> {
    const ort = await loadOrtRuntime();
    const response = await fetch(MODEL_URL);
    if (!response.ok) {
      throw new Error(
        `Unable to load ${MODEL_URL} (HTTP ${response.status}). Place the DigitSense v4.1 raw-audio ONNX artifact in web/public/models/.`,
      );
    }

    const model = new Uint8Array(await response.arrayBuffer());
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    this.inputName = session.inputNames[0] ?? "raw_audio";
    this.outputName = session.outputNames[0] ?? "logits";

    await session.run({
      [this.inputName]: new ort.Tensor(
        "float32",
        new Float32Array(WINDOW_SAMPLES),
        [1, WINDOW_SAMPLES],
      ),
    });
    this.session = session;
    return session;
  }
}

/** Applies softmax to the ten model logits and rejects low-confidence output. */
export function classifyLogits(
  logits: Float32Array,
  confidenceThreshold = CONFIDENCE_THRESHOLD,
): InferenceResult {
  if (
    !Number.isFinite(confidenceThreshold) ||
    confidenceThreshold < 0 ||
    confidenceThreshold > 1
  ) {
    throw new Error("Confidence threshold must be between 0 and 1.");
  }
  if (logits.length !== DIGIT_LABELS.length) {
    throw new Error(
      `Expected ${DIGIT_LABELS.length} digit logits, received ${logits.length}.`,
    );
  }

  let maxLogit = Number.NEGATIVE_INFINITY;
  for (const logit of logits) {
    if (!Number.isFinite(logit)) {
      throw new Error("The DigitSense model returned a non-finite logit.");
    }
    maxLogit = Math.max(maxLogit, logit);
  }

  const exponentials = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp(logits[index] - maxLogit);
    exponentials[index] = value;
    total += value;
  }

  let bestIndex = 0;
  let maxConfidence = 0;
  for (let index = 0; index < exponentials.length; index += 1) {
    const confidence = exponentials[index] / total;
    if (confidence > maxConfidence) {
      bestIndex = index;
      maxConfidence = confidence;
    }
  }

  return {
    digit:
      maxConfidence < confidenceThreshold ? "Unknown" : DIGIT_LABELS[bestIndex],
    confidence: maxConfidence,
  };
}

export const onnxInference = new DigitSenseInference();

export function loadModel(): Promise<void> {
  return onnxInference.load();
}

export function runInference(
  audioData: Float32Array,
  calibration?: CalibrationSettings,
): Promise<InferenceResult> {
  return onnxInference.runInference(audioData, calibration);
}
