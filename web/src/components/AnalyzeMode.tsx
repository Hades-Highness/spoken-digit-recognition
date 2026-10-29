"use client";

import { useEffect, useRef, useState } from "react";
import {
  loadModel,
  runInference,
  type CalibrationSettings,
  type InferenceResult,
} from "@/lib/onnxInference";
import {
  decodeRecording,
  getWaveformEnvelope,
  prepareInferenceWindow,
} from "@/lib/singleAudio";

const MAX_RECORDING_MS = 5000;
const GRID_COLUMNS = [100, 200, 300, 400, 500, 600, 700];
const GRID_ROWS = [60, 120, 180];

type AnalyzeModeProps = {
  microphoneId: string;
  calibration: CalibrationSettings;
};

type CaptureStatus = "empty" | "capturing" | "processing" | "ready" | "error";

export default function AnalyzeMode({
  microphoneId,
  calibration,
}: AnalyzeModeProps) {
  const [status, setStatus] = useState<CaptureStatus>("empty");
  const [audio, setAudio] = useState<Float32Array | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [liveEnvelope, setLiveEnvelope] = useState<
    Array<{ min: number; max: number }>
  >([]);
  const [recordingProgress, setRecordingProgress] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      const audioContext = audioContextRef.current;
      audioContextRef.current = null;
      if (audioContext && audioContext.state !== "closed") {
        void audioContext.close();
      }
    };
  }, []);

  useEffect(() => {
    if (!audioUrl) {
      return;
    }
    return () => URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  const startRecording = async () => {
    setErrorMessage(null);
    setResult(null);
    setAudio(null);
    setLiveEnvelope([]);
    setRecordingProgress(0);
    setStatus("processing");
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          ...(microphoneId && microphoneId !== "default"
            ? { deviceId: { exact: microphoneId } }
            : {}),
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      if (typeof MediaRecorder === "undefined") {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("Audio recording is not supported by this browser.");
      }

      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      const mutedOutput = audioContext.createGain();
      mutedOutput.gain.value = 0;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.connect(mutedOutput);
      mutedOutput.connect(audioContext.destination);
      await audioContext.resume();
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        await audioContext.close();
        return;
      }
      streamRef.current = stream;
      recorderRef.current = recorder;
      const timeDomain = new Float32Array(analyser.fftSize);
      const livePoints: Array<{ min: number; max: number }> = [];
      let lastSampleAt = 0;

      const updateLiveWaveform = (now: number) => {
        if (recorder.state !== "recording") {
          return;
        }
        const elapsed = now - recordingStartedAtRef.current;
        if (now - lastSampleAt >= 50) {
          lastSampleAt = now;
          setRecordingProgress(Math.min(1, elapsed / MAX_RECORDING_MS));
          analyser.getFloatTimeDomainData(timeDomain);
          let min = 0;
          let max = 0;
          for (const sample of timeDomain) {
            min = Math.min(min, sample);
            max = Math.max(max, sample);
          }
          livePoints.push({ min, max });
          setLiveEnvelope([...livePoints]);
        }
        animationFrameRef.current =
          requestAnimationFrame(updateLiveWaveform);
      };

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      recorder.onerror = () => {
        if (animationFrameRef.current !== null) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        recorderRef.current = null;
        audioContextRef.current = null;
        if (audioContext.state !== "closed") {
          void audioContext.close();
        }
        if (mountedRef.current) {
          setStatus("error");
          setErrorMessage("Audio recording failed. Please try again.");
        }
      };
      recorder.onstop = () => {
        if (recorderRef.current !== recorder) {
          return;
        }
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
        if (animationFrameRef.current !== null) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        recorderRef.current = null;
        audioContextRef.current = null;
        if (audioContext.state !== "closed") {
          void audioContext.close();
        }
        if (!mountedRef.current) {
          return;
        }

        setRecordingProgress(1);
        setStatus("processing");
        const recording = new Blob(chunks, { type: recorder.mimeType });
        void decodeRecording(recording)
          .then((recordedAudio) => {
            if (!mountedRef.current) {
              return;
            }
            setAudio(recordedAudio);
            setAudioUrl(URL.createObjectURL(recording));
            setAudioName("Microphone recording");
            setStatus("ready");
          })
          .catch((error: unknown) => {
            if (!mountedRef.current) {
              return;
            }
            setStatus("error");
            setErrorMessage(
              error instanceof Error
                ? `The recording could not be processed: ${error.message}`
                : "The recording could not be processed.",
            );
          });
      };

      recorder.start();
      recordingStartedAtRef.current = performance.now();
      setStatus("capturing");
      animationFrameRef.current =
        requestAnimationFrame(updateLiveWaveform);
      timeoutRef.current = setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
        }
      }, MAX_RECORDING_MS);
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      recorderRef.current = null;
      const audioContext = audioContextRef.current;
      audioContextRef.current = null;
      if (audioContext && audioContext.state !== "closed") {
        void audioContext.close();
      }
      setStatus("error");
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "The microphone could not be started.",
      );
    }
  };

  const stopRecording = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  };

  const handleAudioUpload = async (file: File | undefined) => {
    if (!file) {
      return;
    }

    setErrorMessage(null);
    setResult(null);
    setIsAnalyzing(false);
    setStatus("processing");
    try {
      const decodedAudio = await decodeRecording(file);
      if (!mountedRef.current) {
        return;
      }
      setAudio(decodedAudio);
      setAudioUrl(URL.createObjectURL(file));
      setAudioName(file.name);
      setStatus("ready");
    } catch (error) {
      if (!mountedRef.current) {
        return;
      }
      setStatus(audio ? "ready" : "error");
      setErrorMessage(
        error instanceof Error
          ? `The audio file could not be opened: ${error.message}`
          : "The audio file could not be opened.",
      );
    }
  };

  useEffect(() => {
    if (recorderRef.current?.state === "recording") {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      recorderRef.current.stop();
    }
  }, [microphoneId]);

  const analyzeRecording = async () => {
    if (!audio) {
      return;
    }
    setIsAnalyzing(true);
    setErrorMessage(null);
    setResult(null);
    try {
      const input = prepareInferenceWindow(audio);
      await loadModel();
      const inferenceResult = await runInference(input, calibration);
      if (mountedRef.current) {
        setResult(inferenceResult);
      }
    } catch (error) {
      if (mountedRef.current) {
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "The recording could not be analyzed.",
        );
      }
    } finally {
      if (mountedRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  const waveformPath = audio
    ? buildWaveformPath(audio)
    : status === "capturing"
      ? buildEnvelopePath(liveEnvelope)
      : "";
  const rejected = result
    ? result.digit === "Unknown"
    : false;

  return (
    <section
      aria-label="Single-shot waveform analysis"
      className="w-full rounded-3xl border border-white/5 bg-white/[0.02] p-5 sm:p-8"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[12px] font-medium uppercase tracking-[0.22em] text-text-muted">
            Analyze Mode
          </p>
        </div>

        <span
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
            status === "capturing"
              ? "border-primary-hover/60 text-primary-hover"
              : "border-white/10 text-text-muted"
          }`}
        >
          <span
            aria-hidden="true"
            className={`h-1.5 w-1.5 rounded-full ${
              status === "capturing"
                ? "animate-pulse-slow bg-primary-hover"
                : "bg-text-muted/60"
            }`}
          />
          {status === "capturing"
            ? "Recording"
            : status === "processing"
              ? "Processing"
              : status === "ready"
                ? "Recording ready"
                : status === "error"
                  ? "Error"
                  : "Ready to record"}
        </span>
      </div>

      <div className="relative mt-6 overflow-hidden rounded-2xl border border-white/5 bg-background/60">
        <svg
          viewBox="0 0 800 240"
          preserveAspectRatio="none"
          role="img"
          aria-label={
            audio || status === "capturing"
              ? "Waveform from the recorded audio"
              : "No audio recorded yet"
          }
          className="h-56 w-full sm:h-64"
        >
          <defs>
            <linearGradient
              id="oscilloscope-stroke"
              x1="0"
              y1="0"
              x2="1"
              y2="0"
            >
              <stop offset="0%" stopColor="#c20009" stopOpacity="0.35" />
              <stop offset="50%" stopColor="#f92b35" stopOpacity="1" />
              <stop offset="100%" stopColor="#c20009" stopOpacity="0.35" />
            </linearGradient>
          </defs>

          {GRID_COLUMNS.map((x) => (
            <line
              key={`column-${x}`}
              x1={x}
              y1="0"
              x2={x}
              y2="240"
              stroke="#ffffff"
              strokeOpacity="0.05"
            />
          ))}
          {GRID_ROWS.map((y) => (
            <line
              key={`row-${y}`}
              x1="0"
              y1={y}
              x2="800"
              y2={y}
              stroke="#ffffff"
              strokeOpacity="0.05"
            />
          ))}
          <line
            x1="0"
            y1="120"
            x2="800"
            y2="120"
            stroke="#ffffff"
            strokeOpacity="0.1"
            strokeDasharray="4 8"
          />
          {audio || status === "capturing" ? (
            <path
              d={waveformPath}
              fill="none"
              stroke="url(#oscilloscope-stroke)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ) : (
            <text
              x="400"
              y="126"
              textAnchor="middle"
              fill="#777777"
              fontSize="14"
            >
              Record a digit to see its waveform
            </text>
          )}
        </svg>

        <span className="pointer-events-none absolute left-4 top-3 text-[10px] uppercase tracking-[0.2em] text-text-muted">
          16 kHz mono
        </span>
        {status === "capturing" ? (
          <span className="pointer-events-none absolute bottom-3 right-4 text-[10px] uppercase tracking-[0.2em] text-text-muted">
            {(recordingProgress * MAX_RECORDING_MS / 1000).toFixed(1)} / 5.0 s
          </span>
        ) : audio ? (
          <span className="pointer-events-none absolute bottom-3 right-4 text-[10px] uppercase tracking-[0.2em] text-text-muted">
            {(audio.length / 16000).toFixed(2)} s
          </span>
        ) : null}
      </div>

      {audioUrl && (
        <div className="mt-4 rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
            {audioName ?? "Listen to your audio"}
          </p>
          <audio
            key={audioUrl}
            controls
            src={audioUrl}
            className="h-10 w-full"
          >
            Your browser does not support audio playback.
          </audio>
        </div>
      )}

      <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <ResultCard
          label="Digit"
          value={
            result
              ? rejected
                ? "Unknown"
                : String(result.digit)
              : isAnalyzing
                ? "Analyzing"
                : "Awaiting analysis"
          }
        />
        <ResultCard
          label="Confidence"
          value={
            result
              ? `${Math.round(result.confidence * 100)}%`
              : isAnalyzing
                ? "Pending"
                : "—"
          }
        />
        <ResultCard label="Model window" value="1.00 s" />
      </dl>

      {errorMessage && (
        <p role="alert" className="mt-4 text-sm text-primary-hover">
          {errorMessage}
        </p>
      )}

      <div className="mt-6 flex flex-wrap justify-end gap-3">
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          className="sr-only"
          aria-label="Upload an audio file"
          disabled={
            status === "capturing" || status === "processing" || isAnalyzing
          }
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            void handleAudioUpload(file);
          }}
        />
        {status !== "capturing" && (
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={status === "processing" || isAnalyzing}
            className="inline-flex items-center justify-center rounded-full border border-white/15 px-5 py-2.5 text-sm font-medium text-text-main transition-colors hover:border-primary/60 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {audio ? "Upload another audio" : "Upload audio"}
          </button>
        )}
        {audio && status !== "capturing" && (
          <>
            <button
              type="button"
              onClick={() => void analyzeRecording()}
              disabled={isAnalyzing || status === "processing"}
              className="inline-flex items-center justify-center rounded-full border border-primary/50 px-5 py-2.5 text-sm font-medium text-primary-hover transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isAnalyzing ? "Analyzing…" : "Analyze digit"}
            </button>
            <button
              type="button"
              onClick={() => void startRecording()}
              disabled={isAnalyzing || status === "processing"}
              className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-text-main transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50"
            >
              Record again
            </button>
          </>
        )}
        {status === "capturing" ? (
          <button
            type="button"
            onClick={stopRecording}
            className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-text-main transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            Stop recording
          </button>
        ) : (
          !audio && (
            <button
              type="button"
              onClick={() => void startRecording()}
              disabled={status === "processing"}
              className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-text-main transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50"
            >
              {status === "processing" ? "Preparing…" : "Start recording"}
            </button>
          )
        )}
      </div>
      <p className="mt-3 text-right text-xs text-text-muted">
        Say one digit, then stop. Recordings end automatically after 5 seconds.
      </p>
    </section>
  );
}

function buildWaveformPath(audio: Float32Array): string {
  const envelope = getWaveformEnvelope(audio, 400);
  const centerY = 120;
  const step = 800 / (envelope.length - 1);

  return envelope
    .map(({ min, max }, index) => {
      const x = index * step;
      const minY = centerY - min * 88;
      const maxY = centerY - max * 88;
      return `M${x.toFixed(1)},${minY.toFixed(1)}L${x.toFixed(1)},${maxY.toFixed(1)}`;
    })
    .join(" ");
}

function buildEnvelopePath(
  envelope: Array<{ min: number; max: number }>,
): string {
  if (envelope.length === 0) {
    return "";
  }

  let peak = 0;
  for (const point of envelope) {
    peak = Math.max(peak, Math.abs(point.min), Math.abs(point.max));
  }
  const scale = 88 / Math.max(peak, 0.015);
  const pointsPerRecording = MAX_RECORDING_MS / 50;

  return envelope
    .map(({ min, max }, index) => {
      const x = (index / pointsPerRecording) * 800;
      const minY = 120 - min * scale;
      const maxY = 120 - max * scale;
      return `M${x.toFixed(1)},${minY.toFixed(1)}L${x.toFixed(1)},${maxY.toFixed(1)}`;
    })
    .join(" ");
}

function ResultCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] px-4 py-3">
      <dt className="text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
        {label}
      </dt>
      <dd className="mt-1 break-words text-sm font-medium capitalize text-text-main">
        {value}
      </dd>
    </div>
  );
}
