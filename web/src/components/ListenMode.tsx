"use client";

import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type Ref,
} from "react";
import ListenVisualizer, {
  type HistoryEntry,
} from "@/components/ListenVisualizer";
import { AudioProcessor } from "@/lib/audioProcessor";
import {
  loadModel,
  type CalibrationSettings,
  type InferenceResult,
} from "@/lib/onnxInference";

/** Overlapping windows can repeat a detection; ignore the same digit this long. */
const DUPLICATE_WINDOW_MS = 1000;

/** Newest entries are kept; the panel is a scrolling list, not a transcript. */
const MAX_HISTORY_ENTRIES = 50;

type SessionStatus = "idle" | "loading" | "listening" | "error";

const STATUS_MESSAGES: Record<SessionStatus, string> = {
  idle: "Tap the microphone to start listening.",
  loading: "Loading DigitSense v4.1 and opening the microphone.",
  listening: "Listening at 16 kHz. Detections appear in the history panel.",
  error: "The recognition session stopped.",
};

export type ListenModeHandle = {
  clearHistory: () => void;
};

type ListenModeProps = {
  ref?: Ref<ListenModeHandle>;
  /** Device selected in Settings; "default" follows the system input. */
  microphoneId: string;
  calibration: CalibrationSettings;
};

/**
 * Owns the listen-mode session: the ONNX session (through the singleton in
 * lib/onnxInference), the microphone pipeline and the detection history.
 *
 * ListenVisualizer stays presentational, so the expensive audio objects
 * never end up in component state or re-render paths.
 */
export default function ListenMode({
  ref,
  microphoneId,
  calibration,
}: ListenModeProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);

  const processorRef = useRef<AudioProcessor | null>(null);
  const mountedRef = useRef(true);
  const sessionTokenRef = useRef(0);
  const entryIdRef = useRef(0);
  const lastAcceptedRef = useRef<{ digit: number; at: number } | null>(null);

  useImperativeHandle(
    ref,
    () => ({
      clearHistory: () => setHistory([]),
    }),
    [],
  );

  useEffect(() => {
    processorRef.current?.setCalibration(calibration);
  }, [calibration]);

  const handleResult = useCallback((result: InferenceResult) => {
    if (result.digit === "Unknown") {
      return;
    }

    const now = Date.now();
    const lastAccepted = lastAcceptedRef.current;
    if (
      lastAccepted &&
      lastAccepted.digit === result.digit &&
      now - lastAccepted.at < DUPLICATE_WINDOW_MS
    ) {
      return;
    }
    lastAcceptedRef.current = { digit: result.digit, at: now };

    entryIdRef.current += 1;
    const entry: HistoryEntry = {
      id: `${now}-${entryIdRef.current}`,
      digit: result.digit,
      confidence: result.confidence,
      timestamp: new Date(now).toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };

    setHistory((current) => [entry, ...current].slice(0, MAX_HISTORY_ENTRIES));
  }, []);

  const handleError = useCallback((error: Error) => {
    processorRef.current = null;
    if (!mountedRef.current) {
      return;
    }
    setStatus("error");
    setErrorMessage(error.message);
  }, []);

  const stopListening = useCallback(async () => {
    sessionTokenRef.current += 1;

    const processor = processorRef.current;
    processorRef.current = null;
    if (processor) {
      await processor.stop();
    }

    if (mountedRef.current) {
      setStatus("idle");
      setErrorMessage(null);
    }
  }, []);

  const startListening = useCallback(async () => {
    sessionTokenRef.current += 1;
    const token = sessionTokenRef.current;

    setStatus("loading");
    setErrorMessage(null);
    lastAcceptedRef.current = null;

    try {
      await loadModel();
      if (!mountedRef.current || sessionTokenRef.current !== token) {
        return;
      }

      const processor = new AudioProcessor({
        deviceId: microphoneId,
        onResult: handleResult,
        onAudioLevel: setAudioLevel,
        onError: handleError,
        calibration,
      });
      await processor.start();

      // The user may have pressed stop, switched modes or closed the tab while
      // the microphone permission prompt was open.
      if (!mountedRef.current || sessionTokenRef.current !== token) {
        await processor.stop();
        return;
      }

      // A worklet error during startup tears the graph down and reports through
      // onError; do not overwrite that state with "listening".
      if (!processor.isRunning) {
        return;
      }

      processorRef.current = processor;
      setStatus("listening");
    } catch (error) {
      if (!mountedRef.current || sessionTokenRef.current !== token) {
        return;
      }
      setStatus("error");
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "The recognition session could not start.",
      );
    }
  }, [calibration, handleError, handleResult, microphoneId]);

  const handleToggleListening = useCallback(() => {
    if (status === "loading") {
      return;
    }
    if (status === "listening") {
      void stopListening();
      return;
    }
    void startListening();
  }, [startListening, status, stopListening]);

  // A device change invalidates the open AudioContext, so end the session and
  // let the user restart with the new input.
  useEffect(() => {
    if (!processorRef.current) {
      return;
    }
    void stopListening();
  }, [microphoneId, stopListening]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      sessionTokenRef.current += 1;
      const processor = processorRef.current;
      processorRef.current = null;
      void processor?.stop();
    };
  }, []);

  return (
    <ListenVisualizer
      isListening={status === "listening"}
      isBusy={status === "loading"}
      audioLevel={audioLevel}
      statusMessage={STATUS_MESSAGES[status]}
      errorMessage={errorMessage}
      history={history}
      onToggleListening={handleToggleListening}
    />
  );
}
