"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { CSSProperties } from "react";

const CIRCULAR_WAVE_BARS = Array.from({ length: 48 }, (_, index) => ({
  angle: (index / 48) * 360,
  height: 7 + ((Math.sin(index * 1.73) + 1) / 2) * 9,
  amplitude: 0.45 + ((Math.sin(index * 0.61) + 1) / 2) * 0.55,
}));

export type HistoryEntry = {
  id: string;
  digit: number;
  /** Confidence in the 0..1 range, rendered as a percentage. */
  confidence: number;
  /** Pre-formatted wall-clock time, for example "14:02:11". */
  timestamp: string;
};

type ListenVisualizerProps = {
  isListening: boolean;
  /** True while the model is loading or the microphone is opening. */
  isBusy: boolean;
  /** Normalized microphone signal level in the 0..1 range. */
  audioLevel: number;
  /** One-line explanation of the current state. */
  statusMessage: string;
  /** Replaces the status line when the session failed. */
  errorMessage: string | null;
  /** Detections in reverse chronological order. */
  history: HistoryEntry[];
  onToggleListening: () => void;
};

export default function ListenVisualizer({
  isListening,
  isBusy,
  audioLevel,
  statusMessage,
  errorMessage,
  history,
  onToggleListening,
}: ListenVisualizerProps) {
  return (
    <section
      aria-label="Listen mode"
      className="flex w-full flex-col overflow-hidden rounded-3xl border border-white/10 bg-[#111111] sm:flex-row"
    >
      <div className="relative flex min-w-0 flex-1 flex-col items-center justify-center gap-6 px-2 py-10 sm:gap-8 sm:px-4 sm:py-12">
        <p className="absolute left-4 top-4 text-[12px] font-medium uppercase tracking-[0.22em] text-text-muted sm:left-6 sm:top-6">
          Listen Mode
        </p>

        <div className="relative flex h-40 w-40 items-center justify-center sm:h-64 sm:w-64">
          <span
            aria-hidden="true"
            className={`absolute inset-0 rounded-full border border-primary/20 transition-opacity duration-500 ${
              isListening ? "animate-pulse-slow opacity-100" : "opacity-40"
            }`}
          />
          <span
            aria-hidden="true"
            className="absolute inset-2 rounded-full border border-white/5 sm:inset-6"
          />
          <span
            aria-hidden="true"
            className={`absolute inset-5 rounded-full bg-primary/20 blur-2xl transition-opacity duration-500 sm:inset-10 ${
              isListening ? "opacity-100" : "opacity-0"
            }`}
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 [--wave-radius:61px] sm:[--wave-radius:101px]"
            style={{ "--audio-level": audioLevel } as CSSProperties}
          >
            {CIRCULAR_WAVE_BARS.map((bar, index) => (
              <span
                key={index}
                className="absolute left-1/2 top-1/2 w-[2px] rounded-full bg-gradient-to-t from-primary/50 to-primary-hover transition-[transform,opacity] duration-100 motion-reduce:transition-none sm:w-[3px]"
                style={{
                  height: `${bar.height}px`,
                  opacity: isListening ? 0.35 + audioLevel * 0.65 : 0.2,
                  transform: `translate(-50%, -50%) rotate(${bar.angle}deg) translateY(calc(-1 * var(--wave-radius))) scaleY(calc(${bar.amplitude} * (0.25 + var(--audio-level, 0) * 2.4)))`,
                }}
              />
            ))}
          </div>

          <button
            type="button"
            onClick={onToggleListening}
            disabled={isBusy}
            aria-busy={isBusy}
            aria-pressed={isListening}
            aria-label={
              isBusy
                ? "Preparing the recognition session"
                : isListening
                  ? "Stop listening"
                  : "Start listening"
            }
            className={`relative z-10 flex h-20 w-20 items-center justify-center rounded-full border backdrop-blur-md transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-4 focus-visible:ring-offset-background disabled:cursor-progress sm:h-36 sm:w-36 ${
              isListening
                ? "border-primary-hover bg-primary/20 shadow-[0_0_60px_-10px_rgba(249,43,53,0.65)]"
                : "border-white/15 bg-[#111111]/45 hover:border-primary/60 hover:bg-primary/10"
            }`}
          >
            <span
              aria-hidden="true"
              className={`absolute inset-1.5 rounded-full border sm:inset-2 ${
                isListening
                  ? "animate-pulse-slow border-primary-hover/40"
                  : "border-white/5"
              }`}
            />
            <MicrophoneIcon
              className={`h-6 w-6 transition-colors duration-300 sm:h-9 sm:w-9 ${
                isListening ? "text-text-main" : "text-text-muted"
              }`}
            />
          </button>
        </div>

        <h2 className="max-w-[10rem] text-center text-base font-semibold tracking-tight text-text-main sm:max-w-sm sm:text-2xl">
          {isBusy
            ? "Preparing the model"
            : isListening
              ? "Speak a digit"
              : "Start listening"}
        </h2>

        <p
          role="status"
          className={`max-w-[13rem] text-center text-[11px] leading-4 sm:max-w-sm sm:text-xs ${
            errorMessage ? "text-primary-hover" : "text-text-muted"
          }`}
        >
          {errorMessage ?? statusMessage}
        </p>
      </div>

      <aside
        aria-labelledby="history-heading"
        className="flex w-full shrink-0 flex-col border-t border-white/10 px-2 sm:px-4 py-4 sm:w-[200px] sm:border-l sm:border-t-0 sm:py-6"
      >
        <h2
          id="history-heading"
          className="text-[12px] font-medium uppercase tracking-[0.18em] text-text-muted"
        >
          History
        </h2>

        <div className="relative mt-3 min-h-[120px] sm:min-h-[400px]">
          <AnimatePresence mode="wait" initial={false}>
            {history.length === 0 ? (
              <motion.div
                key="empty-state"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="flex h-full min-h-[120px] flex-col items-center justify-center gap-2 px-2 text-center sm:min-h-[400px]"
              >
                <p className="text-[11px] leading-4 text-text-muted">
                  No digits yet
                  <br />
                  <span className="text-[10px] opacity-70">
                    Start speaking to see your history
                  </span>
                </p>
              </motion.div>
            ) : (
              <motion.ul
                key="history-list"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                aria-live="polite"
                className="history-scroll grid max-h-[400px] grid-cols-2 gap-x-4 gap-y-1.5 overflow-x-hidden overflow-y-auto pr-0.5 sm:flex sm:flex-col sm:gap-x-0"
              >
                <AnimatePresence initial={false}>
                  {history.map((entry) => (
                    <HistoryRow key={entry.id} entry={entry} />
                  ))}
                </AnimatePresence>
              </motion.ul>
            )}
          </AnimatePresence>
        </div>
      </aside>
    </section>
  );
}

type HistoryRowProps = {
  entry: HistoryEntry;
};

function HistoryRow({ entry }: HistoryRowProps) {
  const confidencePercent = Math.round(entry.confidence * 100);

  return (
    <motion.li
      layout="position"
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className="group flex items-baseline justify-between gap-1.5"
    >
      <span className="min-w-0 truncate text-sm font-medium capitalize leading-5 text-text-main">
        {entry.digit}
      </span>

      <span className="digit-details shrink-0 whitespace-nowrap text-[11px] leading-4 tabular-nums text-text-muted opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-within:opacity-100 motion-reduce:transition-none">
        {confidencePercent}% <span aria-hidden="true">•</span> {entry.timestamp}
      </span>
    </motion.li>
  );
}

function MicrophoneIcon({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect x="9" y="2.75" width="6" height="11" rx="3" />
      <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0" />
      <path d="M12 18v3" />
      <path d="M8.5 21h7" />
    </svg>
  );
}