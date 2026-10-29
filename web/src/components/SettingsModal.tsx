"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { CalibrationSettings } from "@/lib/onnxInference";

export type MicrophoneOption = {
  deviceId: string;
  label: string;
};

type SettingsModalProps = {
  open: boolean;
  microphones: MicrophoneOption[];
  selectedMicrophoneId: string;
  onMicrophoneChange: (deviceId: string) => void;
  microphoneError: string | null;
  isLoadingMicrophones: boolean;
  onRefreshMicrophones: () => Promise<void>;
  calibration: CalibrationSettings;
  onCalibrationChange: (calibration: CalibrationSettings) => void;
  onResetCalibration: () => void;
  onClearHistory: () => void;
  onClose: () => void;
};

export default function SettingsModal({
  open,
  microphones,
  selectedMicrophoneId,
  onMicrophoneChange,
  microphoneError,
  isLoadingMicrophones,
  onRefreshMicrophones,
  calibration,
  onCalibrationChange,
  onResetCalibration,
  onClearHistory,
  onClose,
}: SettingsModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const [microphoneMenuOpen, setMicrophoneMenuOpen] = useState(false);

  const handleClose = useCallback(() => {
    setMicrophoneMenuOpen(false);
    onClose();
  }, [onClose]);

  const toggleMicrophoneMenu = () => {
    const willOpen = !microphoneMenuOpen;
    setMicrophoneMenuOpen(willOpen);
    if (willOpen) {
      void onRefreshMicrophones();
    }
  };

  useEffect(() => {
    if (!open) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        handleClose();
      }
    };

    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    closeButtonRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [open, handleClose]);

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          key="settings-modal"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-end justify-center p-4 sm:items-center"
        >
          <button
            type="button"
            aria-label="Close settings"
            onClick={handleClose}
            className="absolute inset-0 h-full w-full cursor-default bg-black/70 backdrop-blur-sm"
          />

          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
            initial={{ opacity: 0, y: 24, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="relative max-h-[calc(100dvh-2rem)] w-full max-w-md overflow-y-auto rounded-2xl border border-white/10 bg-[#111111] shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4 border-b border-white/5 px-5 py-4">
              <div>
                <h2
                  id="settings-title"
                  className="text-base font-semibold tracking-tight text-text-main"
                >
                  Settings
                </h2>
                <p className="mt-1 text-xs text-text-muted">
                  Microphone input and local detection history.
                </p>
              </div>

              <button
                type="button"
                ref={closeButtonRef}
                onClick={handleClose}
                aria-label="Close settings"
                className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-text-muted transition-colors hover:border-primary/60 hover:text-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
              >
                <svg
                  aria-hidden="true"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  className="h-4 w-4"
                >
                  <path d="M6 6l12 12" />
                  <path d="M18 6L6 18" />
                </svg>
              </button>
            </div>

            <div className="flex flex-col gap-6 px-5 py-5">
              <div>
                <label
                  htmlFor="microphone-select"
                  className="text-[11px] font-medium uppercase tracking-[0.18em] text-text-muted"
                >
                  Microphone
                </label>

                <div className="relative mt-2">
                  <button
                    id="microphone-select"
                    type="button"
                    aria-haspopup="listbox"
                    aria-expanded={microphoneMenuOpen}
                    onClick={toggleMicrophoneMenu}
                    className="flex w-full items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2.5 text-left text-sm text-text-main transition-colors hover:border-white/20 focus:border-primary/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
                  >
                    <span className="truncate">
                      {microphones.find(
                        (microphone) =>
                          microphone.deviceId === selectedMicrophoneId,
                      )?.label ?? "System default microphone"}
                    </span>
                    {isLoadingMicrophones ? (
                      <span
                        aria-hidden="true"
                        className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-white/20 border-t-primary-hover"
                      />
                    ) : (
                      <svg
                        aria-hidden="true"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className={`h-4 w-4 shrink-0 text-text-muted transition-transform ${
                          microphoneMenuOpen ? "rotate-180" : ""
                        }`}
                      >
                        <path d="M6 9l6 6 6-6" />
                      </svg>
                    )}
                  </button>

                  {microphoneMenuOpen && (
                    <div
                      role="listbox"
                      aria-labelledby="microphone-select"
                      className="absolute left-0 right-0 z-10 mt-2 max-h-52 overflow-y-auto rounded-xl border border-white/10 bg-[#171717]/95 p-1.5 shadow-xl backdrop-blur-xl"
                    >
                      {isLoadingMicrophones ? (
                        <p className="px-3 py-2.5 text-sm text-text-muted">
                          Searching available microphones…
                        </p>
                      ) : microphones.map((microphone) => (
                        <button
                          key={microphone.deviceId}
                          type="button"
                          role="option"
                          aria-selected={
                            microphone.deviceId === selectedMicrophoneId
                          }
                          onClick={() => {
                            onMicrophoneChange(microphone.deviceId);
                            setMicrophoneMenuOpen(false);
                          }}
                          className={`flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover ${
                            microphone.deviceId === selectedMicrophoneId
                              ? "bg-primary/15 text-text-main"
                              : "text-text-muted hover:bg-white/[0.06] hover:text-text-main"
                          }`}
                        >
                          <span className="truncate">{microphone.label}</span>
                          {microphone.deviceId === selectedMicrophoneId && (
                            <span
                              aria-hidden="true"
                              className="h-1.5 w-1.5 rounded-full bg-primary-hover"
                            />
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <p className="mt-2 text-xs leading-5 text-text-muted">
                  The available devices are refreshed whenever you open this
                  list.
                </p>
                {microphoneError && (
                  <p role="alert" className="mt-2 text-xs text-primary-hover">
                    {microphoneError}
                  </p>
                )}
              </div>

              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <p className="text-sm font-medium text-text-main">
                  Confidence threshold
                </p>
                <p className="mt-1 text-xs leading-5 text-text-muted">
                  Predictions below this confidence are returned as Unknown.
                  Applied to listen and analyze modes.
                </p>
                <label
                  htmlFor="confidence-threshold-setting"
                  className="mt-4 flex items-center justify-between text-xs text-text-muted"
                >
                  <span>Minimum confidence</span>
                  <span className="font-mono tabular-nums text-text-main">
                    {calibration.confidenceThreshold.toFixed(2)}
                  </span>
                </label>
                <input
                  id="confidence-threshold-setting"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={calibration.confidenceThreshold}
                  onChange={(event) =>
                    onCalibrationChange({
                      ...calibration,
                      confidenceThreshold: Number(event.target.value),
                    })
                  }
                  className="mt-2 w-full accent-primary-hover"
                />
                <button
                  type="button"
                  onClick={onResetCalibration}
                  className="mt-3 w-full inline-flex items-center justify-center rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-text-muted transition-colors hover:border-primary/60 hover:text-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
                >
                  Reset to defaults
                </button>
              </div>
              

              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <p className="text-sm font-medium text-text-main">History</p>
                <p className="mt-1 text-xs leading-5 text-text-muted">
                  Remove every detection stored in this browser session. The
                  action cannot be undone.
                </p>

                <button
                  type="button"
                  onClick={onClearHistory}
                  className="mt-3 w-full inline-flex items-center justify-center rounded-full border border-primary/40 px-4 py-2 text-xs font-medium text-primary-hover transition-colors hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
                >
                  Clear history
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between gap-4 border-t border-white/5 px-5 py-4">
              <span className="text-[11px] text-text-muted">
                DigitSense v4.1
              </span>

              <button
                type="button"
                onClick={handleClose}
                className="inline-flex items-center justify-center rounded-full bg-primary px-5 py-2 text-sm font-medium text-text-main transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-[#111111]"
              >
                Done
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
