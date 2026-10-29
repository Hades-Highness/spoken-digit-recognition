"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion, MotionConfig } from "framer-motion";
import ListenMode, {
  type ListenModeHandle,
} from "@/components/ListenMode";
import Header from "@/components/Header";
import ModeToggle, { type RecognitionMode } from "@/components/ModeToggle";
import SettingsModal, {
  type MicrophoneOption,
} from "@/components/SettingsModal";
import AnalyzeMode from "@/components/AnalyzeMode";
import ModelInfo from "@/components/ModelInfo";
import {
  CONFIDENCE_THRESHOLD,
  type CalibrationSettings,
} from "@/lib/onnxInference";

const MICROPHONES: MicrophoneOption[] = [
  { deviceId: "default", label: "System default microphone" },
];

export default function Home() {
  const [mode, setMode] = useState<RecognitionMode>("listen");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [microphoneId, setMicrophoneId] = useState(MICROPHONES[0].deviceId);
  const [microphones, setMicrophones] = useState(MICROPHONES);
  const [microphoneError, setMicrophoneError] = useState<string | null>(null);
  const [isLoadingMicrophones, setIsLoadingMicrophones] = useState(false);
  const [calibration, setCalibration] = useState<CalibrationSettings>({
    confidenceThreshold: CONFIDENCE_THRESHOLD,
  });
  const listenRef = useRef<ListenModeHandle>(null);

  const handleModeChange = (nextMode: RecognitionMode) => {
    setMode(nextMode);
  };

  const handleClearHistory = useCallback(() => {
    listenRef.current?.clearHistory();
  }, []);

  const handleOpenSettings = useCallback(() => {
    setSettingsOpen(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setSettingsOpen(false);
  }, []);

  const handleRefreshMicrophones = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicrophoneError(
        "This browser does not support microphone device discovery.",
      );
      return;
    }

    setIsLoadingMicrophones(true);
    setMicrophoneError(null);
    let temporaryStream: MediaStream | null = null;
    try {
      let inputs = (await navigator.mediaDevices.enumerateDevices()).filter(
        (device) => device.kind === "audioinput",
      );
      if (inputs.some((device) => !device.label)) {
        temporaryStream = await navigator.mediaDevices.getUserMedia({
          audio: true,
          video: false,
        });
        inputs = (await navigator.mediaDevices.enumerateDevices()).filter(
          (device) => device.kind === "audioinput",
        );
      }
      const options = [
        { deviceId: "default", label: "System default microphone" },
        ...inputs
          .filter((device) => device.deviceId && device.deviceId !== "default")
          .map((device, index) => ({
            deviceId: device.deviceId,
            label: device.label || `Microphone ${index + 1}`,
          })),
      ];
      setMicrophones(
        options.filter(
          (option, index) =>
            options.findIndex(
              (candidate) => candidate.deviceId === option.deviceId,
            ) === index,
        ),
      );
      setMicrophoneId((current) =>
        options.some((option) => option.deviceId === current)
          ? current
          : "default",
      );
    } catch (error) {
      setMicrophoneError(
        error instanceof Error
          ? error.message
          : "Microphone devices could not be loaded.",
      );
    } finally {
      temporaryStream?.getTracks().forEach((track) => track.stop());
      setIsLoadingMicrophones(false);
    }
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <div className="flex flex-1 flex-col">
        <Header onOpenSettings={handleOpenSettings} />

        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 pb-12 pt-10 sm:px-6 sm:pt-12 lg:px-8">
          <h1 className="sr-only">DigitSense spoken digit recognition</h1>

          <div className="flex justify-center">
            <ModeToggle mode={mode} onModeChange={handleModeChange} />
          </div>

          <div className="mt-8 flex flex-1 flex-col">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="w-full"
              >
                {mode === "listen" ? (
                  <ListenMode
                    ref={listenRef}
                    microphoneId={microphoneId}
                    calibration={calibration}
                  />
                ) : (
                  <AnalyzeMode
                    microphoneId={microphoneId}
                    calibration={calibration}
                  />
                )}
              </motion.div>
            </AnimatePresence>
          </div>
          <ModelInfo />
        </main>

        <SettingsModal
          open={settingsOpen}
          microphones={microphones}
          selectedMicrophoneId={microphoneId}
          onMicrophoneChange={setMicrophoneId}
          microphoneError={microphoneError}
          isLoadingMicrophones={isLoadingMicrophones}
          onRefreshMicrophones={handleRefreshMicrophones}
          calibration={calibration}
          onCalibrationChange={setCalibration}
          onResetCalibration={() =>
            setCalibration({
              confidenceThreshold: CONFIDENCE_THRESHOLD,
            })
          }
          onClearHistory={handleClearHistory}
          onClose={handleCloseSettings}
        />
      </div>
    </MotionConfig>
  );
}
