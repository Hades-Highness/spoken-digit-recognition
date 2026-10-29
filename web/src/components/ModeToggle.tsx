"use client";

import { motion } from "framer-motion";

export type RecognitionMode = "listen" | "analyze";

const MODE_OPTIONS: ReadonlyArray<{ id: RecognitionMode; label: string }> = [
  { id: "listen", label: "Listen Mode" },
  { id: "analyze", label: "Analyze Mode" },
];

type ModeToggleProps = {
  mode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
};

export default function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
  return (
    <div
      role="group"
      aria-label="Recognition mode"
      className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.04] p-1 backdrop-blur-sm"
    >
      {MODE_OPTIONS.map((option) => {
        const isActive = option.id === mode;
        const isLisentMode = option.id === "listen";

        return (
          <button
            key={option.id}
            type="button"
            aria-pressed={isActive}
            onClick={() => onModeChange(option.id)}
            className={`relative rounded-full px-5 py-2 text-sm transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:px-6 ${
              isActive ? "text-black font-bold" : "text-text-muted font-medium hover:text-text-main"
            }`}
          >
            {isActive ? (
              <motion.span
                layoutId="mode-toggle-indicator"
                aria-hidden="true"
                className={`absolute inset-0 rounded-full  ${isLisentMode ? 'bg-gradient-to-r' : 'bg-gradient-to-l'} from-primary to-secondary`}
                transition={{ type: "spring", stiffness: 380, damping: 32 }}
              />
            ) : null}

            <span className="relative z-10 whitespace-nowrap">
              {option.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
