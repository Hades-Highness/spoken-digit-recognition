import { DIGIT_LABELS } from "@/lib/onnxInference";

const specs = [
  { label: "Input", value: "16 kHz mono Float32" },
  { label: "Format", value: "ONNX (WASM)" },
  { label: "Classes", value: "10 digits (0-9)" },
  { label: "Unknown gate", value: "0.60 default" },
];

export default function ModelInfo() {
  return (
    <section
      aria-labelledby="model-info-title"
      className="mt-30 rounded-2xl border border-white/10 bg-[#111111] p-5 sm:p-6"
    >
      <div>
        <h2
          id="model-info-title"
          className="text-[12px] font-medium uppercase tracking-[0.22em] text-text-muted"
        >
          About Our Model
        </h2>
        <p className="text-[12px] font-medium opacity-70 text-text-muted">
          A lightweight, on-device spoken digit recognizer designed for
          WebAssembly in your browser.
        </p>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        {specs.map(({ label, value }) => (
          <div
            key={label}
            className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-3"
          >
            <dt className="text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
              {label}
            </dt>
            <dd className="mt-1 text-sm font-medium text-text-main">{value}</dd>
          </div>
        ))}
      </dl>

      <div className="mt-5 border-t border-white/10 pt-4">
        <h3 className="text-[12px] font-medium uppercase tracking-[0.22em] text-text-muted">
          Supported Digits
        </h3>
        <ul className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-5 md:grid-cols-7">
          {DIGIT_LABELS.map((digit) => (
            <li
              key={digit}
              className="rounded border border-white/5 bg-white/5 px-2 py-1 text-center font-mono text-xs text-gray-300 transition-colors hover:border-[#f92b35]/30 hover:bg-white/10"
            >
              {digit}
            </li>
          ))}
        </ul>
      </div>

      <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-5 text-text-muted">
        The current Python pipeline uses noise reduction, voice activity
        detection, and log-mel feature extraction. These steps are too heavy to
        reproduce in browser JavaScript. The v4.1 raw-audio ONNX export is
        pending; this client sends raw audio and requires that export to include
        or accept the preprocessing.
      </p>

      <div className="mt-5 flex items-center justify-between gap-4 border-t border-white/10 pt-4 text-xs">
        <p className="text-text-muted">Powered by ONNX Runtime Web</p>
        <a
          href="https://github.com/Hades-Highness/spoken-digit-recognition"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-text-main transition-colors hover:text-secondary focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
        >
          View Repository
          <svg
            aria-hidden="true"
            className="h-3.5 w-3.5"
            viewBox="0 0 16 16"
            fill="none"
          >
            <path
              d="M6 3h7v7M13 3 7 9"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="1.5"
            />
            <path
              d="M11 9v3.25a.75.75 0 0 1-.75.75h-7.5a.75.75 0 0 1-.75-.75v-7.5A.75.75 0 0 1 2.75 4H6"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="1.5"
            />
          </svg>
        </a>
      </div>
    </section>
  );
}
