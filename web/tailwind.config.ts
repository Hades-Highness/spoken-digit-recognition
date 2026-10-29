import type { Config } from "tailwindcss";

/**
 * DigitSense v4.1 design tokens.
 *
 * Tailwind CSS v4 is configured CSS-first, so this legacy config file is loaded
 * through the `@config` directive in `src/app/globals.css`. Keeping the tokens
 * here makes the palette and the motion primitives available to any
 * JavaScript-based tooling that reads the Tailwind config.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0d0d0d",
        primary: {
          DEFAULT: "#c20009",
          hover: "#f92b35",
        },
        secondary: {
          DEFAULT: "#f92b35",
          hover: "#c20009",
        },
        "text-main": "#ffffff",
        "text-muted": "#a1a1aa",
      },
      fontFamily: {
        sans: [
          "var(--font-geist-sans)",
          "ui-sans-serif",
          "system-ui",
          "sans-serif",
        ],
        mono: [
          "var(--font-geist-mono)",
          "ui-monospace",
          "SFMono-Regular",
          "monospace",
        ],
      },
      keyframes: {
        "pulse-slow": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.65", transform: "scale(1.04)" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-slow": "pulse-slow 3.2s ease-in-out infinite",
        "fade-in-up": "fade-in-up 0.45s ease-out both",
      },
      boxShadow: {
        "primary-glow": "0 0 60px -12px rgba(194, 0, 9, 0.55)",
      },
    },
  },
  plugins: [],
};

export default config;
