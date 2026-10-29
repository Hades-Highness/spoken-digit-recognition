"use client";

import { useSyncExternalStore } from "react";

const SCROLL_THRESHOLD_PX = 8;

function subscribeToScroll(onStoreChange: () => void) {
  window.addEventListener("scroll", onStoreChange, { passive: true });

  return () => window.removeEventListener("scroll", onStoreChange);
}

function getScrollSnapshot() {
  return window.scrollY > SCROLL_THRESHOLD_PX;
}

function getServerSnapshot() {
  return false;
}

type HeaderProps = {
  onOpenSettings: () => void;
};

export default function Header({ onOpenSettings }: HeaderProps) {
  const isScrolled = useSyncExternalStore(
    subscribeToScroll,
    getScrollSnapshot,
    getServerSnapshot,
  );

  return (
    <header
      className={`sticky top-0 z-40 w-full border-b transition-colors duration-300 ${
        isScrolled
          ? "border-white/10 bg-background/70 backdrop-blur-xl"
          : "border-transparent bg-transparent"
      }`}
    >
      <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <span className="flex items-center text-base tracking-tight sm:text-lg">
            <span className="text-text-main font-semibold">Digit</span>
            <span className="ml-1 rounded-[20px] font-bold bg-gradient-to-r from-primary to-secondary px-3 py-0.5 text-black">
              Sense
            </span>
          </span>

          <span className="rounded-full border border-white/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-text-muted">
            v4.1
          </span>
        </div>

        <button
          type="button"
          onClick={onOpenSettings}
          aria-label="Open settings"
          className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/[0.02] text-text-muted transition-colors hover:border-primary/70 hover:text-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            className="h-5 w-5"
          >
            <path d="M4 7h9" />
            <path d="M17 7h3" />
            <path d="M4 17h3" />
            <path d="M11 17h9" />
            <circle cx="15" cy="7" r="2" />
            <circle cx="9" cy="17" r="2" />
          </svg>
        </button>
      </div>
    </header>
  );
}
