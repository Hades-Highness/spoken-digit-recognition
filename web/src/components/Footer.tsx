"use client";

const socialLinks = [
  { label: "GitHub", href: "https://github.com/Hades-Highness" },
  { label: "LinkedIn", href: "https://www.linkedin.com/in/mohamed-charif-moussaoui-aab355294/" },
  { label: "Instagram", href: "https://www.instagram.com/hades_highness/" },
];

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-white/10 bg-background/40">
      <div className="mx-auto flex w-full max-w-7xl flex-col items-center gap-4 px-4 py-6 text-center text-xs text-text-muted sm:px-6 md:grid md:grid-cols-[1fr_auto_1fr] md:gap-6 md:text-left lg:px-8">
        <p className="md:justify-self-start">
          © 2026 Hades-Highness, All rights reserved
        </p>

        <nav
          aria-label="Social links"
          className="flex items-center gap-2 md:justify-self-center"
        >
          {socialLinks.map(({ label, href }, index) => (
            <span key={label} className="flex items-center gap-2">
              {index > 0 && <span aria-hidden="true">•</span>}
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="transition-colors hover:text-secondary focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover"
              >
                {label}
              </a>
            </span>
          ))}
        </nav>

        <a
          href="https://github.com/Hades-Highness/spoken-digit-recognition"
          target="_blank"
          rel="noreferrer"
          className="transition-colors hover:text-text-main focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-hover md:justify-self-end"
        >
          DigitSense v4.1
        </a>
      </div>
    </footer>
  );
}