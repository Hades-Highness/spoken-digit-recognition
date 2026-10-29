import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Footer from "@/components/Footer";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DigitSense v4.1 | Spoken Digit Recognition",
  description:
    "Low-latency spoken digit recognition in the browser, powered by ONNX Runtime Web.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-background text-text-main">
        <div
          aria-hidden="true"
          className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
        >
          <div className="background-blob -left-32 -top-40 h-[26rem] w-[26rem] sm:h-[34rem] sm:w-[34rem]" />
          <div className="background-blob background-blob-vivid -right-40 top-1/3 h-[30rem] w-[30rem]" />
          <div className="background-blob -bottom-48 left-1/3 h-[24rem] w-[24rem]" />
        </div>

        <div className="relative z-10 flex flex-1 flex-col">
          {children}
          <Footer />
        </div>
      </body>
    </html>
  );
}
