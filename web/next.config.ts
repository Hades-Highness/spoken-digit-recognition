import type { NextConfig } from "next";

/**
 * Cross-origin isolation headers required by the multithreaded ONNX Runtime Web
 * WASM backend (public/ort/ort-wasm-simd-threaded.wasm).
 *
 * The threaded build allocates its linear memory as a SharedArrayBuffer so the
 * worker pool can operate on the same heap. Browsers refuse to expose
 * SharedArrayBuffer unless the document is cross-origin isolated, and a document
 * only becomes isolated when its own response carries BOTH of these headers:
 *
 * - Cross-Origin-Opener-Policy: same-origin  -> severs the browsing context
 *   group, so no other origin can hold a reference to this window.
 * - Cross-Origin-Embedder-Policy: require-corp -> every cross-origin subresource
 *   must opt in explicitly through CORP or CORS.
 *
 * Without them `crossOriginIsolated` stays false, ONNX Runtime Web silently
 * falls back to a single thread and the threaded artefacts become unusable on
 * any host (tunnels, LAN IPs, preview deployments) that does not add the headers
 * itself. The app only loads same-origin resources (model, /ort WASM assets,
 * self-hosted fonts), so the strict embedder policy is safe here.
 */
const crossOriginIsolationHeaders = [
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
];

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  /**
   * Applied to every route, including the files served from public/. Headers are
   * matched before the filesystem, and same-origin assets must return the same
   * policy so the document keeps its isolation after the first navigation.
   */
  async headers() {
    return [
      {
        source: "/:path*",
        headers: crossOriginIsolationHeaders,
      },
      {
        source: '/ort/:path*',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=3000000, immutable' }],
      },
      {
        source: '/models/:path*',
        headers: [{ key: 'Cache-Control', value: 'public, max-age=3000000, immutable' }],
      },
    ];
  },
};

export default nextConfig;
