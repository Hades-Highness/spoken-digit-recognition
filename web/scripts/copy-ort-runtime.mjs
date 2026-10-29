/**
 * Copies the ONNX Runtime Web WASM artifacts into public/ort so the browser can
 * fetch them from a stable URL (ort.env.wasm.wasmPaths = "/ort/").
 *
 * Only the CPU (WASM) execution provider is needed, so the asyncify, jsep and
 * jspi variants are unused. The simd-threaded build starts a worker pool because
 * web/next.config.ts enables cross-origin isolation (COOP and COEP), which is
 * what unlocks SharedArrayBuffer for the WASM threads.
 * Model weights are NOT handled here; those are release artifacts placed in
 * public/models/ manually.
 */
import { copyFile, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const sourceDir = join(projectRoot, "node_modules", "onnxruntime-web", "dist");
const targetDir = join(projectRoot, "public", "ort");

const ARTIFACTS = [
  "ort.wasm.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];
const LOADER = `import * as ort from "./ort.wasm.min.mjs";
window.__digitSenseOrt = ort;
`;

async function isCurrent(source, target) {
  try {
    const [sourceStats, targetStats] = await Promise.all([
      stat(source),
      stat(target),
    ]);
    return sourceStats.size === targetStats.size;
  } catch {
    return false;
  }
}

async function main() {
  await mkdir(targetDir, { recursive: true });
  let copied = 0;

  for (const name of ARTIFACTS) {
    const source = join(sourceDir, name);
    const target = join(targetDir, name);

    if (await isCurrent(source, target)) {
      continue;
    }

    await copyFile(source, target);
    copied += 1;
    console.log(`[ort] copied ${name} to public/ort`);
  }

  await writeFile(join(targetDir, "ort-runtime-loader.mjs"), LOADER);

  if (copied === 0) {
    console.log("[ort] public/ort is up to date");
  }
}

main().catch((error) => {
  console.error("[ort] failed to prepare the ONNX Runtime assets:", error);
  process.exitCode = 1;
});
