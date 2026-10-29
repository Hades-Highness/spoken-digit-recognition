# Model Card — DigitSense v4.1

| Field | Value |
| :--- | :--- |
| Version | **v4.1 — web inference export, pending** |
| Git tag | Not released |
| Checkpoints | Planned raw-audio ONNX artifact: `web/public/models/DigitSense_v4.1.onnx` |
| Status | Web client implemented; compatible raw-audio ONNX export and validation are pending |
| Task | Spoken digit classification, 10 classes (0–9), English |
| Dataset | Intended to retain the v4.0 AudioMNIST domain; export/evaluation details pending |
| Split | v4.0 split: 50 training speakers / 5 validation (`03`, `04`, `05`, `12`, `26`) / 5 test (`01`, `02`, `07`, `28`, `47`) |
| Sample rate | 16 kHz |
| Input | Raw mono Float32 audio, one second (`[1, 16000]`); requires a compatible ONNX export |
| Output | 10 logits, class order 0 through 9 |
| Execution | ONNX Runtime Web, WebAssembly (WASM) |
| Rejection | Return `Unknown` when the maximum softmax confidence is below 0.60 by default; threshold is configurable |
| Metrics | Pending raw-audio export and evaluation |

## Why this version exists

DigitSense v4.0 uses Python preprocessing before the CNN: `noisereduce`,
Silero VAD, and librosa log-mel, delta, and delta-delta feature extraction.
That pipeline is too heavy to reproduce in browser JavaScript and would add
latency to a web inference path.

DigitSense v4.1 defines a browser-facing ONNX interface that accepts raw 16 kHz
mono Float32 audio and executes in WebAssembly. The web client performs only
audio capture, anti-alias filtering, resampling, and fixed-window framing. The
ONNX export must either include the preprocessing required by its weights or
be updated to operate on raw audio. The current v4.0 feature-input ONNX export
does not satisfy this contract.

## Configuration

The web client's target interface and inference settings are:

| Setting | Value |
| :--- | :--- |
| Input tensor | `raw_audio`, float32, shape `[1, 16000]`, mono samples in `[-1, 1]` |
| Output tensor | `logits`, float32, shape `[1, 10]` |
| Sample rate | 16 kHz |
| Window | 1.0 second |
| Class order | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` |
| Runtime | ONNX Runtime Web, WASM execution provider |
| Confidence | Numerically stable softmax over the ten logits |
| Default rejection threshold | 0.60; configurable in the web client |
| Low-confidence result | `Unknown` |
| JavaScript model preprocessing | None; no VAD, denoising, or spectrogram generation |

These are the v4.1 web interface requirements, not verified properties of an
available model artifact. The existing v4.0 checkpoint and ONNX export continue
to require the Python feature pipeline.

## Results

No v4.1 raw-audio artifact has been exported or evaluated. Accuracy, latency,
model size, and per-class metrics are pending. The v4.0 metrics below are not
evidence of v4.1 browser performance.

### Classification report

Pending raw-audio export and evaluation.

### Training curves

No v4.1 training run is associated with this web export. Any future model
training curves should be added after the raw-audio model is trained and
validated.

### Confusion matrix

Pending raw-audio export and evaluation.

## Interpretation

The v4.1 web application defines a low-overhead browser pipeline and a
confidence-based rejection rule. Its predictions cannot be considered
validated until a compatible raw-audio ONNX model is available and evaluated
with the intended WebAssembly runtime. The 0.60 threshold is a configurable
default, not a threshold calibrated on a v4.1 validation split.

## Caveats and known issues

- **The raw-audio export is pending.** The current v4.0 ONNX graph expects
  precomputed features, so it cannot be loaded by this web client.
- **The current Python preprocessing is not implemented in JavaScript.**
  `noisereduce`, Silero VAD, and librosa log-mel plus delta feature extraction
  are too heavy for the browser path. The v4.1 graph must incorporate the
  required preprocessing or the model must be updated for raw audio.
- **No v4.1 browser metrics are available.** The v4.0 test accuracy does not
  establish raw-audio or WASM accuracy, latency, or compatibility.
- **There are no explicit silence or unknown classes.** `Unknown` is a
  client-side decision based solely on the maximum softmax confidence being
  below the configured threshold; it is not a learned class.
- **Domain.** The intended task remains isolated English digit words from
  AudioMNIST. Continuous speech, noisy environments, and far-field audio have
  not been validated for the web pipeline.

## Artifacts

| File | Content |
| :--- | :--- |
| `web/src/lib/onnxInference.ts` | WASM runtime loading, raw-audio input, softmax, and confidence rejection |
| `web/src/lib/audioProcessor.ts` | AudioWorklet capture, resampling, and raw 16 kHz windowing |
| `web/public/models/DigitSense_v4.1.onnx` | Planned release artifact; not yet available |
| `web/README.md` | Browser setup and export compatibility requirements |
| `models/digitsense_v4.0.pth` | Existing v4.0 weights; uses the Python feature pipeline |
| `onnx/digitsense_v4.0.onnx` | Existing v4.0 feature-input ONNX export; incompatible with the v4.1 web input |
