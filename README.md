# 🎙️ FSDD - Spoken Digit Classification

Spoken digit recognition pipeline (digits 0–9) based on the **FSDD** (*Free Spoken Digit Dataset*).  
This project documents the transition from a naive baseline to a robust, **speaker-independent** CNN architecture.

---

## 🛠️ Audio Technical Specifications

* **Sampling Rate ($f_s$)**: 8,000 Hz (mono)
* **Normalized Duration**: 1.0 second (8,000 samples)
* **Spectral Representation**: Log-Mel Spectrogram
  * `n_mels`: 64
  * `n_fft`: 512 (~64 ms window)
  * `hop_length`: 256 (~32 ms step)
  * **Input Tensor Shape**: `[Batch, 1, 64, 32]`

---

## 📜 Version History & Project Evolution

### 🔹 Version 1.0 — Baseline (Random Split)
* **Goal**: Validate the core training pipeline with a basic CNN architecture.
* **Data Split**: Random 80/20 train/val split mixing all speakers.
* **Outcome**: High accuracy (~98%+), but severely **overfitted to speaker identity**. The model memorized specific voice signatures present in both train and val sets instead of learning phoneme geometry.

---

### 🔹 Version 2.0 — Speaker-Independent Split
* **Goal**: Evaluate true generalization on completely unseen voices.
* **New Split (Strictly separated by speaker)**:
  * **Train Set (4 speakers / 2,000 clips)**: `jackson`, `nicolas`, `theo`, `yweweler`
  * **Validation Set (1 speaker / 500 clips)**: `lucas`
  * **Test Set (1 speaker / 500 clips)**: `george`
* **Observation**: Validation accuracy dropped sharply. The model failed to generalize to `lucas` because it relied heavily on the pitch and fundamental frequencies of the 4 training voices.

---

### 🔹 Version 2.1 — Data Augmentation (Acoustic Perturbations)
* **Goal**: Artificially expand acoustic diversity across the 4 training speakers.
* **Techniques Introduced**:
  * **Speed / Pitch Shift**: Dynamic linear interpolation ($\pm 10\%$).
  * **Additive White Noise**: Low-level Gaussian noise injection ($20\%$ chance).
  * **SpecAugment**: Random frequency (`FrequencyMasking`) and time (`TimeMasking`) band erasure.
* **Key Finding**: Overly aggressive pitch shifting distorted vocal formants, causing training instability. Augmentation parameters were calibrated down to preserve phoneme integrity.

---

### 🔹 Version 2.2 — Speaker Invariance & Regularization
* **Goal**: Neutralize speaker identity (timbre/pitch/volume) to force the network to focus strictly on digit phonemes.
* **Major Upgrades**:
  1. **Per-Sample Instance Standardization**:  
     Zero-mean unit-variance scaling per spectrogram $X_{norm} = \frac{X - \mu}{\sigma + \epsilon}$ inside `dataset.py`.
  2. **`InstanceNorm2d` Layers in `model.py`**:  
     Replaced BatchNorm with Instance Normalization to strip speaker-specific style across convolutional feature maps.
  3. **Advanced Regularization**:
     * **Dropout (0.4)** before the linear classification head.
     * **L2 Weight Decay ($10^{-4}$)** added to the Adam optimizer.
     * **Learning Rate Scheduler**: `CosineAnnealingLR` for smooth convergence.

---

## 📂 Project Structure

```text
.
├── data/
│   └── recordings/      # Raw FSDD WAV files (e.g., 7_jackson_12.wav)
├── models_data/         # Saved model checkpoints (.pth) and loss/accuracy plots
├── dataset.py           # PyTorch Dataset pipeline & data augmentations
├── model.py             # SpokenDigitCNN architecture (InstanceNorm2d, Dropout)
├── train.py             # Training loop, evaluation, and plotting logic
└── README.md            # Project documentation