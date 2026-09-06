"""FSDD (Free Spoken Digit Dataset) loading pipeline — v2.0 (Robust).

Exposes FSDDDataset plus module-level train_loader / val_loader / test_loader.
Implements:
 1. Speaker-Independent Split (Train: 4 speakers, Val: 1, Test: 1)
 2. Dynamic White Noise Augmentation on Train set (30% probability)
"""

import os
import glob

import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader

# --- Audio / feature extraction constants -----------------------------------

DATA_DIR = "data/recordings"

SAMPLE_RATE = 8000          # FSDD recordings are 8kHz mono
TARGET_LENGTH = 8000        # Pad/truncate every clip to exactly 1s (8000 samples)

N_MELS = 64                 # Required by model.py's conv stack (input height)
N_FFT = 512                 # ~64ms window at 8kHz
HOP_LENGTH = 256            # 32 time frames, matching model tensor width [B, 1, 64, 32]

BATCH_SIZE = 32

# --- Speaker-Independent Split Definition (v2.0) ---------------------------
# Prevents voice memorization by holding out entire speakers for evaluation.
TRAIN_SPEAKERS = ['jackson', 'nicolas', 'theo', 'yweweler']  # 2,000 clips (~66.7%)
VAL_SPEAKERS   = ['lucas']                                   # 500 clips (~16.7%)
TEST_SPEAKERS  = ['george']                                  # 500 clips (~16.7%)


def _validate_data_dir(root):
    if not os.path.isdir(root) or not glob.glob(os.path.join(root, "*.wav")):
        raise FileNotFoundError(f"No FSDD recordings found in '{root}/'.")
    return sorted(glob.glob(os.path.join(root, "*.wav")))


def _parse_meta(file_path):
    # e.g. "7_jackson_12.wav" -> label 7, speaker "jackson"
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    label = int(parts[0])
    speaker = parts[1]
    return label, speaker


def _load_waveform(file_path):
    waveform, sr = librosa.load(file_path, sr=None)
    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    waveform = torch.from_numpy(waveform).float()

    if waveform.shape[0] < TARGET_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[0]))
    else:
        waveform = waveform[:TARGET_LENGTH]

    return waveform


# --- Dataset -------------------------------------------------------------------


class FSDDDataset(Dataset):
    def __init__(self, file_paths, is_train=False):
        self.file_paths = file_paths
        self.is_train = is_train
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label, _ = _parse_meta(file_path)

        waveform = _load_waveform(file_path)

        # --- Dynamic Data Augmentation: White Noise ---
        if self.is_train and torch.rand(1).item() < 0.30:
            # Controlled noise amplitude between 0.005 and 0.015 (SNR ~15-25 dB)
            noise_level = torch.empty(1).uniform_(0.005, 0.015).item()
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        mel_spec = self.mel_transform(waveform.unsqueeze(0))  # [1, N_MELS, time]
        mel_spec = torch.log(mel_spec + 1e-9)  # log-compression

        return mel_spec, label


# --- Module-level splits / DataLoaders ------------------------------------------

_all_files = _validate_data_dir(DATA_DIR)

train_files, val_files, test_files = [], [], []

for _f in _all_files:
    _, _speaker = _parse_meta(_f)
    if _speaker in TRAIN_SPEAKERS:
        train_files.append(_f)
    elif _speaker in VAL_SPEAKERS:
        val_files.append(_f)
    elif _speaker in TEST_SPEAKERS:
        test_files.append(_f)

train_dataset = FSDDDataset(train_files, is_train=True)
val_dataset   = FSDDDataset(val_files, is_train=False)
test_dataset  = FSDDDataset(test_files, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    print("--- FSDD Dataset v2.0 Summary ---")
    print(f"Train set ({TRAIN_SPEAKERS}): {len(train_dataset)} samples")
    print(f"Val set   ({VAL_SPEAKERS}): {len(val_dataset)} samples")
    print(f"Test set  ({TEST_SPEAKERS}): {len(test_dataset)} samples")

    batch, labels = next(iter(train_loader))
    print("\nBatch tensor shape:", tuple(batch.shape))
    print(f"Label range in batch: min={labels.min().item()}, max={labels.max().item()}")