"""AudioMNIST dataset pipeline - v4.0 (16 kHz, speaker-independent)."""

import glob
import os

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# ============================================================
# GLOBAL CONFIGURATION v4.0
# ============================================================
DATA_DIR = "data"
SAMPLE_RATE = 16000
TARGET_LENGTH = 16000  # 1 second at 16 kHz
BATCH_SIZE = 64

# Speaker-independent split (50 train / 5 val / 5 test speakers).
TEST_SPEAKERS = {"01", "02", "07", "28", "47"}  # 5 held-out test speakers
VAL_SPEAKERS = {"03", "04", "05", "12", "26"}   # 5 held-out validation speakers


def _parse_meta(file_path):
    """Extract the digit and speaker id from a name like '0_01_12.wav'."""
    filename = os.path.basename(file_path)
    parts = filename.replace(".wav", "").split("_")
    return int(parts[0]), parts[1]


def _load_waveform(file_path, resampler):
    """Load a clip, convert to mono, resample to 16 kHz, and pad/truncate to 1 s."""
    waveform, sr = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    if waveform.shape[0] < TARGET_LENGTH:
        waveform = torch.nn.functional.pad(
            waveform, (0, TARGET_LENGTH - waveform.shape[0])
        )
    else:
        waveform = waveform[:TARGET_LENGTH]

    return waveform


class AudioMNISTDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        # Source recordings are 48 kHz; resample to 16 kHz on load.
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=SAMPLE_RATE
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label, _ = _parse_meta(file_path)
        waveform = _load_waveform(file_path, self.resampler)
        return waveform, label


# ============================================================
# GLOB DATA DIRECTORIES AND APPLY THE SPEAKER SPLIT
# ============================================================
search_pattern = os.path.join(DATA_DIR, "*", "*.wav")
all_files = glob.glob(search_pattern)

if not all_files:
    raise FileNotFoundError(
        f"No WAV files found in '{DATA_DIR}/*/*.wav'. Check the data folder."
    )

train_files, val_files, test_files = [], [], []

for f in all_files:
    _, speaker = _parse_meta(f)
    if speaker in TEST_SPEAKERS:
        test_files.append(f)
    elif speaker in VAL_SPEAKERS:
        val_files.append(f)
    else:
        train_files.append(f)

train_dataset = AudioMNISTDataset(train_files)
val_dataset = AudioMNISTDataset(val_files)
test_dataset = AudioMNISTDataset(test_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)

if __name__ == "__main__":
    print(f"=== DATASET v4.0 INITIALIZED ({SAMPLE_RATE} Hz) ===")
    print(f"Train Set : {len(train_files)} samples (50 speakers)")
    print(f"Val Set   : {len(val_files)} samples (5 speakers)")
    print(f"Test Set  : {len(test_files)} samples (5 speakers)")
