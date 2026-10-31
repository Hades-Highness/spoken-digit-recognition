"""In-memory AudioMNIST data pipeline for DigitSense v5.0."""

import glob
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
BATCH_SIZE = 64
SEED = 42
AUDIOMNIST_URL = "https://zenodo.org/records/10519652/files/AudioMNIST.zip"

TEST_SPEAKERS = {"01", "02", "07", "28", "47"}
VAL_SPEAKERS = {"03", "04", "05", "12", "26"}
_RESAMPLERS = {}


def _download_dataset():
    """Download and extract AudioMNIST with aria2c when no WAV files exist."""
    aria2c = shutil.which("aria2c")
    if aria2c is None:
        raise FileNotFoundError(
            "AudioMNIST WAV files are missing and aria2c is not installed. "
            "Install aria2c or place the AudioMNIST speaker folders under data/."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / ".AudioMNIST.zip"
    command = [
        aria2c,
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=1M",
        "--dir",
        str(DATA_DIR),
        "--out",
        archive_path.name,
        AUDIOMNIST_URL,
    ]
    logger.info("Downloading AudioMNIST with aria2c.")
    try:
        subprocess.run(command, check=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(DATA_DIR)
        archive_path.unlink()
    except (OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Unable to download or extract AudioMNIST: {exc}") from exc


def _parse_meta(file_path):
    """Return the digit label and speaker id from an AudioMNIST filename."""
    filename = os.path.basename(file_path)
    parts = Path(filename).stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Unexpected filename '{filename}'; expected "
            "'<digit>_<speaker>_<index>.wav'."
        )
    label = int(parts[0])
    if not 0 <= label <= 9:
        raise ValueError(f"Digit label in '{filename}' is outside the range 0-9.")
    return label, parts[1]


def _get_resampler(orig_freq):
    if orig_freq not in _RESAMPLERS:
        _RESAMPLERS[orig_freq] = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=SAMPLE_RATE
        )
    return _RESAMPLERS[orig_freq]


def _normalize_waveform(waveform):
    """RMS-normalize a waveform while bounding gain and avoiding clipping."""
    rms = waveform.square().mean().sqrt()
    gain = (0.1 / rms.clamp_min(1e-6)).clamp(0.1, 10.0)
    waveform = waveform * gain
    return waveform.clamp(-0.99, 0.99)


def _load_waveform(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.ndim != 2 or waveform.shape[0] == 0:
        raise ValueError(f"Audio file '{file_path}' has an invalid waveform shape.")

    waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    if sample_rate != SAMPLE_RATE:
        waveform = _get_resampler(sample_rate)(waveform)
    waveform = waveform.squeeze(0)

    if waveform.numel() < TARGET_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.numel()))
    else:
        waveform = waveform[:TARGET_LENGTH]
    return _normalize_waveform(waveform.contiguous())


class AudioMNISTDataset(Dataset):
    """A split fully decoded and cached in RAM as 16 kHz mono tensors."""

    def __init__(self, file_paths, interference_pool=None):
        self.file_paths = list(file_paths)
        if not self.file_paths:
            raise ValueError("A dataset split cannot be empty.")

        self.waveforms = torch.empty(
            len(self.file_paths), TARGET_LENGTH, dtype=torch.float32
        )
        for index, file_path in enumerate(self.file_paths):
            self.waveforms[index] = _load_waveform(file_path)
        self.labels = torch.tensor(
            [_parse_meta(path)[0] for path in self.file_paths], dtype=torch.long
        )
        self.interference_pool = interference_pool

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        waveform = self.waveforms[index]
        label = self.labels[index]
        if self.interference_pool is None:
            return waveform, label

        if self.interference_pool is self.waveforms and len(self.interference_pool) > 1:
            pool_index = torch.randint(len(self.interference_pool) - 1, ()).item()
            pool_index += pool_index >= index
        else:
            pool_index = torch.randint(len(self.interference_pool), ()).item()
        return waveform, label, self.interference_pool[pool_index]


def _find_audio_files():
    files = sorted(glob.glob(str(DATA_DIR / "**" / "*.wav"), recursive=True))
    if not files:
        _download_dataset()
        files = sorted(glob.glob(str(DATA_DIR / "**" / "*.wav"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No AudioMNIST WAV files found below '{DATA_DIR}'.")
    return files


def create_datasets():
    train_files, val_files, test_files = [], [], []
    for file_path in _find_audio_files():
        _, speaker = _parse_meta(file_path)
        if speaker in TEST_SPEAKERS:
            test_files.append(file_path)
        elif speaker in VAL_SPEAKERS:
            val_files.append(file_path)
        else:
            train_files.append(file_path)

    train_dataset = AudioMNISTDataset(train_files)
    train_dataset.interference_pool = train_dataset.waveforms
    val_dataset = AudioMNISTDataset(val_files)
    test_dataset = AudioMNISTDataset(test_files)
    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = create_datasets()

# Zero workers avoid copying the multi-gigabyte in-memory cache on Windows.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

if __name__ == "__main__":
    print(f"DigitSense v5.0 dataset initialized at {SAMPLE_RATE} Hz.")
    print(f"Train split: {len(train_dataset)} samples")
    print(f"Validation split: {len(val_dataset)} samples")
    print(f"Test split: {len(test_dataset)} samples")
