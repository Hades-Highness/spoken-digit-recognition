"""FSDD + AudioMNIST Dataset Pipeline — v3.0 (Dataset Expansion & Gender Balancing)."""

import os
import glob
import torch
import torchaudio.transforms as T
import librosa
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/recordings"
SAMPLE_RATE = 8000
TARGET_LENGTH = 8000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
BATCH_SIZE = 32

# 8 locuteurs d'entraînement (4 hommes FSDD + 4 femmes AudioMNIST)
TRAIN_SPEAKERS = ['jackson', 'nicolas', 'theo', 'yweweler', '12', '26', '28', '47']
VAL_SPEAKERS   = ['lucas']
TEST_SPEAKERS  = ['george']


def _validate_data_dir(root):
    if not os.path.isdir(root) or not glob.glob(os.path.join(root, "*.wav")):
        raise FileNotFoundError(f"No WAV files found in '{root}/'")
    return sorted(glob.glob(os.path.join(root, "*.wav")))


def _parse_meta(file_path):
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    return int(parts[0]), parts[1]


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


class FSDDDataset(Dataset):
    def __init__(self, file_paths, is_train=False):
        self.file_paths = file_paths
        self.is_train = is_train

        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

        # Light SpecAugment (narrow masks)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=4)
        self.time_mask = T.TimeMasking(time_mask_param=2)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label, _ = _parse_meta(file_path)

        waveform = _load_waveform(file_path)

        # 1. Subtle White Noise Injection (20% chance)
        if self.is_train and torch.rand(1).item() < 0.20:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        # Extract Mel-Spectrogram
        mel_spec = self.mel_transform(waveform.unsqueeze(0))
        mel_spec = torch.log(mel_spec + 1e-9)

        # --- Instance Standardization ---
        mean = mel_spec.mean()
        std = mel_spec.std()
        mel_spec = (mel_spec - mean) / (std + 1e-6)

        # 2. Subtle SpecAugment (30% chance)
        if self.is_train and torch.rand(1).item() < 0.30:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        return mel_spec, label


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
    print(f"[v3.0 Data Split Summary]")
    print(f" Train Samples : {len(train_dataset)} (8 locuteurs)")
    print(f" Val Samples   : {len(val_dataset)} (1 locuteur: lucas)")
    print(f" Test Samples  : {len(test_dataset)} (1 locuteur: george)")
    print(f" Total Audios  : {len(train_dataset) + len(val_dataset) + len(test_dataset)}")