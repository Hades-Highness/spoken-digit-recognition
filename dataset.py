"""FSDD + AudioMNIST Dataset Pipeline — v3.1 (Fast CPU Loader)."""

import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/recordings"
SAMPLE_RATE = 8000
TARGET_LENGTH = 8000
BATCH_SIZE = 32

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
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label, _ = _parse_meta(file_path)
        waveform = _load_waveform(file_path)
        return waveform, label


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

train_dataset = FSDDDataset(train_files)
val_dataset   = FSDDDataset(val_files)
test_dataset  = FSDDDataset(test_files)

# pin_memory=True et num_workers permettent d'envoyer les données au GPU sans attente
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)