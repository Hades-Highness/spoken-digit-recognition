"""AudioMNIST Dataset Pipeline — v4.0 (16 kHz Speaker-Independent)."""

import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIGURATION GLOBALE v4.0
# ============================================================
DATA_DIR = "data"
SAMPLE_RATE = 16000
TARGET_LENGTH = 16000  # 1.0 seconde à 16 kHz
BATCH_SIZE = 64        # Augmenté pour accélérer le training sur 30k samples

# Split Speaker-Independent (50 Train / 5 Val / 5 Test)
TEST_SPEAKERS = {'01', '02', '07', '28', '47'}  # 5 locuteurs isolés (~2 500 audios)
VAL_SPEAKERS = {'03', '04', '05', '12', '26'}  # 5 locuteurs isolés (~2 500 audios)


def _parse_meta(file_path):
    """Extrait le chiffre et le speaker_id depuis '0_01_12.wav'."""
    filename = os.path.basename(file_path)
    parts = filename.replace(".wav", "").split("_")
    return int(parts[0]), parts[1]


def _load_waveform(file_path, resampler):
    """Charge le fichier audio, rééchantillonne à 16kHz et applique le Pad/Crop."""
    waveform, sr = torchaudio.load(file_path)

    # Conversion Mono si nécessaire
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resampling de 48 kHz -> 16 kHz
    if sr != SAMPLE_RATE:
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    # Padding / Cropping strict à 16 000 échantillons (1s)
    if waveform.shape[0] < TARGET_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[0]))
    else:
        waveform = waveform[:TARGET_LENGTH]

    return waveform


class AudioMNISTDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        # Instanciation du Resampler (48000 Hz -> 16000 Hz)
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=SAMPLE_RATE)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label, _ = _parse_meta(file_path)
        waveform = _load_waveform(file_path, self.resampler)
        return waveform, label


# ============================================================
# PARCOURS DE L'ARBORESCENCE (data/*/*.wav)
# ============================================================
search_pattern = os.path.join(DATA_DIR, "*", "*.wav")
all_files = glob.glob(search_pattern)

if not all_files:
    raise FileNotFoundError(f"Aucun fichier WAV trouvé dans '{DATA_DIR}/*/*.wav'. Vérifie le dossier.")

train_files, val_files, test_files = [], [], []

for f in all_files:
    _, speaker = _parse_meta(f)
    if speaker in TEST_SPEAKERS:
        test_files.append(f)
    elif speaker in VAL_SPEAKERS:
        val_files.append(f)
    else:
        train_files.append(f)

# Instanciation des Datasets
train_dataset = AudioMNISTDataset(train_files)
val_dataset   = AudioMNISTDataset(val_files)
test_dataset  = AudioMNISTDataset(test_files)

# Multi-processing DataLoader pour un chargement CPU -> GPU ultra rapide
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

if __name__ == "__main__":
    print(f"=== DATASET v4.0 INITIALISÉ ({SAMPLE_RATE} Hz) ===")
    print(f"Train Set : {len(train_files)} samples (50 locuteurs)")
    print(f"Val Set   : {len(val_files)} samples (5 locuteurs)")
    print(f"Test Set  : {len(test_files)} samples (5 locuteurs)")