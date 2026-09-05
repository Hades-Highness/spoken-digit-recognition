"""FSDD (Free Spoken Digit Dataset) loading pipeline.

Exposes FSDDDataset plus module-level train_loader / val_loader / test_loader,
as expected by train.py (`from dataset import FSDDDataset`) and model.py
(spectrograms shaped [1, N_MELS, time], with N_MELS=64 matching the
torch.randn(8, 1, 64, 32) test tensor in model.py).
"""

import os
import glob

import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader, random_split

# --- Audio / feature extraction constants -----------------------------------

DATA_DIR = "data/recordings"

SAMPLE_RATE = 8000          # FSDD recordings are 8kHz mono
TARGET_LENGTH = 8000        # pad/truncate every clip to exactly 1s (8000 samples)

N_MELS = 64                 # required by model.py's conv stack (input height)
N_FFT = 512                 # ~64ms window at 8kHz: enough resolution for short digit utterances
HOP_LENGTH = 256            # 1 + TARGET_LENGTH // HOP_LENGTH = 32 time frames, matching
                             # model.py's test tensor width (torch.randn(8, 1, 64, 32))

BATCH_SIZE = 32
RANDOM_SEED = 42            # fixed seed for a reproducible train/val/test split

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
# remaining ~0.1 goes to test

# NOTE ON THE RANDOM SPLIT (please read before trusting eval numbers):
# FSDD only has 6 speakers. A plain random 80/10/10 split (as implemented
# below, per the task spec) puts recordings from the same speaker in both
# train and test. The model can partly learn to recognize *voices* rather
# than *digits*, which inflates train/val/test accuracy compared to how the
# model would perform on a genuinely unseen speaker. A speaker-held-out split
# (e.g. train on 4 speakers, test on the remaining ones) would give a more
# honest estimate. Flagging this for whoever reads the reported accuracy.

# --- Audio loading ------------------------------------------------------------
#
# Feature extraction (the Mel-spectrogram) uses torchaudio.transforms, which
# is a pure tensor op and keeps this pipeline PyTorch-native as requested.
# For decoding the .wav files themselves, this uses librosa.load instead of
# torchaudio.load: recent torchaudio versions dropped their built-in
# sox/soundfile decoding backends and now require a separate torchcodec +
# system FFmpeg install for torchaudio.load to work at all, which isn't a
# reasonable thing to assume is present on every machine. librosa is already
# a pinned dependency in requirements.txt and decodes wav files without any
# extra system install, so it's used here purely for file I/O.


def _validate_data_dir(root):
    if not os.path.isdir(root) or not glob.glob(os.path.join(root, "*.wav")):
        raise FileNotFoundError(
            f"No FSDD recordings found in '{root}/'.\n"
            "Download the Free Spoken Digit Dataset before running this script:\n"
            "  git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git fsdd_tmp\n"
            f"  mv fsdd_tmp/recordings/*.wav {root}/\n"
            "  rm -rf fsdd_tmp\n"
        )
    return sorted(glob.glob(os.path.join(root, "*.wav")))


def _parse_label(file_path):
    # e.g. "7_jackson_12.wav" -> digit label 7
    filename = os.path.basename(file_path)
    return int(filename.split("_")[0])


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
    def __init__(self, file_paths):
        self.file_paths = file_paths
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
        label = _parse_label(file_path)

        waveform = _load_waveform(file_path)
        mel_spec = self.mel_transform(waveform.unsqueeze(0))  # [1, N_MELS, time]
        mel_spec = torch.log(mel_spec + 1e-9)  # log-compress for a saner dynamic range

        return mel_spec, label


# --- Module-level splits / DataLoaders ------------------------------------------

_all_files = _validate_data_dir(DATA_DIR)
_full_dataset = FSDDDataset(_all_files)

_n_total = len(_full_dataset)
_n_train = int(_n_total * TRAIN_FRACTION)
_n_val = int(_n_total * VAL_FRACTION)
_n_test = _n_total - _n_train - _n_val

_generator = torch.Generator().manual_seed(RANDOM_SEED)
train_dataset, val_dataset, test_dataset = random_split(
    _full_dataset, [_n_train, _n_val, _n_test], generator=_generator
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    print(f"Total clips: {_n_total} (train={_n_train}, val={_n_val}, test={_n_test})")

    batch, labels = next(iter(train_loader))
    print("Batch shape:", tuple(batch.shape))
    print(f"Label range in batch: min={labels.min().item()}, max={labels.max().item()}")
