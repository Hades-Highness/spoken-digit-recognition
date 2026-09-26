"""Confusion matrix and classification report for a DigitSense checkpoint.

Reproduces the v4.0 feature recipe (16 kHz, 3-channel log-mel + deltas), which
is the only recipe the current model.py can consume: it expects 3 input
channels. Checkpoints v1.0-v3.1 were trained on 1-channel log-mel and cannot be
evaluated with this script without restoring their model definition.

Usage:
    python evaluate.py                        # v4.0 on the test split
    python evaluate.py --version v3.2
    python evaluate.py --version v4.0 --split val --out-dir models_data/model_v4
"""

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio.functional as FA
import torchaudio.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from dataset import SAMPLE_RATE, test_loader, val_loader
from model import SpokenDigitCNN


def evaluate_version(model_path, version_label, loader=None, out_dir="."):
    """Evaluate a checkpoint on a held-out split and save metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating model [{version_label}] on: {device} ---")

    loader = test_loader if loader is None else loader
    os.makedirs(out_dir, exist_ok=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=64
    ).to(device)

    model = SpokenDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for waveforms, targets in loader:
            waveforms, targets = waveforms.to(device), targets.to(device)

            # Reproduce the 3-channel features used during training.
            mel_specs = mel_transform(waveforms)
            log_mel = torch.log(mel_specs + 1e-9)

            delta = FA.compute_deltas(log_mel)
            delta_delta = FA.compute_deltas(delta)

            inputs = torch.stack([log_mel, delta, delta_delta], dim=1)

            # Per-instance standardization, matching train.py.
            mean = inputs.mean(dim=(-2, -1), keepdim=True)
            std = inputs.std(dim=(-2, -1), keepdim=True)
            inputs = (inputs - mean) / (std + 1e-6)

            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.title(f"Confusion Matrix - {version_label} (Test Set)")
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.tight_layout()

    plot_filename = os.path.join(out_dir, f"cm_{version_label}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    report = classification_report(all_targets, all_preds, digits=4)
    report_filename = os.path.join(out_dir, f"report_{version_label}.txt")
    with open(report_filename, "w") as f:
        f.write(f"=== CLASSIFICATION REPORT: {version_label} ===\n\n")
        f.write(report)

    accuracy = (sum(p == t for p, t in zip(all_preds, all_targets))
                / len(all_targets) * 100)
    print(f"[+] Evaluation complete for {version_label}!")
    print(f"   -> Accuracy: {accuracy:.2f}% on {len(all_targets)} clips")
    print(f"   -> Saved confusion matrix plot: {plot_filename}")
    print(f"   -> Saved text report: {report_filename}\n")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v4.0",
                        help="Version tag used for file names (default: v4.0)")
    parser.add_argument("--model", default=None,
                        help="Checkpoint path (default: models/digitsense_<version>.pth)")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which held-out split to evaluate (default: test)")
    parser.add_argument("--out-dir", default=".",
                        help="Where to write cm_*.png and report_*.txt (default: .)")
    args = parser.parse_args()

    version_tag = args.version
    candidates = [
        args.model,
        f"models/digitsense_{version_tag}.pth",
        f"best_model_{version_tag}.pth",
    ]
    model_file = next(
        (p for p in candidates if p and os.path.exists(p)),
        candidates[1],
    )
    loader = val_loader if args.split == "val" else test_loader

    evaluate_version(model_file, version_tag, loader=loader, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
