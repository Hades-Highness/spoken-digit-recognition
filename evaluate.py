import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio.functional as FA
import torchaudio.transforms as T
from sklearn.metrics import classification_report, confusion_matrix

from dataset import test_loader
from model import SpokenDigitCNN


def evaluate_version(model_path, version_label):
    """Evaluate a checkpoint on the held-out test split and save metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating model [{version_label}] on: {device} ---")

    mel_transform = T.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64
    ).to(device)

    model = SpokenDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for waveforms, targets in test_loader:
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

    plot_filename = f"cm_{version_label}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    report = classification_report(all_targets, all_preds, digits=4)
    report_filename = f"report_{version_label}.txt"
    with open(report_filename, "w") as f:
        f.write(f"=== CLASSIFICATION REPORT: {version_label} ===\n\n")
        f.write(report)

    print(f"[+] Evaluation complete for {version_label}!")
    print(f"   -> Saved confusion matrix plot: {plot_filename}")
    print(f"   -> Saved text report: {report_filename}\n")
    print(report)


if __name__ == "__main__":
    version_tag = "v4.0"

    # Shipped checkpoints live in models/ (same convention as inference.py);
    # keep the root-level training filename as a fallback for local runs.
    candidates = [
        f"models/digitsense_{version_tag}.pth",
        f"best_model_{version_tag}.pth",
    ]
    model_file = next((p for p in candidates if os.path.exists(p)), candidates[0])

    evaluate_version(model_file, version_tag)
