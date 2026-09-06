import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import SpokenDigitCNN
from dataset import test_loader  # Loads the held-out test set


def evaluate_version(model_path, version_label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating model [{version_label}] on: {device} ---")

    # 1. Load Model Weights
    model = SpokenDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 2. Generate and Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {version_label}")
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.tight_layout()

    plot_filename = f"cm_{version_label}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    # 3. Save Text Classification Report
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
    # Command line usage: python evaluate_single.py <model_path> <version_label>
    model_file = "best_model_v2.1.pth"
    version_tag = "v2.1"

    evaluate_version(model_file, version_tag)