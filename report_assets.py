import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Imports de tes modules locaux
from model import SpokenDigitCNN
from dataset import test_loader  # test_loader contient le locuteur 'george' (v2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path):
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

    return np.array(all_targets), np.array(all_preds)

# 1. Évaluation V1 vs V2 sur le Test Set (Locuteur 'george')
print("Génération des métriques pour le rapport...")
y_true_v1, y_pred_v1 = evaluate_model("best_model_v1.pth")
y_true_v2, y_pred_v2 = evaluate_model("best_model_v2.pth")

# 2. Génération des Matrices de Confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_v1 = confusion_matrix(y_true_v1, y_pred_v1)
cm_v2 = confusion_matrix(y_true_v2, y_pred_v2)

sns.heatmap(cm_v1, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title("V1 Baseline (Random Split Weights) on Unseen Speaker")
axes[0].set_xlabel("Predicted Digit")
axes[0].set_ylabel("True Digit")

sns.heatmap(cm_v2, annot=True, fmt="d", cmap="Reds", ax=axes[1], cbar=False)
axes[1].set_title("V2 Robust (Speaker-Independent Split) on Unseen Speaker")
axes[1].set_xlabel("Predicted Digit")
axes[1].set_ylabel("True Digit")

plt.tight_layout()
plt.savefig("report_confusion_matrices_v1_vs_v2.png", dpi=300)
print(" Image sauvegardée : report_confusion_matrices_v1_vs_v2.png")

# 3. Graphique Comparatif de l'Accuracy Globale
acc_v1 = (y_pred_v1 == y_true_v1).mean() * 100
acc_v2 = (y_pred_v2 == y_true_v2).mean() * 100

plt.figure(figsize=(7, 5))
bars = plt.bar(["v1.0 Baseline\n(Data Leakage)", "v2.0 Robust\n(Speaker-Independent)"], 
               [acc_v1, acc_v2], color=['#2b5c8f', '#d9534f'])

plt.ylabel("Accuracy on Unseen Speaker (%)")
plt.title("Generalization Gap: V1 vs V2 Evaluation")
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("report_accuracy_comparison.png", dpi=300)
print(" Image sauvegardée : report_accuracy_comparison.png")