import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import SpokenDigitCNN

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset and plot the confusion matrix."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 1. Print metrics (Precision, Recall, F1-Score)
    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=[str(i) for i in range(10)], digits=4))

    # 2. Generate and save confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10)
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.title('Confusion Matrix - Spoken Digit Recognition')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    print("Script evaluate.py is ready to receive the model and TestLoader.")