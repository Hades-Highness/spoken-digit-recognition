import torch
from model import SpokenDigitCNN
from dataset import test_loader
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluation sur : {device} ---")
    
    # Chargement du modèle et des poids
    model = SpokenDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("\n--- RAPPORT DE CLASSIFICATION (TEST SET) ---")
    print(classification_report(all_targets, all_preds, digits=4))
    
    print("--- MATRICE DE CONFUSION ---")
    print(confusion_matrix(all_targets, all_preds))

if __name__ == "__main__":
    evaluate()