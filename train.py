import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SpokenDigitCNN
from dataset import train_loader, val_loader

# Note : Walid when you finish dataset.py, uncomment the line below to import FSDDDataset
# from dataset import FSDDDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle sur une époque."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Évalue le modèle sur le jeu de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = (correct / total) * 100
    return val_loss, val_acc


def run_training(train_loader, val_loader, epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device : {device} ---")

    model = SpokenDigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Époch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"   -> Best model saved with Accuracy: {best_val_acc:.2f}%")

    print("\n Training completed successfully")


if __name__ == "__main__":
    run_training(train_loader, val_loader, epochs=30, lr=0.001)