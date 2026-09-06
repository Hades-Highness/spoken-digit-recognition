import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import SpokenDigitCNN

from dataset import train_loader, val_loader

VERSION_NAME = "v2.1"  # Dynamic tag (e.g., v1_baseline, v2_0_robust, v2_1_aug)
EPOCHS = 30
LR = 0.001


def train_one_epoch(model, dataloader, criterion, optimizer, device):
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

    return running_loss / total, (correct / total) * 100


def validate(model, dataloader, criterion, device):
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

    return running_loss / total, (correct / total) * 100


def save_training_history(version_name, train_accs, val_accs, train_losses, val_losses):
    # 1. Save raw metrics to JSON
    history = {
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    with open(f"history_{version_name}.json", "w") as f:
        json.dump(history, f, indent=4)

    # 2. Generate training curves plot
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label="Train Acc", color='blue')
    plt.plot(epochs, val_accs, label="Val Acc", color='orange')
    plt.title(f"Accuracy - {version_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs, val_losses, label="Val Loss", color='orange')
    plt.title(f"Loss - {version_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"curve_{version_name}.png", dpi=300)
    print(f"\n[+] Results saved: curve_{version_name}.png and history_{version_name}.json")


def run_training(train_loader, val_loader, version_name="v1_baseline", epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training version [{version_name}] on: {device} ---")

    model = SpokenDigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_acc = 0.0

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"best_model_{version_name}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"   -> Model saved to: {model_filename} ({best_val_acc:.2f}%)")

    print("\nTraining completed successfully!")
    save_training_history(version_name, train_accs, val_accs, train_losses, val_losses)


if __name__ == "__main__":
    run_training(train_loader, val_loader, version_name=VERSION_NAME, epochs=EPOCHS, lr=LR)