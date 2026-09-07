import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import torchaudio.functional as FA

from model import SpokenDigitCNN
from dataset import train_loader, val_loader

VERSION_NAME = "v3.1"  
EPOCHS = 30
LR = 0.001


def get_gpu_transforms(device):
    """Initialise les transformations PyTorch sur le GPU."""
    mel_transform = T.MelSpectrogram(
        sample_rate=8000, n_fft=512, hop_length=256, n_mels=64
    ).to(device)
    
    freq_mask = T.FrequencyMasking(freq_mask_param=8).to(device)
    time_mask = T.TimeMasking(time_mask_param=6).to(device)
    
    return mel_transform, freq_mask, time_mask


def process_batch_gpu(waveforms, mel_transform, freq_mask, time_mask, device, is_train=True):
    """Transforme les ondes sonores [B, 8000] en Spectrogrammes [B, 1, 64, 32] directement sur GPU."""
    waveforms = waveforms.to(device)

    if is_train:
        # 1. Pitch Shift GPU (30% chance)
        if torch.rand(1).item() < 0.30:
            n_steps = torch.randint(-2, 3, (1,)).item()
            if n_steps != 0:
                waveforms = FA.pitch_shift(waveforms, sample_rate=8000, n_steps=n_steps)

        # 2. Time Shift GPU (30% chance)
        if torch.rand(1).item() < 0.30:
            shift = torch.randint(-800, 800, (1,)).item()
            waveforms = torch.roll(waveforms, shifts=shift, dims=1)

        # 3. Injection de bruit blanc GPU (20% chance)
        if torch.rand(1).item() < 0.20:
            waveforms = waveforms + torch.randn_like(waveforms) * 0.005

    # Extraction Mel-Spectrogramme
    mel_specs = mel_transform(waveforms)
    mel_specs = torch.log(mel_specs + 1e-9)

    # Instance Standardization globale par batch
    mean = mel_specs.mean(dim=(-2, -1), keepdim=True)
    std = mel_specs.std(dim=(-2, -1), keepdim=True)
    mel_specs = (mel_specs - mean) / (std + 1e-6)

    # Ajout du canal d'entrée pour la Conv2D -> [B, 1, 64, 32]
    mel_specs = mel_specs.unsqueeze(1)

    # SpecAugment
    if is_train and torch.rand(1).item() < 0.30:
        mel_specs = freq_mask(mel_specs)
        mel_specs = time_mask(mel_specs)

    return mel_specs


def train_one_epoch(model, dataloader, criterion, optimizer, device, transforms):
    model.train()
    mel_transform, freq_mask, time_mask = transforms
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, targets in dataloader:
        targets = targets.to(device)
        
        # Transformation Audio -> Spectrogramme sur GPU
        inputs = process_batch_gpu(waveforms, mel_transform, freq_mask, time_mask, device, is_train=True)

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


def validate(model, dataloader, criterion, device, transforms):
    model.eval()
    mel_transform, freq_mask, time_mask = transforms
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, targets in dataloader:
            targets = targets.to(device)
            
            # Transformation Audio -> Spectrogramme sur GPU (SANS Augmentation)
            inputs = process_batch_gpu(waveforms, mel_transform, freq_mask, time_mask, device, is_train=False)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, (correct / total) * 100


def save_training_history(version_name, train_accs, val_accs, train_losses, val_losses):
    history = {
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    with open(f"history_{version_name}.json", "w") as f:
        json.dump(history, f, indent=4)

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


def run_training(train_loader, val_loader, version_name="v3_1", epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training version [{version_name}] on: {device} ---")

    transforms = get_gpu_transforms(device)
    model = SpokenDigitCNN(num_classes=10).to(device)
    
    # Label Smoothing pour contrer les classes aimants (3 et 5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, transforms)
        val_loss, val_acc = validate(model, val_loader, criterion, device, transforms)

        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{epochs}] (LR: {current_lr:.6f}) | "
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