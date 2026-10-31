"""GPU-accelerated raw-audio training for DigitSense v5.0."""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import SpokenDigitCNN

SEED = 42
VERSION_NAME = "v5.0"
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
CHECKPOINT_DIR = "models"
SPEECH_MIX_PROBABILITY = 0.65
NOISE_MIX_PROBABILITY = 0.75


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mix_training_batch(waveforms, interference):
    """Mix training utterances with other training speech and colored noise."""
    batch_size = waveforms.shape[0]
    device = waveforms.device
    mixed = waveforms.clone()

    speech_mask = torch.rand(batch_size, device=device) < SPEECH_MIX_PROBABILITY
    speech_sir = torch.empty(batch_size, 1, device=device).uniform_(0.25, 0.8)
    speech = interference * speech_sir
    mixed = mixed + speech * speech_mask[:, None]

    noise_mask = torch.rand(batch_size, 1, device=device) < NOISE_MIX_PROBABILITY
    white_noise = torch.randn_like(mixed)
    colored_noise = torch.nn.functional.avg_pool1d(
        white_noise.unsqueeze(1), kernel_size=9, stride=1, padding=4
    ).squeeze(1)
    colored_noise = colored_noise / colored_noise.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise = white_noise * 0.5 + colored_noise * 0.5
    noise = noise / noise.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise_snr = torch.empty(batch_size, 1, device=device).uniform_(0.01, 0.12)
    mixed = mixed + noise * noise_snr * noise_mask
    return mixed.clamp(-1.0, 1.0)


def _autocast(device):
    return torch.autocast(device_type=device.type, enabled=device.type == "cuda")


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0

    for waveforms, targets, interference in tqdm(dataloader, leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        interference = interference.to(device, non_blocking=True)
        inputs = mix_training_batch(waveforms, interference)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach() * targets.size(0)
        correct += outputs.detach().argmax(dim=1).eq(targets).sum()
        total += targets.size(0)

    return (running_loss / total).item(), (correct * 100.0 / total).item()


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0

    with torch.inference_mode():
        for waveforms, targets in dataloader:
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device):
                outputs = model(waveforms)
                loss = criterion(outputs, targets)
            running_loss += loss * targets.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum()
            total += targets.size(0)

    return (running_loss / total).item(), (correct * 100.0 / total).item()


def save_training_history(version_name, train_accs, val_accs, train_losses, val_losses, config):
    history = {
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "config": config,
    }
    with open(f"history_{version_name}.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label="Train accuracy")
    plt.plot(epochs, val_accs, label="Validation accuracy")
    plt.title(f"Accuracy - {version_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.title(f"Loss - {version_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curve_path = f"curve_{version_name}.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Training history saved to history_{version_name}.json and {curve_path}.")


def run_training(
    train_loader,
    val_loader,
    test_loader,
    version_name=VERSION_NAME,
    epochs=EPOCHS,
    lr=LR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DigitSense {version_name} on {device}.")
    set_seed()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = SpokenDigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_filename = os.path.join(CHECKPOINT_DIR, f"digitsense_{version_name}.pth")
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(
            f"Epoch {epoch}/{epochs} | LR {scheduler.get_last_lr()[0]:.6g} | "
            f"Train loss {train_loss:.4f}, accuracy {train_acc:.2f}% | "
            f"Validation loss {val_loss:.4f}, accuracy {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_filename)
            print(f"Saved best checkpoint to {model_filename}.")

    save_training_history(
        version_name,
        train_accs,
        val_accs,
        train_losses,
        val_losses,
        {
            "version": version_name,
            "seed": SEED,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "batch_size": getattr(train_loader, "batch_size", None),
            "best_validation_accuracy": round(best_val_acc, 2),
            "mixed_speech_probability": SPEECH_MIX_PROBABILITY,
            "background_noise_probability": NOISE_MIX_PROBABILITY,
        },
    )

    model.load_state_dict(torch.load(model_filename, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.2f}% (loss {test_loss:.4f}).")


if __name__ == "__main__":
    from dataset import test_loader, train_loader, val_loader

    run_training(train_loader, val_loader, test_loader)
