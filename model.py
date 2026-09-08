import torch
import torch.nn as nn


class SpokenDigitCNN(nn.Module):
    """CNN for speaker-independent spoken-digit classification.

    Expects 3-channel log-mel features (Log-Mel + Delta + Delta-Delta).
    InstanceNorm2d and Dropout reduce sensitivity to speaker identity, and
    AdaptiveAvgPool2d lets the classifier accept any input resolution.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SpokenDigitCNN()
    # Smoke test: [batch, 3 channels, 64 mel bins, 63 frames] @ 16 kHz.
    dummy_input = torch.randn(8, 3, 64, 63)
    output = model(dummy_input)
    print("Architecture check passed.")
    print("Output shape (Batch size, Classes):", output.shape)
