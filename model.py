import torch
import torch.nn as nn

class SpokenDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpokenDigitCNN, self).__init__()

        # Bloc Convolutif 1 : Extraction de motifs + Instance Normalization (Invariant au locuteur)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc Convolutif 2 : Capture des phonèmes
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc Convolutif 3 : Motifs acoustiques complexes
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Tête de classification avec régularisation renforcée
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
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = SpokenDigitCNN()
    dummy_input = torch.randn(8, 1, 64, 32)
    output = model(dummy_input)
    print("Architecture v2.2 créée avec succès !")
    print("Forme de la sortie (Batch size, Classes) :", output.shape)