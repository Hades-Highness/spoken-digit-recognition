import torch
import torch.nn as nn

class SpokenDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpokenDigitCNN, self).__init__()

        # Bloc Convolutif 1 : Détecte les motifs acoustiques de base
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc Convolutif 2 : Combine les motifs pour capturer des phonèmes
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bloc Convolutif 3 : Extraction de caractéristiques complexes
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Tête de classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixe la taille de sortie peu importe la durée de l'audio
            nn.Flatten(),
            nn.Dropout(p=0.3),            # Évite le surapprentissage sur les voix
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)   # Sortie de 10 logits (chiffres 0 à 9)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# Test rapide du modèle (s'exécute uniquement si tu lances model.py directement)
if __name__ == "__main__":
    model = SpokenDigitCNN()
    # Simulation d'un batch de 8 spectrogrammes
    dummy_input = torch.randn(8, 1, 64, 32)
    output = model(dummy_input)
    print(" Architecture créée avec succès !")
    print("Forme de la sortie (Batch size, Classes) :", output.shape)