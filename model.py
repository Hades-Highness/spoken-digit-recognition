"""Raw-audio DigitSense v5.0 model and the legacy v4.0 architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64


def _hz_to_mel(frequency):
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class RawAudioFeatureExtractor(nn.Module):
    """ONNX-friendly Log-Mel, delta, and delta-delta feature extraction."""

    def __init__(self):
        super().__init__()
        window = torch.hann_window(N_FFT, periodic=True)
        time = torch.arange(N_FFT, dtype=torch.float32)
        frequencies = torch.arange(N_FFT // 2 + 1, dtype=torch.float32)
        phase = (2.0 * math.pi / N_FFT) * frequencies[:, None] * time[None, :]
        fourier = torch.cat((torch.cos(phase), -torch.sin(phase)), dim=0)
        self.register_buffer("window", window)
        self.register_buffer("fourier_kernel", (fourier * window).unsqueeze(1))
        self.register_buffer("mel_filter", self._make_mel_filter())
        self.register_buffer(
            "delta_kernel",
            torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
            .div(10.0)
            .reshape(1, 1, 5),
        )

    @staticmethod
    def _make_mel_filter():
        mel_min = _hz_to_mel(0.0)
        mel_max = _hz_to_mel(SAMPLE_RATE / 2)
        mel_points = torch.linspace(mel_min, mel_max, N_MELS + 2)
        hz_points = torch.tensor([_mel_to_hz(float(mel)) for mel in mel_points])
        fft_frequencies = torch.linspace(0.0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
        lower = hz_points[:-2, None]
        center = hz_points[1:-1, None]
        upper = hz_points[2:, None]
        rising = (fft_frequencies[None, :] - lower) / (center - lower)
        falling = (upper - fft_frequencies[None, :]) / (upper - center)
        return torch.minimum(rising, falling).clamp_min(0.0)

    def _delta(self, features):
        batch, channels, mel_bins, frames = features.shape
        flattened = features.reshape(batch * channels * mel_bins, 1, frames)
        padded = F.pad(flattened, (2, 2), mode="replicate")
        delta = F.conv1d(padded, self.delta_kernel)
        return delta.reshape(batch, channels, mel_bins, frames)

    def forward(self, waveform):
        if waveform.ndim != 2:
            raise ValueError(f"Expected audio with shape [batch, {TARGET_LENGTH}].")
        if not torch.jit.is_tracing() and waveform.shape[1] != TARGET_LENGTH:
            raise ValueError(f"Expected audio with shape [batch, {TARGET_LENGTH}].")

        padded = F.pad(
            waveform.unsqueeze(1),
            (N_FFT // 2, N_FFT // 2),
            mode="reflect",
        )
        spectrum = F.conv1d(padded, self.fourier_kernel, stride=HOP_LENGTH)
        real = spectrum[:, : N_FFT // 2 + 1]
        imaginary = spectrum[:, N_FFT // 2 + 1 :]
        power = real.square() + imaginary.square()
        mel = torch.matmul(self.mel_filter, power)
        log_mel = torch.log(mel.clamp_min(1e-10)).unsqueeze(1)
        delta = self._delta(log_mel)
        delta_delta = self._delta(delta)
        features = torch.cat((log_mel, delta, delta_delta), dim=1)

        mean = features.mean(dim=(-2, -1), keepdim=True)
        variance = features.var(dim=(-2, -1), unbiased=False, keepdim=True)
        return (features - mean) / torch.sqrt(variance + 1e-6)


class FixedAdaptiveAvgPool2d(nn.Module):
    """Reproduce AdaptiveAvgPool2d((4, 4)) for the fixed 8-by-7 feature map."""

    def __init__(self):
        super().__init__()
        self.height_bounds = self._bounds(8, 4)
        self.width_bounds = self._bounds(7, 4)

    @staticmethod
    def _bounds(input_size, output_size):
        return [
            (
                math.floor(index * input_size / output_size),
                math.ceil((index + 1) * input_size / output_size),
            )
            for index in range(output_size)
        ]

    def forward(self, features):
        rows = []
        for height_start, height_end in self.height_bounds:
            columns = [
                features[:, :, height_start:height_end, width_start:width_end].mean(
                    dim=(2, 3), keepdim=True
                )
                for width_start, width_end in self.width_bounds
            ]
            rows.append(torch.cat(columns, dim=3))
        return torch.cat(rows, dim=2)


class SpokenDigitCNN(nn.Module):
    """Classify a one-second, mono, 16 kHz waveform directly."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = RawAudioFeatureExtractor()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            FixedAdaptiveAvgPool2d(),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveform):
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            features = self.feature_extractor(waveform.float())
        return self.classifier(self.conv3(self.conv2(self.conv1(features))))


class LegacySpokenDigitCNN(nn.Module):
    """Feature-input CNN retained for existing v4.0 checkpoints and the demo."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, features):
        return self.classifier(self.conv3(self.conv2(self.conv1(features))))


if __name__ == "__main__":
    model = SpokenDigitCNN().eval()
    with torch.no_grad():
        output = model(torch.randn(1, TARGET_LENGTH))
    print("DigitSense v5.0 architecture check passed.")
    print(f"Output shape: {tuple(output.shape)}")
