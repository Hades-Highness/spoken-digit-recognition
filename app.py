import os
import time

import gradio as gr
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torchaudio.transforms as T

from model import SpokenDigitCNN


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 8000
TARGET_LENGTH = 8000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256

MODEL_PATH = "best_model_v3.1.pth"


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

if os.path.exists(MODEL_PATH):
    model = SpokenDigitCNN(num_classes=10).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    print(f"Model loaded on: {device}")

else:
    print(f"Model not found: {MODEL_PATH}")
    print("The interface will work, but prediction is disabled.")


# ============================================================
# MEL-SPECTROGRAMME
# ============================================================

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_audio(file_path):

    # Charger le fichier audio
    waveform, sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # Rééchantillonnage à 8000 Hz
    if sr != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    # Conversion en Tensor PyTorch
    waveform = torch.from_numpy(waveform).float()

    # Padding ou découpage à 8000 échantillons
    if waveform.shape[0] < TARGET_LENGTH:

        waveform = torch.nn.functional.pad(
            waveform,
            (0, TARGET_LENGTH - waveform.shape[0])
        )

    else:

        waveform = waveform[:TARGET_LENGTH]

    # Mel-Spectrogramme
    mel_spec = mel_transform(
        waveform.unsqueeze(0)
    )

    # Log Mel-Spectrogramme
    mel_spec = torch.log(
        mel_spec + 1e-9
    )

    # Instance Standardization
    mean = mel_spec.mean()
    std = mel_spec.std()

    mel_spec = (
        (mel_spec - mean)
        / (std + 1e-6)
    )

    return waveform, mel_spec


# ============================================================
# WAVEFORM
# ============================================================

def create_waveform_plot(waveform):

    waveform = waveform.numpy()

    time_axis = np.arange(len(waveform)) / SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(time_axis, waveform)

    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    ax.grid(True)

    plt.tight_layout()

    return fig


# ============================================================
# MEL-SPECTROGRAMME
# ============================================================

def create_mel_plot(mel_spec):

    mel = mel_spec.squeeze().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))

    image = ax.imshow(
        mel,
        aspect="auto",
        origin="lower"
    )

    ax.set_title("Log Mel-Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Frequency")

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    return fig


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(file_path):

    if file_path is None:
        return (
            "Please upload or record an audio file.",
            None,
            None,
            "No audio",
            None
        )

    start_time = time.time()

    try:

        # Prétraitement
        waveform, mel_spec = preprocess_audio(file_path)

        # Création des graphiques
        waveform_plot = create_waveform_plot(waveform)

        mel_plot = create_mel_plot(mel_spec)

        # Vérification du modèle
        if model is None:

            processing_time = time.time() - start_time

            return (
                "Model weights not found.\n"
                "Waiting for best_model_v2.2.pth",
                waveform_plot,
                mel_plot,
                f"{processing_time:.3f} seconds",
                None
            )

        # Ajouter dimension batch
        input_tensor = mel_spec.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_digit = torch.argmax(
                probabilities,
                dim=1
            ).item()

        processing_time = time.time() - start_time

        # Probabilités
        probs = probabilities[0].cpu().numpy()

        probability_text = "\n".join(
            [
                f"Digit {i}: {probs[i] * 100:.2f}%"
                for i in range(10)
            ]
        )

        return (
            f"Predicted digit: {predicted_digit}",
            waveform_plot,
            mel_plot,
            f"{processing_time:.3f} seconds",
            probability_text
        )

    except Exception as e:

        return (
            f"Error: {str(e)}",
            None,
            None,
            "Error",
            None
        )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Spoken Digit Recognition",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🎙️ Spoken Digit Recognition

        ### Recognize spoken digits from 0 to 9

        Upload an audio file or record your voice using the microphone.
        """
    )

    with gr.Row():

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        with gr.Column():

            audio_input = gr.Audio(
                label="Audio Input",
                sources=[
                    "upload",
                    "microphone"
                ],
                type="filepath"
            )

            predict_button = gr.Button(
                "🔍 Recognize Digit",
                variant="primary"
            )

            audio_player = gr.Audio(
                label="Audio"
            )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with gr.Column():

            prediction_output = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            probability_output = gr.Textbox(
                label="Class Probabilities",
                lines=10,
                interactive=False
            )

            processing_time_output = gr.Textbox(
                label="Processing Time",
                interactive=False
            )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    gr.Markdown("## 📈 Audio Analysis")

    with gr.Row():

        waveform_output = gr.Plot(
            label="Waveform"
        )

        mel_output = gr.Plot(
            label="Mel-Spectrogram"
        )

    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    predict_button.click(
        fn=predict_digit,
        inputs=audio_input,
        outputs=[
            prediction_output,
            waveform_output,
            mel_output,
            processing_time_output,
            probability_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()