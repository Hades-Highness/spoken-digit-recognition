import os
import time

import gradio as gr
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torchaudio.functional as F_audio

from model import SpokenDigitCNN


# ============================================================
# CONFIGURATION ET REGISTRE DES MODÈLES (v4.0 @ 16 kHz)
# ============================================================

SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
MODELS_DIR = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_model = None
current_model_name = None
current_in_channels = 3


def get_available_models():
    """Liste tous les fichiers de poids disponibles dans la racine ou models/."""
    search_dirs = [".", MODELS_DIR]
    files = []
    
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pth") or f.endswith(".pt"):
                    path = os.path.join(d, f) if d != "." else f
                    files.append(path)
                    
    files.sort()
    return files


def load_model_by_name(model_path):
    """Charge un modèle PyTorch à partir de son chemin."""
    global current_model, current_model_name, current_in_channels

    if not model_path:
        current_model = None
        current_model_name = None
        return "⚠️ Aucun modèle sélectionné."

    if not os.path.exists(model_path):
        return f"❌ Fichier introuvable : {model_path}"

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Détection automatique du nombre de canaux d'entrée à partir des poids
        in_channels = 3
        for key, tensor in checkpoint.items():
            if "weight" in key and len(tensor.shape) == 4:
                in_channels = tensor.shape[1]
                break

        try:
            model = SpokenDigitCNN(num_classes=10, in_channels=in_channels).to(device)
        except TypeError:
            model = SpokenDigitCNN(num_classes=10).to(device)

        model.load_state_dict(checkpoint)
        model.eval()

        current_model = model
        current_model_name = model_path
        current_in_channels = in_channels

        return f"✅ Modèle '{os.path.basename(model_path)}' chargé ({in_channels} canaux) sur {device}."

    except Exception as e:
        current_model = None
        current_model_name = None
        return f"❌ Erreur lors du chargement : {str(e)}"


# ============================================================
# PREPROCESSING v4.0 (16 kHz / 3 Canaux)
# ============================================================

def preprocess_audio(file_path, in_channels=3):
    # 1. Chargement et rééchantillonnage à 16 kHz
    waveform, sr = librosa.load(file_path, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)

    waveform = torch.from_numpy(waveform).float()

    # 2. Ajustement de la longueur à 1 seconde (16000 échantillons)
    if waveform.shape[0] < TARGET_LENGTH:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[0]))
    else:
        waveform = waveform[:TARGET_LENGTH]

    # 3. Calcul Log-Mel Spectrogram (16kHz, n_fft=1024, hop=256)
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    mel_spec = mel_transform(waveform.unsqueeze(0))
    log_mel = torch.log(mel_spec + 1e-9)

    # 4. Construction multi-canaux (Log-Mel, Delta, Delta-Delta)
    if in_channels == 3:
        delta1 = F_audio.compute_deltas(log_mel)
        delta2 = F_audio.compute_deltas(delta1)
        feature_tensor = torch.stack([log_mel.squeeze(0), delta1.squeeze(0), delta2.squeeze(0)], dim=0)
    else:
        feature_tensor = log_mel

    # 5. Instance Standardization (exactement comme dans train.py)
    mean = feature_tensor.mean(dim=(-2, -1), keepdim=True)
    std = feature_tensor.std(dim=(-2, -1), keepdim=True)
    feature_tensor = (feature_tensor - mean) / (std + 1e-6)

    return waveform, feature_tensor


# ============================================================
# VISUALISATION
# ============================================================

def create_waveform_plot(waveform):
    waveform = waveform.numpy()
    time_axis = np.arange(len(waveform)) / SAMPLE_RATE

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(time_axis, waveform, color="#1f77b4")
    ax.set_title("Audio Waveform (16 kHz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def create_mel_plot(feature_tensor):
    mel = feature_tensor[0].numpy() if feature_tensor.dim() == 3 else feature_tensor.squeeze().numpy()

    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Log Mel-Spectrogram (Channel 0)")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    return fig


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(file_path, selected_model_name):
    if file_path is None:
        return "Veuillez importer ou enregistrer un fichier audio.", None, None, "Aucun audio", None

    if current_model is None or current_model_name != selected_model_name:
        status = load_model_by_name(selected_model_name)
        if current_model is None:
            return f"Erreur modèle : {status}", None, None, "Erreur", None

    start_time = time.time()

    try:
        waveform, feature_tensor = preprocess_audio(file_path, in_channels=current_in_channels)

        waveform_plot = create_waveform_plot(waveform)
        mel_plot = create_mel_plot(feature_tensor)

        input_tensor = feature_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = current_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()

        processing_time = time.time() - start_time
        probs = probabilities[0].cpu().numpy()

        probability_text = "\n".join([f"Chiffre {i}: {probs[i] * 100:.2f}%" for i in range(10)])

        return (
            f"Chiffre prédit : {predicted_digit}",
            waveform_plot,
            mel_plot,
            f"{processing_time * 1000:.1f} ms",
            probability_text
        )

    except Exception as e:
        return f"Erreur : {str(e)}", None, None, "Erreur", None


# ============================================================
# GRADIO INTERFACE
# ============================================================

available_models = get_available_models()
default_model = available_models[0] if available_models else None

with gr.Blocks(title="Spoken Digit Recognition v4.0", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🎙️ Spoken Digit Recognition (v4.0 @ 16 kHz)
        Sélectionne ton checkpoint `.pth`, enregistre ta voix ou importe un fichier audio pour tester la prédiction.
        """
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=available_models,
            value=default_model,
            label="Sélectionner un checkpoint (.pth)",
            interactive=True
        )
        refresh_btn = gr.Button("🔄 Rafraîchir", scale=0)

    model_status = gr.Textbox(
        value=load_model_by_name(default_model) if default_model else "Aucun fichier .pth trouvé",
        label="Statut du Modèle",
        interactive=False
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Entrée Audio",
                sources=["upload", "microphone"],
                type="filepath"
            )
            predict_button = gr.Button("🔍 Reconnaître le Chiffre", variant="primary")

        with gr.Column():
            prediction_output = gr.Textbox(label="Résultat", interactive=False)
            probability_output = gr.Textbox(label="Probabilités par classe", lines=10, interactive=False)
            processing_time_output = gr.Textbox(label="Temps d'inférence", interactive=False)

    gr.Markdown("## 📈 Analyse Spectrale")

    with gr.Row():
        waveform_output = gr.Plot(label="Forme d'onde")
        mel_output = gr.Plot(label="Log Mel-Spectrogramme")

    def refresh_models():
        models = get_available_models()
        new_default = models[0] if models else None
        status = load_model_by_name(new_default)
        return gr.update(choices=models, value=new_default), status

    refresh_btn.click(fn=refresh_models, inputs=[], outputs=[model_dropdown, model_status])
    model_dropdown.change(fn=load_model_by_name, inputs=model_dropdown, outputs=model_status)

    predict_button.click(
        fn=predict_digit,
        inputs=[audio_input, model_dropdown],
        outputs=[prediction_output, waveform_output, mel_output, processing_time_output, probability_output]
    )

if __name__ == "__main__":
    demo.launch()