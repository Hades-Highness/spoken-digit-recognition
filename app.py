"""DigitSense — Gradio interface for spoken-digit recognition.

Pulls preprocessing (VAD, denoising, feature extraction) from preprocessing.py
and model loading/inference from inference.py. See those files for the
feature-pipeline and ONNX/PyTorch backend details.
"""

import logging

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from inference import CONFIDENCE_THRESHOLD, MODEL_VERSION_LABEL, InferenceEngine
from preprocessing import AudioProcessingError, prepare_multi, prepare_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODE_SINGLE = "Single Digit"
MODE_MULTI = "Multi-Digit"

# --- Palette (see design plan) -----------------------------------------------
INK = "#12141C"
PANEL = "#1B1E29"
SIGNAL = "#5EEAD4"
EMBER = "#F5A65B"
ALERT = "#EF6461"
MIST = "#9AA0B4"

engine = InferenceEngine()


# ==============================================================================
# Plotting
# ==============================================================================

def _style_axes(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MIST, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(MIST)
        spine.set_alpha(0.3)
    ax.xaxis.label.set_color(MIST)
    ax.yaxis.label.set_color(MIST)
    ax.title.set_color(MIST)


def plot_waveform(waveform, sample_rate=16000, segments=None):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor(INK)
    time_axis = np.arange(len(waveform)) / sample_rate

    ax.plot(time_axis, waveform, color=SIGNAL, linewidth=0.8)
    if segments:
        for start, end in segments:
            ax.axvspan(start / sample_rate, end / sample_rate, color=EMBER, alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    _style_axes(ax)
    plt.tight_layout()
    return fig


def plot_mel(feature_tensor):
    log_mel = feature_tensor[0].numpy()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    fig.patch.set_facecolor(INK)
    ax.imshow(log_mel, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Bin")
    _style_axes(ax)
    plt.tight_layout()
    return fig


# ==============================================================================
# Result rendering (HTML built only from numeric model output — no user text)
# ==============================================================================

def _confidence_color(confidence):
    if confidence >= CONFIDENCE_THRESHOLD:
        return SIGNAL if confidence >= 0.85 else EMBER
    return ALERT


def render_single_result(digit, confidence):
    accepted = confidence >= CONFIDENCE_THRESHOLD
    color = _confidence_color(confidence)
    pct = confidence * 100

    if accepted:
        digit_html = f'<div class="digit-display" style="color:{SIGNAL}">{digit}</div>'
    else:
        digit_html = (
            f'<div class="reject-message" style="color:{ALERT}">'
            "Didn't catch that clearly — try again a little closer to the mic."
            "</div>"
        )

    bar_html = (
        '<div class="confidence-track">'
        f'<div class="confidence-fill" style="width:{pct:.1f}%; background:{color}"></div>'
        "</div>"
        f'<div class="confidence-label" style="color:{MIST}">'
        f"confidence {pct:.1f}% &middot; threshold {CONFIDENCE_THRESHOLD * 100:.0f}%</div>"
    )
    return digit_html + bar_html


def render_multi_result(results):
    """results: list of (digit, confidence) tuples, one per detected burst."""
    chips = []
    for digit, confidence in results:
        color = _confidence_color(confidence)
        if confidence >= CONFIDENCE_THRESHOLD:
            chips.append(f'<span class="digit-chip" style="color:{color}">{digit}</span>')
        else:
            chips.append(f'<span class="digit-chip" style="color:{ALERT}">?</span>')

    joined = f' <span style="color:{MIST}">-</span> '.join(chips)
    avg_conf = sum(c for _, c in results) / len(results) * 100

    return (
        f'<div class="digit-display digit-display--multi">{joined}</div>'
        f'<div class="confidence-label" style="color:{MIST}">'
        f"{len(results)} digit(s) detected &middot; avg confidence {avg_conf:.1f}%</div>"
    )


def render_error(message):
    return f'<div class="reject-message" style="color:{ALERT}">{message}</div>'


# ==============================================================================
# Prediction handlers
# ==============================================================================

def predict(file_path, mode):
    empty_plot = None

    if file_path is None:
        return render_error("Record or upload an audio clip first."), empty_plot, empty_plot, ""

    if not engine.is_ready:
        return render_error(engine.status_message), empty_plot, empty_plot, ""

    try:
        if mode == MODE_MULTI:
            waveform, segments, feature_list = prepare_multi(file_path)

            results = [engine.predict(f)[:2] for f in feature_list]
            result_html = render_multi_result(results)

            waveform_plot = plot_waveform(waveform, segments=segments)
            mel_plot = plot_mel(feature_list[0])
            probs_text = "\n".join(
                f"Segment {i + 1}: digit {d} ({c * 100:.1f}%)"
                for i, (d, c) in enumerate(results)
            )
        else:
            waveform, segments, feature_tensor = prepare_single(file_path)
            digit, confidence, probs = engine.predict(feature_tensor)
            result_html = render_single_result(digit, confidence)

            waveform_plot = plot_waveform(waveform, segments=segments)
            mel_plot = plot_mel(feature_tensor)
            probs_text = "\n".join(f"Digit {i}: {probs[i] * 100:.2f}%" for i in range(10))

        return result_html, waveform_plot, mel_plot, probs_text

    except AudioProcessingError as exc:
        logger.info("Audio rejected: %s", exc)
        return render_error(str(exc)), empty_plot, empty_plot, ""
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        return (
            render_error("Something went wrong processing that clip. Please try a different recording."),
            empty_plot,
            empty_plot,
            "",
        )


# ==============================================================================
# Theme
# ==============================================================================

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Space Mono"), "monospace"],
).set(
    body_background_fill=INK,
    background_fill_primary=PANEL,
    background_fill_secondary=INK,
    block_background_fill=PANEL,
    block_border_color=MIST,
    block_label_text_color=MIST,
    body_text_color=MIST,
    body_text_color_subdued=MIST,
    button_primary_background_fill=SIGNAL,
    button_primary_background_fill_hover=SIGNAL,
    button_primary_text_color=INK,
    border_color_primary=MIST,
    input_background_fill=INK,
)

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

.app-header h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    margin-bottom: 0;
}}

.version-badge {{
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: {MIST};
    border: 1px solid {MIST};
    border-radius: 4px;
    padding: 2px 8px;
    opacity: 0.8;
}}

.digit-display {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 4.5rem;
    text-align: center;
    line-height: 1.1;
    padding: 0.5rem 0 0.25rem 0;
}}

.digit-display--multi {{
    font-size: 3rem;
}}

.digit-chip {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
}}

.reject-message {{
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    text-align: center;
    padding: 1.5rem 0;
}}

.confidence-track {{
    width: 100%;
    height: 6px;
    background: {INK};
    border-radius: 3px;
    overflow: hidden;
    margin-top: 0.25rem;
}}

.confidence-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}}

.confidence-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-align: center;
    margin-top: 0.4rem;
}}
"""


# ==============================================================================
# Interface
# ==============================================================================

with gr.Blocks(title="DigitSense") as demo:
    with gr.Row(elem_classes=["app-header"]):
        with gr.Column():
            gr.Markdown("# DigitSense")
            gr.Markdown("Spoken digit recognition")
        with gr.Column(scale=0, min_width=140):
            gr.HTML(f'<div class="version-badge">{MODEL_VERSION_LABEL}</div>')

    if not engine.is_ready:
        gr.HTML(render_error(engine.status_message))
    else:
        gr.HTML(f'<div class="confidence-label" style="color:{MIST}">{engine.status_message}</div>')

    with gr.Row():
        with gr.Column():
            mode_selector = gr.Radio(
                choices=[MODE_SINGLE, MODE_MULTI],
                value=MODE_SINGLE,
                label="Mode",
            )
            audio_input = gr.Audio(
                label="Record or upload",
                sources=["microphone", "upload"],
                type="filepath",
            )
            predict_button = gr.Button("Recognize", variant="primary")

        with gr.Column():
            result_display = gr.HTML(
                f'<div class="reject-message" style="color:{MIST}">'
                "Record or upload a clip, then press Recognize."
                "</div>"
            )

    with gr.Accordion("Details", open=False):
        with gr.Row():
            waveform_output = gr.Plot(label="Waveform")
            mel_output = gr.Plot(label="Log Mel-Spectrogram")
        probs_output = gr.Textbox(label="Per-digit / per-segment detail", lines=10, interactive=False)

    predict_button.click(
        fn=predict,
        inputs=[audio_input, mode_selector],
        outputs=[result_display, waveform_output, mel_output, probs_output],
    )

if __name__ == "__main__":
    demo.launch(theme=theme, css=CUSTOM_CSS)
