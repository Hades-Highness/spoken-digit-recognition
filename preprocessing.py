"""Audio preprocessing for DigitSense inference: VAD, denoising, and feature
extraction. Mirrors the exact feature pipeline used to train the v4.0
checkpoint (see train.py's get_gpu_transforms / process_batch_gpu and
evaluate.py's evaluate_version) since dataset.py no longer defines these
constants itself — it just yields raw waveforms now.
"""

import logging

import av
import librosa
import noisereduce as nr
import numpy as np
import torch
import torchaudio.functional as FA
import torchaudio.transforms as T
from silero_vad import get_speech_timestamps, load_silero_vad

logger = logging.getLogger(__name__)

# --- Feature extraction constants (must match train.py / evaluate.py) -------
SAMPLE_RATE = 16000
TARGET_LENGTH = 16000  # 1 second at 16 kHz
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# --- VAD constants ------------------------------------------------------------
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 100  # digits can be short bursts; keep this low
MIN_SILENCE_DURATION_MS = 100  # gap needed to treat two bursts as separate digits
SPEECH_PAD_MS = 30  # padding kept around each detected burst

_vad_model = None
_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)


class AudioProcessingError(Exception):
    """Raised when an input clip can't be turned into a usable feature tensor."""


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def _decode_with_av(file_path):
    """Decode to mono float32 at SAMPLE_RATE via PyAV.

    Browser microphone recordings arrive as WebM/Opus (MediaRecorder does not
    support audio/wav), which libsndfile cannot open. PyAV ships its own FFmpeg
    libraries, so this works without a system FFmpeg install.
    """
    with av.open(file_path) as container:
        if not container.streams.audio:
            raise AudioProcessingError("That file doesn't contain an audio track.")

        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(container.streams.audio[0]):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        raise AudioProcessingError("The recording is empty.")

    return np.concatenate(chunks).astype(np.float32)


def load_audio(file_path):
    """Load a clip as mono float32 at SAMPLE_RATE. Raises AudioProcessingError
    on anything unreadable, silent, or too short to be meaningful speech."""
    try:
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as librosa_exc:
        # libsndfile rejects browser recording containers (WebM/Opus); fall back
        # to PyAV before giving up.
        logger.info(
            "librosa could not decode '%s' (%s); retrying with PyAV.",
            file_path, librosa_exc,
        )
        try:
            waveform = _decode_with_av(file_path)
        except AudioProcessingError:
            raise
        except Exception as av_exc:
            # Log real causes server-side; never echo raw library/OS error text
            # (which can include filesystem paths) back into the UI.
            logger.warning("Failed to decode audio file '%s': %s", file_path, av_exc)
            raise AudioProcessingError(
                "Couldn't read that audio file — try a different recording or format."
            ) from av_exc

    if waveform.size == 0:
        raise AudioProcessingError("The recording is empty.")

    if not np.isfinite(waveform).all():
        raise AudioProcessingError("The recording contains invalid audio data.")

    peak = np.abs(waveform).max()
    if peak < 1e-4:
        raise AudioProcessingError("The recording is silent — no audio was captured.")

    return waveform


def denoise(waveform):
    """Spectral-gate noise reduction. Falls back to the original signal if
    noisereduce chokes on a pathological clip rather than failing the request."""
    try:
        return nr.reduce_noise(y=waveform, sr=SAMPLE_RATE).astype(np.float32)
    except Exception as exc:
        logger.warning("noisereduce failed (%s); continuing with raw audio.", exc)
        return waveform


def detect_speech_segments(waveform):
    """Return VAD speech spans as a list of (start_sample, end_sample) tuples,
    sorted chronologically. Empty list if no speech is detected."""
    model = _get_vad_model()
    audio_tensor = torch.from_numpy(waveform).float()

    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
        return_seconds=False,
    )
    return [(ts["start"], ts["end"]) for ts in timestamps]


def _fix_length(waveform):
    if waveform.shape[0] < TARGET_LENGTH:
        return torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[0]))
    return waveform[:TARGET_LENGTH]


def extract_features(waveform_segment):
    """Turn a 1-D float32 waveform (numpy array or tensor) into the 3-channel
    [3, N_MELS, time] feature tensor the model expects: log-mel + delta +
    delta-delta, per-instance standardized. Same recipe as train.py."""
    if isinstance(waveform_segment, np.ndarray):
        waveform_segment = torch.from_numpy(waveform_segment).float()

    waveform_segment = _fix_length(waveform_segment)

    mel_spec = _mel_transform(waveform_segment.unsqueeze(0))
    log_mel = torch.log(mel_spec + 1e-9)

    delta = FA.compute_deltas(log_mel)
    delta_delta = FA.compute_deltas(delta)

    features = torch.stack([log_mel.squeeze(0), delta.squeeze(0), delta_delta.squeeze(0)], dim=0)

    mean = features.mean(dim=(-2, -1), keepdim=True)
    std = features.std(dim=(-2, -1), keepdim=True)
    features = (features - mean) / (std + 1e-6)

    return features


def prepare_single(file_path):
    """Full pipeline for Single Digit mode: load, denoise, VAD-trim to the
    speech span, extract features. Returns (waveform_np, segments, feature_tensor)."""
    waveform = load_audio(file_path)
    waveform = denoise(waveform)

    segments = detect_speech_segments(waveform)
    if not segments:
        raise AudioProcessingError("No speech detected in the recording.")

    start, end = segments[0][0], segments[-1][1]
    trimmed = waveform[start:end]

    features = extract_features(trimmed)
    return waveform, segments, features


def prepare_multi(file_path):
    """Full pipeline for Multi-Digit mode: load, denoise, split into per-burst
    VAD segments, extract features for each. Returns (waveform_np, segments, [feature_tensor, ...])."""
    waveform = load_audio(file_path)
    waveform = denoise(waveform)

    segments = detect_speech_segments(waveform)
    if not segments:
        raise AudioProcessingError("No speech detected in the recording.")

    feature_list = [extract_features(waveform[start:end]) for start, end in segments]
    return waveform, segments, feature_list
