"""Model loading and prediction for DigitSense.

Tries ONNX Runtime first (models/onnx/*.onnx), falls back to the PyTorch
checkpoint (models/*.pth) if no ONNX file exists or the ONNX graph turns out
to be incompatible with the real feature shape.
"""

import glob
import logging
import os

import numpy as np
import torch

from model import SpokenDigitCNN

logger = logging.getLogger(__name__)

MODEL_VERSION_LABEL = "Model v4.0.0"  # static UI label only, see app.py

PTH_DIR = "models"
ONNX_DIR = "onnx"
PTH_PATTERN = "digitsense_v4.0.pth"
ONNX_PATTERN = "digitsense_v4.0.onnx"

CONFIDENCE_THRESHOLD = 0.60


class ModelUnavailableError(Exception):
    """Raised when no usable model (ONNX or PyTorch) can be loaded."""


class InferenceEngine:
    """Wraps whichever backend (ONNX Runtime or PyTorch) is actually usable.

    Resolution order: ONNX Runtime is tried first if a matching .onnx file
    exists; if it's missing, or its input shape doesn't match the real
    feature tensor shape, this falls back to the PyTorch .pth checkpoint.
    """

    def __init__(self):
        self.backend = None  # "onnx" | "pytorch" | None
        self.status_message = ""
        self._onnx_session = None
        self._onnx_input_name = None
        self._torch_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _find_file(self, directory, pattern):
        exact = os.path.join(directory, pattern)
        if os.path.exists(exact):
            return exact
        matches = sorted(glob.glob(os.path.join(directory, "*" + os.path.splitext(pattern)[1])))
        return matches[-1] if matches else None

    def _load(self):
        onnx_path = self._find_file(ONNX_DIR, ONNX_PATTERN)
        pth_path = self._find_file(PTH_DIR, PTH_PATTERN)

        if onnx_path and self._try_load_onnx(onnx_path):
            return

        if pth_path and self._try_load_pytorch(pth_path):
            if onnx_path:
                logger.warning(
                    "ONNX model at %s was found but incompatible; using PyTorch instead.",
                    onnx_path,
                )
            return

        self.backend = None
        self.status_message = (
            "No trained model found — waiting on teammate. "
            f"Expected a checkpoint at '{PTH_DIR}/{PTH_PATTERN}' or "
            f"'{ONNX_DIR}/{ONNX_PATTERN}'."
        )
        logger.error(self.status_message)

    def _try_load_onnx(self, onnx_path):
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed; skipping ONNX backend.")
            return False

        try:
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_meta = session.get_inputs()[0]
            declared_shape = input_meta.shape  # e.g. ['batch_size', 3, 64, 64]

            # The real feature tensor is [1, 3, N_MELS, 63] (see preprocessing.py).
            # If the export baked in a different static time dimension, this
            # backend is unusable — fail loudly here so we fall back to PyTorch
            # instead of feeding it a shape it will reject at inference time.
            expected_time_frames = 63
            declared_time = declared_shape[3] if len(declared_shape) == 4 else None
            if isinstance(declared_time, int) and declared_time != expected_time_frames:
                logger.warning(
                    "ONNX model '%s' declares a static input shape %s "
                    "(expects %d time frames) but the real feature pipeline "
                    "produces %d frames. This looks like a stale/mismatched "
                    "ONNX export — falling back to the PyTorch checkpoint. "
                    "This needs a re-export on the training side.",
                    onnx_path, declared_shape, declared_time, expected_time_frames,
                )
                return False

            self._onnx_session = session
            self._onnx_input_name = input_meta.name
            self.backend = "onnx"
            self.status_message = f"Loaded ONNX model: {os.path.basename(onnx_path)}"
            return True
        except Exception as exc:
            logger.warning("Failed to load ONNX model '%s': %s", onnx_path, exc)
            return False

    def _try_load_pytorch(self, pth_path):
        try:
            model = SpokenDigitCNN(num_classes=10).to(self._device)
            state_dict = torch.load(pth_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.eval()

            self._torch_model = model
            self.backend = "pytorch"
            self.status_message = (
                f"ONNX unavailable — using PyTorch checkpoint: {os.path.basename(pth_path)} "
                f"(device: {self._device})"
            )
            return True
        except Exception as exc:
            logger.error("Failed to load PyTorch checkpoint '%s': %s", pth_path, exc)
            return False

    @property
    def is_ready(self):
        return self.backend is not None

    def predict(self, feature_tensor):
        """feature_tensor: torch.Tensor [3, N_MELS, time]. Returns (digit, confidence, probs)."""
        if not self.is_ready:
            raise ModelUnavailableError(self.status_message)

        batch = feature_tensor.unsqueeze(0)

        if self.backend == "onnx":
            inputs = {self._onnx_input_name: batch.numpy().astype(np.float32)}
            logits = self._onnx_session.run(None, inputs)[0]
            probs = _softmax_np(logits[0])
        else:
            with torch.no_grad():
                logits = self._torch_model(batch.to(self._device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        return digit, confidence, probs


def _softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
