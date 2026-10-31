"""Export and dynamically quantize the DigitSense v5.0 raw-audio model."""

import argparse
import os

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from model import SpokenDigitCNN, TARGET_LENGTH

DEFAULT_CHECKPOINT = "models/digitsense_v5.0.pth"
DEFAULT_OUTPUT = "onnx/digitsense_v5.0.onnx"
OPSET_VERSION = 17


def export_model(checkpoint_path=DEFAULT_CHECKPOINT, output_path=DEFAULT_OUTPUT):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_directory = os.path.dirname(output_path) or "."
    os.makedirs(output_directory, exist_ok=True)
    quantized_path = os.path.splitext(output_path)[0] + ".int8.onnx"

    device = torch.device("cpu")
    model = SpokenDigitCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    dummy_audio = torch.zeros(1, TARGET_LENGTH, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        dummy_audio,
        output_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    onnx.checker.check_model(output_path)
    # Keep the Log-Mel projection full precision and quantize the classifier.
    quantize_dynamic(
        model_input=output_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=["/feature_extractor/MatMul"],
    )
    onnx.checker.check_model(quantized_path)
    print(f"Exported opset {OPSET_VERSION} model to {output_path}.")
    print(f"Exported dynamically quantized INT8 model to {quantized_path}.")
    return output_path, quantized_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
