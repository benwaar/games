"""Export Signal Hunt transfer model to ONNX.

Loads the best fine-tuned checkpoint (12-class piano note classifier) and
exports to signal-hunt/output/chord_cnn.onnx.

Usage:
    python -m scripts.export_onnx
    python scripts/export_onnx.py
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Allow running from scripts/ directly or as module from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.cnn import SoundClassifier

CHECKPOINT = Path("output/transfer/finetune/best_model.pt")
OUTPUT = Path("output/chord_cnn.onnx")
NUM_CLASSES = 12
INPUT_SHAPE = (1, 1, 128, 65)  # (batch, channels, mels, frames)


def export(checkpoint: Path = CHECKPOINT, output: Path = OUTPUT) -> Path:
    checkpoint = Path(checkpoint)
    output = Path(output)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_classes = len(ckpt["label_map"])
    label_map: dict[str, int] = ckpt["label_map"]

    model = SoundClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(*INPUT_SHAPE)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=18,
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Inline all tensors — prevents split .onnx.data file (required for browser WASM).
    model_proto = onnx.load(str(output))
    onnx.save(model_proto, str(output), save_as_external_data=False)

    # Verify the export loads and produces the right output shape.
    session = ort.InferenceSession(str(output))
    out = session.run(None, {"mel_spectrogram": dummy.numpy()})[0]
    assert out.shape == (1, num_classes), f"Unexpected output shape: {out.shape}"

    # Reverse the label map so index → note name.
    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"Exported: {output}")
    print(f"Input:    {session.get_inputs()[0].shape}")
    print(f"Output:   {session.get_outputs()[0].shape}  ({num_classes} classes)")
    print(f"Labels:   {[idx_to_label[i] for i in range(num_classes)]}")
    val_acc = ckpt.get('val_acc')
    print(f"Val acc at export: {val_acc:.1%}" if val_acc is not None else "Val acc at export: n/a")

    return output


if __name__ == "__main__":
    export()
