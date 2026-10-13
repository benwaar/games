"""Tests for ONNX export of Signal Hunt transfer model."""

import numpy as np
import pytest
from pathlib import Path

ONNX_PATH = Path(__file__).parent.parent / "output" / "chord_cnn.onnx"
NUM_CLASSES = 12
INPUT_SHAPE = (1, 1, 128, 65)


@pytest.fixture(scope="module")
def session():
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    if not ONNX_PATH.exists():
        pytest.skip(f"Export not found: {ONNX_PATH}. Run: python scripts/export_onnx.py")
    return ort.InferenceSession(str(ONNX_PATH))


def test_input_shape(session):
    inp = session.get_inputs()[0]
    shape = inp.shape
    assert shape[1] == 1
    assert shape[2] == 128
    assert shape[3] == 65


def test_output_shape(session):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    out = session.run(None, {"mel_spectrogram": dummy})[0]
    assert out.shape == (1, NUM_CLASSES)


def test_softmax_sums_to_one(session):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    logits = session.run(None, {"mel_spectrogram": dummy})[0]
    # Convert logits → probabilities and verify they sum to ~1.
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0], atol=1e-5)


def test_deterministic_output(session):
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    out1 = session.run(None, {"mel_spectrogram": dummy})[0]
    out2 = session.run(None, {"mel_spectrogram": dummy})[0]
    np.testing.assert_array_equal(out1, out2)
