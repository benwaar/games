"""Tests for M10: ONNX → INT8 → Z80 inference pipeline.

Tests cover:
- ONNX parser: layer shapes, weight dimensions
- Quantisation: INT8 range, scale factors
- Z80 code gen: correct assembly labels in output
- Assembly: generated asm assembles without errors
- Inference: Z80 argmax matches float32 reference on 100+ inputs
- Accuracy: >80% agreement (expected ~100% for this tiny test model)

All tests are skipped gracefully when optional dependencies (onnx,
onnxruntime, numpy) are not installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Skip everything if ML deps are unavailable
try:
    import numpy as np  # type: ignore[import-untyped]
    import onnx  # type: ignore[import-untyped]
    import onnxruntime as ort  # type: ignore[import-untyped]

    ML_DEPS_AVAILABLE = True
except ImportError:
    ML_DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ML_DEPS_AVAILABLE,
    reason="onnx, onnxruntime, numpy not installed",
)

# ---------------------------------------------------------------------------
# Fixture: test model path
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).parent
_MODEL_PATH = _TESTS_DIR / "test_model.onnx"


@pytest.fixture(scope="session")
def test_model_path() -> Path:
    """Create (or reuse) the 4→3→2 test model."""
    if not _MODEL_PATH.exists():
        _create_test_model()
    return _MODEL_PATH


def _create_test_model() -> None:
    """Build a small deterministic ONNX model and save it."""
    import onnx.numpy_helper as nph
    from onnx import TensorProto, helper

    rng = np.random.default_rng(42)
    w1 = (rng.standard_normal((3, 4)) * 0.5).astype(np.float32)
    b1 = (rng.standard_normal(3) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((2, 3)) * 0.5).astype(np.float32)
    b2 = (rng.standard_normal(2) * 0.1).astype(np.float32)

    W1 = nph.from_array(w1, name="W1")
    B1 = nph.from_array(b1, name="B1")
    W2 = nph.from_array(w2, name="W2")
    B2 = nph.from_array(b2, name="B2")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])

    gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["H_pre"], transB=1)
    relu1 = helper.make_node("Relu", inputs=["H_pre"], outputs=["H"])
    gemm2 = helper.make_node("Gemm", inputs=["H", "W2", "B2"], outputs=["Y"], transB=1)

    graph = helper.make_graph([gemm1, relu1, gemm2], "tiny_test", [X], [Y], [W1, B1, W2, B2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    onnx.save(model, str(_MODEL_PATH))


# ---------------------------------------------------------------------------
# ONNX parser tests
# ---------------------------------------------------------------------------


def test_parse_layers(test_model_path):
    """Parser extracts the correct layer sequence."""
    from incant.onnx_parser import parse_onnx

    parsed = parse_onnx(str(test_model_path))
    layers = parsed["layers"]

    assert len(layers) == 3
    assert layers[0]["type"] == "linear"
    assert layers[1]["type"] == "relu"
    assert layers[2]["type"] == "linear"


def test_parse_shapes(test_model_path):
    """Parser extracts correct in/out dimensions."""
    from incant.onnx_parser import parse_onnx

    parsed = parse_onnx(str(test_model_path))
    fc1 = parsed["layers"][0]
    fc2 = parsed["layers"][2]

    assert fc1["in"] == 4
    assert fc1["out"] == 3
    assert fc2["in"] == 3
    assert fc2["out"] == 2


def test_parse_weight_shapes(test_model_path):
    """Weights have expected (out, in) shape."""
    from incant.onnx_parser import parse_onnx

    parsed = parse_onnx(str(test_model_path))
    fc1 = parsed["layers"][0]
    fc2 = parsed["layers"][2]

    assert fc1["weight"].shape == (3, 4)
    assert fc1["bias"].shape == (3,)
    assert fc2["weight"].shape == (2, 3)
    assert fc2["bias"].shape == (2,)


def test_architecture_summary(test_model_path):
    """Architecture summary string is non-empty and contains dimensions."""
    from incant.onnx_parser import architecture_summary, parse_onnx

    parsed = parse_onnx(str(test_model_path))
    summary = architecture_summary(parsed)

    assert "4" in summary
    assert "3" in summary
    assert "2" in summary


# ---------------------------------------------------------------------------
# Quantisation tests
# ---------------------------------------------------------------------------


def test_quantise_int8_range(test_model_path):
    """All quantised weights are in INT8 range [-128, 127]."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import quantise_model

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)

    for layer in quantised["layers"]:
        if layer["type"] != "linear":
            continue
        w = layer["weight"]
        assert w.dtype == np.int8, f"expected int8, got {w.dtype}"
        assert int(w.min()) >= -128
        assert int(w.max()) <= 127


def test_quantise_scale_positive(test_model_path):
    """Scale factors are positive floats."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import quantise_model

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)

    for layer in quantised["layers"]:
        if layer["type"] != "linear":
            continue
        assert layer["scale_w"] > 0
        assert layer["scale_in"] > 0
        assert layer["scale_out"] > 0


def test_quantise_bias_dtype(test_model_path):
    """Biases are stored as int32."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import quantise_model

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)

    for layer in quantised["layers"]:
        if layer["type"] != "linear":
            continue
        assert layer["bias"].dtype == np.int32, f"bias dtype={layer['bias'].dtype}"


def test_int8_forward_argmax_matches_float(test_model_path):
    """Python INT8 forward pass argmax matches float32 onnxruntime."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import compare_accuracy, quantise_model

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)

    result = compare_accuracy(parsed, quantised, str(test_model_path), n_samples=100)
    assert result["argmax_agreement"] >= 0.80, (
        f"INT8 argmax agreement too low: {result['argmax_agreement']:.0%}"
    )


# ---------------------------------------------------------------------------
# Z80 code generation tests
# ---------------------------------------------------------------------------


def test_z80_codegen_labels(test_model_path):
    """Generated Z80 asm contains all expected labels."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import quantise_model
    from incant.z80_infer import generate_z80_inference

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)
    asm = generate_z80_inference(quantised)

    for label in [
        "signed_mul",
        "matmul_layer",
        "relu_buf",
        "argmax_buf",
        "inference",
        "fc1_weights",
        "fc1_bias",
        "fc2_weights",
        "fc2_bias",
    ]:
        assert label + ":" in asm, f"Missing label: {label}"


def test_z80_codegen_assembles(test_model_path):
    """Generated Z80 asm assembles without errors."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import quantise_model
    from incant.targets.z80 import assemble
    from incant.z80_infer import generate_z80_inference

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)
    asm = generate_z80_inference(quantised)
    ok, binary, err = assemble(asm)

    assert ok, f"Assembly failed: {err}"
    assert binary is not None
    assert len(binary) > 0


# ---------------------------------------------------------------------------
# Z80 inference accuracy test
# ---------------------------------------------------------------------------


def test_z80_inference_accuracy(test_model_path):
    """Z80 inference argmax matches float32 reference on 100 inputs."""
    from incant.onnx_parser import parse_onnx
    from incant.quantise import quantise_model
    from incant.z80_infer import run_z80_inference

    parsed = parse_onnx(str(test_model_path))
    quantised = quantise_model(parsed)

    sess = ort.InferenceSession(str(test_model_path))
    rng = np.random.default_rng(42)
    in_dim = parsed["input_shape"][-1]

    agree = 0
    n = 100
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0, size=(1, in_dim)).astype(np.float32)
        ref_out = sess.run(None, {sess.get_inputs()[0].name: x})[0].flatten()
        ref_am = int(np.argmax(ref_out))

        x_int8 = np.clip(np.round(x.flatten() * 127.0), -128, 127).astype(np.int8)
        x_bytes = bytes([b & 0xFF for b in x_int8.tolist()])
        z80_am = run_z80_inference(quantised, x_bytes, ticks=5_000_000)

        agree += int(z80_am == ref_am)

    rate = agree / n
    assert rate >= 0.80, f"Z80 argmax agreement too low: {rate:.0%} (need ≥80%)"
