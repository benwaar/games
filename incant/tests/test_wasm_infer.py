"""Tests for wasm_infer.generate_wat_inference.

Verifies that WAT output:
1. Assembles without error (gate: wat2wasm)
2. Produces argmax agreement >= 95% vs onnxruntime on 100 random inputs
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

RUNNER = Path(__file__).parent / "wasm_infer_runner.js"
TEST_ONNX = Path(__file__).parent / "test_model.onnx"


def _require_deps():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        pytest.skip("onnxruntime not installed")


def _require_node():
    r = subprocess.run(["which", "node"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("node not on PATH")


def _require_wat2wasm():
    r = subprocess.run(["which", "wat2wasm"], capture_output=True)
    if r.returncode != 0:
        pytest.skip("wat2wasm not on PATH")


@pytest.fixture(scope="module")
def wasm_path():
    _require_node()
    _require_wat2wasm()
    from incant.onnx_parser import parse_onnx
    from incant.wasm_infer import generate_wat_inference
    from incant.targets.wat import assemble

    parsed = parse_onnx(str(TEST_ONNX))
    wat = generate_wat_inference(parsed)
    ok, path, err = assemble(wat)
    assert ok, f"wat2wasm failed: {err}"
    assert path is not None
    return path


@pytest.fixture(scope="module")
def ort_session():
    ort = _require_deps()
    return ort.InferenceSession(str(TEST_ONNX))


def _wasm_infer(wasm_path: Path, x: np.ndarray) -> int:
    hex_args = [struct.pack("<f", float(v)).hex() for v in x.flatten()]
    r = subprocess.run(
        ["node", str(RUNNER), str(wasm_path)] + hex_args,
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"node runner failed: {r.stderr}"
    return int(r.stdout.strip())


def test_assembles(wasm_path):
    """wat2wasm accepted the generated WAT (fixture would fail otherwise)."""
    assert wasm_path.exists()


def test_argmax_agreement(wasm_path, ort_session):
    """100 random inputs: WASM and ORT argmax agree >= 95% of the time."""
    in_name = ort_session.get_inputs()[0].name
    in_dim = ort_session.get_inputs()[0].shape[-1]

    rng = np.random.default_rng(42)
    n = 100
    agree = 0
    for _ in range(n):
        x = rng.standard_normal((1, in_dim)).astype(np.float32)
        ort_out = ort_session.run(None, {in_name: x})[0]
        ort_am = int(np.argmax(ort_out[0]))
        wasm_am = _wasm_infer(wasm_path, x)
        if wasm_am == ort_am:
            agree += 1

    pct = agree / n
    assert pct >= 0.95, f"Agreement {agree}/{n} = {pct:.0%} < 95%"
