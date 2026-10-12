"""INT8 quantisation for neural network weights.

Per-layer symmetric quantisation:
  scale = max(|w|) / 127
  w_int8 = clamp(round(w / scale), -128, 127)

The same approach is used for biases, accumulated with the input scale.

Also provides a Python reference forward pass (INT8 arithmetic) for
comparing against the float32 onnxruntime output.
"""

from __future__ import annotations

import copy
from typing import Any


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-untyped]

        return np
    except ImportError as e:
        raise ImportError("quantise requires numpy: pip install numpy") from e


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------


def _quant_symmetric(arr: Any, np: Any) -> tuple[Any, float]:
    """Quantise a float32 array to INT8 with per-tensor symmetric scaling.

    Returns (int8_array, scale) where scale = max(|arr|) / 127.
    """
    abs_max = float(np.max(np.abs(arr)))
    if abs_max == 0.0:
        scale = 1.0
    else:
        scale = abs_max / 127.0

    quantised = np.round(arr / scale).astype(np.int32)
    quantised = np.clip(quantised, -128, 127).astype(np.int8)
    return quantised, scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantise_model(parsed: dict) -> dict:
    """Quantise float32 weights to INT8 with per-layer scale factors.

    Input: a dict as returned by ``parse_onnx``.
    Output: same structure but each linear layer has:
        "weight":       int8 numpy array (out × in)
        "bias":         int32 numpy array (out,)   — scaled by scale_w * scale_in
        "scale_w":      float  — weight quantisation scale
        "scale_in":     float  — assumed input scale (1/127 for normalised inputs)
        "scale_out":    float  — output scale (scale_w * scale_in)

    ReLU layers are passed through unchanged.
    """
    np = _numpy()
    result = copy.deepcopy(parsed)

    # We propagate the accumulated output scale forward as the input scale
    # for the next layer.  For the first layer we assume inputs are
    # normalised to [-1, 1] → scale_in ≈ 1/127.
    scale_input = 1.0 / 127.0

    for layer in result["layers"]:
        if layer["type"] != "linear":
            continue

        weight = layer["weight"].astype(np.float32)
        bias = layer["bias"].astype(np.float32)

        w_int8, scale_w = _quant_symmetric(weight, np)

        # Bias scale = scale_w * scale_input so that
        #   acc (int32) = sum(x_int8 * w_int8) + bias_int32
        # reproduces the float computation up to quantisation error.
        bias_scale = scale_w * scale_input
        if bias_scale == 0.0:
            bias_scale = 1e-9
        bias_quant = np.round(bias / bias_scale).astype(np.int64)
        bias_quant = np.clip(bias_quant, -(2**31), 2**31 - 1).astype(np.int32)

        scale_output = scale_w * scale_input

        layer["weight"] = w_int8
        layer["bias"] = bias_quant
        layer["scale_w"] = scale_w
        layer["scale_in"] = scale_input
        layer["scale_out"] = scale_output

        # The Z80 matmul takes acc_b1 (byte 1 of int32 accumulator) as output.
        # That is a right-shift by 8, so the effective scale of the output bytes
        # is scale_output * 256.  Propagate that as the input scale for the
        # next layer so its bias is computed at the correct magnitude.
        scale_input = scale_output * 256

    return result


# ---------------------------------------------------------------------------
# Reference INT8 forward pass (Python, not Z80)
# ---------------------------------------------------------------------------


def int8_forward(quantised: dict, x_float: Any) -> tuple[Any, int]:
    """Run a reference INT8 forward pass on float32 input.

    Dequantises INT8 weights back to approximate float32, runs inference,
    and returns (output_float32_approx, argmax_index).

    The scale factors account for the >>8 shift applied when the Z80 stores
    acc_b1 (byte 1 of int32 accumulator) as the inter-layer activation.
    """
    np = _numpy()

    x = x_float.astype(np.float32).flatten()

    for layer in quantised["layers"]:
        if layer["type"] == "relu":
            x = np.maximum(x, 0.0).astype(np.float32)
            continue

        if layer["type"] != "linear":
            continue

        # Dequantise weights: w_float ≈ w_int8 * scale_w
        w_float = layer["weight"].astype(np.float32) * layer["scale_w"]
        # Bias scale: scale_w * scale_in, where scale_in already incorporates
        # the *256 factor from the previous layer's >>8 output shift.
        b_float = layer["bias"].astype(np.float32) * (layer["scale_w"] * layer["scale_in"])

        x = (w_float @ x + b_float).astype(np.float32)

    return x, int(np.argmax(x))


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------


def compare_accuracy(
    parsed_float: dict,
    quantised: dict,
    onnx_path: str,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """Compare INT8 Python reference against float32 onnxruntime on random inputs.

    Returns {
        "n_samples": int,
        "argmax_agreement": float,   # fraction where argmax matches
        "mean_l1_error":    float,   # mean L1 distance of output vectors
    }
    """
    np = _numpy()
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("compare_accuracy requires onnxruntime: pip install onnxruntime") from e

    rng = np.random.default_rng(seed)
    input_dim = parsed_float["input_shape"][-1] if parsed_float["input_shape"] else 4

    sess = ort.InferenceSession(onnx_path)
    in_name = sess.get_inputs()[0].name

    agree = 0
    total_l1 = 0.0

    for _ in range(n_samples):
        x = rng.uniform(-1.0, 1.0, size=(1, input_dim)).astype(np.float32)
        # Float32 reference
        ref_out = sess.run(None, {in_name: x})[0].flatten()
        ref_argmax = int(np.argmax(ref_out))

        # INT8 reference
        _out_float, q_argmax = int8_forward(quantised, x)
        agree += int(q_argmax == ref_argmax)
        total_l1 += float(np.mean(np.abs(ref_out / (np.max(np.abs(ref_out)) + 1e-9) - _out_float)))

    return {
        "n_samples": n_samples,
        "argmax_agreement": agree / n_samples,
        "mean_l1_error": total_l1 / n_samples,
    }
