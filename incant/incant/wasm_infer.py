"""WASM inference engine code generator.

Generates a self-contained WAT module for a 2-layer ReLU network from
float32 weights. No LLM involved — deterministic template + data emitter,
parallel to z80_infer.py.

Memory layout (for a network with in_dim inputs, hid_dim hidden, out_dim outputs):
  offset     0: input   (in_dim  × 4 bytes, f32)
  offset   I_E: hidden  (hid_dim × 4 bytes, f32)  where I_E = in_dim * 4
  offset   H_E: output  (out_dim × 4 bytes, f32)  where H_E = I_E + hid_dim * 4
  offset   O_E: fc1 weights  (in_dim  × hid_dim × 4 bytes)
  offset   W1E: fc1 bias     (hid_dim × 4 bytes)
  offset   B1E: fc2 weights  (hid_dim × out_dim × 4 bytes)
  offset   W2E: fc2 bias     (out_dim × 4 bytes)

All fits in 1 WASM page (64 KB) for networks up to ~TinyNN scale (80→32→95).
"""

from __future__ import annotations

import struct
from typing import Any


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-untyped]

        return np
    except ImportError as e:
        raise ImportError("wasm_infer requires numpy: pip install numpy") from e


# ---------------------------------------------------------------------------
# Raw f32 bytes → WAT string literal (e.g. \xx per byte)
# ---------------------------------------------------------------------------


def _f32_to_wat_string(values: list[float]) -> str:
    """Pack f32 list as little-endian bytes and escape for WAT data segment."""
    raw = struct.pack("<" + "f" * len(values), *values)
    return "".join(f"\\{b:02x}" for b in raw)


# ---------------------------------------------------------------------------
# WAT module template pieces
# ---------------------------------------------------------------------------

_FC_FUNC = """\
  ;; $fc: generic dense layer
  ;; in_ptr, out_ptr, w_ptr, b_ptr: byte offsets into linear memory
  ;; in_dim, out_dim: number of neurons
  (func $fc
        (param $in_ptr i32) (param $out_ptr i32)
        (param $w_ptr  i32) (param $b_ptr  i32)
        (param $in_dim i32) (param $out_dim i32)
    (local $j i32)
    (local $i i32)
    (local $acc f32)
    (local $w_row i32)
    (local.set $j (i32.const 0))
    (block $break_outer
      (loop $outer
        (br_if $break_outer (i32.ge_u (local.get $j) (local.get $out_dim)))
        ;; acc = bias[j]
        (local.set $acc
          (f32.load
            (i32.add (local.get $b_ptr)
                     (i32.mul (local.get $j) (i32.const 4)))))
        ;; w_row = w_ptr + j * in_dim * 4
        (local.set $w_row
          (i32.add (local.get $w_ptr)
                   (i32.mul (local.get $j)
                            (i32.mul (local.get $in_dim) (i32.const 4)))))
        (local.set $i (i32.const 0))
        (block $break_inner
          (loop $inner
            (br_if $break_inner (i32.ge_u (local.get $i) (local.get $in_dim)))
            ;; acc += in[i] * w[j * in_dim + i]
            (local.set $acc
              (f32.add (local.get $acc)
                (f32.mul
                  (f32.load
                    (i32.add (local.get $in_ptr)
                             (i32.mul (local.get $i) (i32.const 4))))
                  (f32.load
                    (i32.add (local.get $w_row)
                             (i32.mul (local.get $i) (i32.const 4)))))))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $inner)))
        ;; out[j] = acc
        (f32.store
          (i32.add (local.get $out_ptr)
                   (i32.mul (local.get $j) (i32.const 4)))
          (local.get $acc))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $outer)))
  )
"""

_RELU_FUNC = """\
  ;; $relu_buf: apply ReLU in-place to a buffer of f32
  (func $relu_buf (param $ptr i32) (param $len i32)
    (local $i i32)
    (local.set $i (i32.const 0))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))
        (f32.store
          (i32.add (local.get $ptr) (i32.mul (local.get $i) (i32.const 4)))
          (f32.max
            (f32.load
              (i32.add (local.get $ptr) (i32.mul (local.get $i) (i32.const 4))))
            (f32.const 0.0)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
  )
"""

_ARGMAX_FUNC = """\
  ;; $argmax: return index of maximum f32 in buffer; len must be >= 1
  (func $argmax (param $ptr i32) (param $len i32) (result i32)
    (local $i i32)
    (local $best_idx i32)
    (local $best_val f32)
    (local $val f32)
    (local.set $best_idx (i32.const 0))
    (local.set $best_val (f32.load (local.get $ptr)))
    (local.set $i (i32.const 1))
    (block $break
      (loop $loop
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))
        (local.set $val
          (f32.load
            (i32.add (local.get $ptr) (i32.mul (local.get $i) (i32.const 4)))))
        (if (f32.gt (local.get $val) (local.get $best_val))
          (then
            (local.set $best_val (local.get $val))
            (local.set $best_idx (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
    (local.get $best_idx)
  )
"""

_INFER_FUNC_TEMPLATE = """\
  ;; infer: run the full 2-layer network; caller writes f32 input at $in_ptr
  ;; returns argmax class index (i32)
  (func (export "infer") (param $in_ptr i32) (result i32)
    ;; layer 1
    (call $fc
      (local.get $in_ptr) (i32.const {hidden_ptr})
      (i32.const {w1_ptr}) (i32.const {b1_ptr})
      (i32.const {in_dim}) (i32.const {hid_dim}))
    ;; relu on hidden buffer
    (call $relu_buf (i32.const {hidden_ptr}) (i32.const {hid_dim}))
    ;; layer 2
    (call $fc
      (i32.const {hidden_ptr}) (i32.const {out_ptr})
      (i32.const {w2_ptr}) (i32.const {b2_ptr})
      (i32.const {hid_dim}) (i32.const {out_dim}))
    ;; argmax
    (call $argmax (i32.const {out_ptr}) (i32.const {out_dim}))
  )
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_wat_inference(parsed: dict) -> str:
    """Emit a self-contained WAT module from an ONNX parse result.

    Supports: any 2-layer ReLU network (linear → relu → linear).
    No LLM. Deterministic template + data emitter, like z80_infer.py.
    Returns WAT source as a string.
    """
    np = _numpy()

    linear_layers = [layer for layer in parsed["layers"] if layer["type"] == "linear"]
    if len(linear_layers) != 2:
        raise ValueError(
            f"generate_wat_inference expects exactly 2 linear layers, got {len(linear_layers)}"
        )

    fc1, fc2 = linear_layers
    in_dim: int = fc1["in"]
    hid_dim: int = fc1["out"]
    out_dim: int = fc2["out"]

    # Memory layout (all offsets in bytes)
    hidden_ptr = in_dim * 4
    out_ptr = hidden_ptr + hid_dim * 4
    w1_ptr = out_ptr + out_dim * 4
    b1_ptr = w1_ptr + in_dim * hid_dim * 4
    w2_ptr = b1_ptr + hid_dim * 4
    b2_ptr = w2_ptr + hid_dim * out_dim * 4
    end_ptr = b2_ptr + out_dim * 4

    pages = max(1, (end_ptr + 65535) // 65536)

    # Flatten weights + biases to Python float lists
    w1_flat = fc1["weight"].flatten().astype(np.float32).tolist()
    b1_flat = fc1["bias"].flatten().astype(np.float32).tolist()
    w2_flat = fc2["weight"].flatten().astype(np.float32).tolist()
    b2_flat = fc2["bias"].flatten().astype(np.float32).tolist()

    w1_str = _f32_to_wat_string(w1_flat)
    b1_str = _f32_to_wat_string(b1_flat)
    w2_str = _f32_to_wat_string(w2_flat)
    b2_str = _f32_to_wat_string(b2_flat)

    lines: list[str] = [
        f";; WAT inference engine — {in_dim}→{hid_dim}→{out_dim} ReLU network",
        f";; Generated by incant wasm_infer — do not edit by hand",
        f";;",
        f";; Memory layout (offsets in bytes):",
        f";;   {0:6d}: input   ({in_dim} × f32)",
        f";;   {hidden_ptr:6d}: hidden  ({hid_dim} × f32)",
        f";;   {out_ptr:6d}: output  ({out_dim} × f32)",
        f";;   {w1_ptr:6d}: fc1 W   ({in_dim}×{hid_dim} × f32)",
        f";;   {b1_ptr:6d}: fc1 b   ({hid_dim} × f32)",
        f";;   {w2_ptr:6d}: fc2 W   ({hid_dim}×{out_dim} × f32)",
        f";;   {b2_ptr:6d}: fc2 b   ({out_dim} × f32)",
        f";;   {end_ptr:6d}: end     ({pages} page(s))",
        "(module",
        f"  (memory (export \"memory\") {pages})",
        f'  (data (i32.const {w1_ptr}) "{w1_str}")',
        f'  (data (i32.const {b1_ptr}) "{b1_str}")',
        f'  (data (i32.const {w2_ptr}) "{w2_str}")',
        f'  (data (i32.const {b2_ptr}) "{b2_str}")',
    ]

    lines.append(_FC_FUNC.rstrip())
    lines.append(_RELU_FUNC.rstrip())
    lines.append(_ARGMAX_FUNC.rstrip())
    lines.append(
        _INFER_FUNC_TEMPLATE.format(
            in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
            hidden_ptr=hidden_ptr, out_ptr=out_ptr,
            w1_ptr=w1_ptr, b1_ptr=b1_ptr,
            w2_ptr=w2_ptr, b2_ptr=b2_ptr,
        ).rstrip()
    )
    lines.append(")")

    return "\n".join(lines) + "\n"
