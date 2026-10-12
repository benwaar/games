"""Z80 inference engine code generator.

Generates Z80 assembly for a 2-layer ReLU network from quantised INT8
weights. No LLM involved — this is a deterministic template + data emitter.

Memory layout (default, for TinyNN 80→32→95):
  0x8000  input buffer   (input_dim bytes, INT8)
  0x8050  hidden buffer  (hidden_dim bytes, INT8)
  0x8070  output buffer  (output_dim bytes, INT8)
  0x80D0  fc1 weights    (output_dim_1 * input_dim bytes)
  ...
  fc1 bias, fc2 weights, fc2 bias follow contiguously

The generated code is self-contained: weights are emitted as defb blocks.
"""

from __future__ import annotations

import textwrap
from typing import Any

# Memory map base addresses
_INPUT_BASE = 0x8000
_SCRATCH_BASE = 0x8100   # work area above buffers; final buffers computed below


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-untyped]

        return np
    except ImportError as e:
        raise ImportError("z80_infer requires numpy: pip install numpy") from e


# ---------------------------------------------------------------------------
# INT8 Z80 matmul: shift-and-add signed multiplication
# ---------------------------------------------------------------------------

# We use a subroutine for signed 8×8→16 multiply.
# Z80 has no MUL instruction, so we implement via repeated addition.
# For speed on the test harness (not cycle-counting here), we use a simple
# approach: convert to unsigned, multiply unsigned with a shift-add loop,
# negate if signs differ.

_SIGNED_MUL_SUB = """\
; signed_mul: multiply A * B -> HL (signed 16-bit result)
; Inputs: A = multiplicand (int8), B = multiplier (int8)
; Output: HL = product (int16)
; Clobbers: A, B, C, D, E, H, L, F
signed_mul:
    ld hl, 0x0000
    ; save sign: D = sign_a XOR sign_b
    ld d, a
    xor b
    and 0x80
    ld e, a
    ; take absolute values
    ld a, d
    bit 7, a
    jr z, mul_a_pos
    neg
mul_a_pos:
    ld c, a
    ld a, b
    bit 7, a
    jr z, mul_b_pos
    neg
mul_b_pos:
    ld d, a
    ld a, c
    or a
    jr z, mul_done
mul_loop:
    ld a, l
    add a, d
    ld l, a
    jr nc, mul_nc
    inc h
mul_nc:
    dec c
    jr nz, mul_loop
mul_done:
    ld a, e
    or a
    ret z
    ; negate HL (two's complement)
    ld a, h
    cpl
    ld h, a
    ld a, l
    cpl
    ld l, a
    inc hl
    ret
"""

# ---------------------------------------------------------------------------
# Matrix multiply subroutine (generic, parametric via registers)
# ---------------------------------------------------------------------------

_MATMUL_SUB = """\
; matmul_layer: out_buf[j] = clamp(sum_i(in_buf[i] * weight[j][i]) + bias[j])
; Set before CALL:
;   IX = in_buf, IY = out_buf
;   (matmul_in_dim) = input size, (matmul_out_dim) = output size
;   (matmul_w_addr) = weight matrix address (2 bytes LE)
;   (matmul_b_addr) = bias array address   (2 bytes LE, int32 per entry)
matmul_layer:
    ld a, (matmul_out_dim)
    ld (matmul_j_count), a
    ld hl, (matmul_w_addr)
    ld de, (matmul_b_addr)
matmul_outer:
    ; load bias (int32 LE) into accumulator
    ld a, (de)
    ld (acc_b0), a
    inc de
    ld a, (de)
    ld (acc_b1), a
    inc de
    ld a, (de)
    ld (acc_b2), a
    inc de
    ld a, (de)
    ld (acc_b3), a
    inc de
    ld a, (matmul_in_dim)
    ld (matmul_i_count), a
    push ix
matmul_inner:
    ld a, (ix+0)
    ld b, (hl)
    inc hl
    push de
    push hl
    call signed_mul
    ; add 16-bit product into 32-bit accumulator
    ld a, (acc_b0)
    add a, l
    ld (acc_b0), a
    ld a, (acc_b1)
    adc a, h
    ld (acc_b1), a
    ; sign-extend the 16-bit product into bits 16..31
    ld a, h
    rlca
    sbc a, a
    ld d, a
    ld a, (acc_b2)
    adc a, d
    ld (acc_b2), a
    ld a, (acc_b3)
    adc a, d
    ld (acc_b3), a
    pop hl
    pop de
    inc ix
    ld a, (matmul_i_count)
    dec a
    ld (matmul_i_count), a
    jr nz, matmul_inner
    pop ix
    ; Use acc_b1 (byte 1 of int32 accumulator) as INT8 output.
    ; This is a >>8 shift — keeps the most significant 8 bits of the
    ; 16-bit low word, giving good dynamic range for INT8.
    ld a, (acc_b1)
    ld (iy+0), a
    inc iy
    ld a, (matmul_j_count)
    dec a
    ld (matmul_j_count), a
    jr nz, matmul_outer
    ret

; matmul_layer parameters (set before calling)
matmul_in_dim:
    db 0
matmul_out_dim:
    db 0
matmul_w_addr:
    dw 0
matmul_b_addr:
    dw 0
matmul_j_count:
    db 0
matmul_i_count:
    db 0
; 32-bit accumulator
acc_b0:
    db 0
acc_b1:
    db 0
acc_b2:
    db 0
acc_b3:
    db 0
"""

# ---------------------------------------------------------------------------
# ReLU subroutine
# ---------------------------------------------------------------------------

_RELU_SUB = """\
; relu_buf: zero out negative bytes in a buffer
; Inputs: IX = buffer address, B = buffer length
; Clobbers: A, B, IX
relu_buf:
    ld a, (ix+0)
    bit 7, a
    jr z, relu_pos
    xor a
    ld (ix+0), a
relu_pos:
    inc ix
    djnz relu_buf
    ret
"""

# ---------------------------------------------------------------------------
# Argmax subroutine
# ---------------------------------------------------------------------------

_ARGMAX_SUB = """\
; argmax_buf: find index of maximum INT8 (signed) value
; Inputs: IX = buffer address, B = buffer length (>= 1)
; Output: A = index of max, L = same
; Clobbers: A, B, C, D, E, H, L, IX
;
; Signed comparison on Z80:
;   after SUB C, A > C (signed) iff:
;     no overflow AND result positive (bit7=0), OR
;     overflow     AND result negative (bit7=1)
argmax_buf:
    ld c, (ix+0)
    ld d, 0
    ld e, 1
    inc ix
    dec b
    jr z, argmax_done
argmax_loop:
    ld a, (ix+0)
    sub c
    jr z, argmax_no_update
    ; signed comparison: A (before sub) > C?
    jp pe, argmax_overflow
    ; no overflow: positive result (bit7=0) means A > C
    bit 7, a
    jr z, argmax_update
    jr argmax_no_update
argmax_overflow:
    ; overflow: negative result (bit7=1) means A > C
    bit 7, a
    jr nz, argmax_update
    jr argmax_no_update
argmax_update:
    ld a, (ix+0)
    ld c, a
    ld d, e
argmax_no_update:
    inc ix
    inc e
    djnz argmax_loop
argmax_done:
    ld a, d
    ld l, a
    ret
"""

# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

_INFERENCE_TEMPLATE = """\
; inference: run a 2-layer ReLU network
; Input:  {in_dim} INT8 bytes at INPUT_BUF (0x{input_buf:04X})
; Output: result index written to RESULT_BYTE (0x{result_addr:04X})
;         also: A = argmax index after return
inference:
    ; --- Layer 1: FC1 matmul ---
    ld ix, {input_buf:#06x}    ; in_buf = INPUT_BUF
    ld iy, {hidden_buf:#06x}   ; out_buf = HIDDEN_BUF
    ld a, {in_dim}
    ld (matmul_in_dim), a
    ld a, {hidden_dim}
    ld (matmul_out_dim), a
    ld hl, fc1_weights
    ld (matmul_w_addr), hl
    ld hl, fc1_bias
    ld (matmul_b_addr), hl
    call matmul_layer

    ; --- ReLU on hidden buffer ---
    ld ix, {hidden_buf:#06x}
    ld b, {hidden_dim}
    call relu_buf

    ; --- Layer 2: FC2 matmul ---
    ld ix, {hidden_buf:#06x}   ; in_buf = HIDDEN_BUF
    ld iy, {output_buf:#06x}   ; out_buf = OUTPUT_BUF
    ld a, {hidden_dim}
    ld (matmul_in_dim), a
    ld a, {out_dim}
    ld (matmul_out_dim), a
    ld hl, fc2_weights
    ld (matmul_w_addr), hl
    ld hl, fc2_bias
    ld (matmul_b_addr), hl
    call matmul_layer

    ; --- Argmax on output buffer ---
    ld ix, {output_buf:#06x}
    ld b, {out_dim}
    call argmax_buf

    ; --- Store result ---
    ld ({result_addr:#06x}), a
    halt
"""


# ---------------------------------------------------------------------------
# Weight / bias emitter
# ---------------------------------------------------------------------------


def _emit_defb(label: str, data: list[int], cols: int = 16) -> str:
    """Emit a labelled db block. Label is on its own line (z80 assembler requirement).

    Negative INT8 values are emitted as unsigned bytes (e.g. -1 → 255).
    """
    lines = [f"{label}:"]
    for i in range(0, len(data), cols):
        chunk = data[i : i + cols]
        vals = ", ".join(str(int(v) & 0xFF) for v in chunk)
        lines.append(f"    db {vals}")
    return "\n".join(lines)


def _emit_defw_bias(label: str, data: list[int], cols: int = 8) -> str:
    """Emit bias as INT32 (4 bytes LE each) db block."""
    lines = [f"{label}:"]
    for val in data:
        v = int(val) & 0xFFFFFFFF
        b0 = v & 0xFF
        b1 = (v >> 8) & 0xFF
        b2 = (v >> 16) & 0xFF
        b3 = (v >> 24) & 0xFF
        lines.append(f"    db {b0}, {b1}, {b2}, {b3}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_z80_inference(quantised: dict) -> str:
    """Generate a complete Z80 assembly source for a 2-layer ReLU network.

    Args:
        quantised: output of ``quantise_model`` — layers with int8 weights,
                   int32 biases, and scale factors.

    Returns:
        A string containing valid Z80 assembly (for the z80 pip package).
    """
    np = _numpy()

    # Extract the two linear layers (skip relu layers for indexing)
    linear_layers = [l for l in quantised["layers"] if l["type"] == "linear"]
    if len(linear_layers) != 2:
        raise ValueError(
            f"generate_z80_inference expects exactly 2 linear layers, "
            f"got {len(linear_layers)}"
        )

    fc1, fc2 = linear_layers
    in_dim = fc1["in"]
    hidden_dim = fc1["out"]
    out_dim = fc2["out"]

    # Memory layout
    input_buf = _INPUT_BASE
    hidden_buf = input_buf + in_dim
    # pad to 16-byte boundary
    hidden_buf = (hidden_buf + 15) & ~15
    output_buf = hidden_buf + hidden_dim
    output_buf = (output_buf + 15) & ~15
    result_addr = output_buf + out_dim
    result_addr = (result_addr + 1) & ~1  # word-align

    # Build weight data (int8 → Python int list)
    w1_flat = fc1["weight"].flatten().tolist()
    b1_flat = fc1["bias"].flatten().tolist()
    w2_flat = fc2["weight"].flatten().tolist()
    b2_flat = fc2["bias"].flatten().tolist()

    parts: list[str] = []

    # Header comment
    parts.append(
        f"; Z80 inference engine — {in_dim}→{hidden_dim}→{out_dim} ReLU network\n"
        f"; Generated by incant M10 — do not edit by hand\n"
        f";\n"
        f"; Memory map:\n"
        f";   0x{input_buf:04X}  input buffer  ({in_dim} bytes)\n"
        f";   0x{hidden_buf:04X}  hidden buffer ({hidden_dim} bytes)\n"
        f";   0x{output_buf:04X}  output buffer ({out_dim} bytes)\n"
        f";   result byte at 0x{result_addr:04X}\n"
    )

    # Address constants
    parts.append(
        f"input_buf  equ 0x{input_buf:04X}\n"
        f"hidden_buf equ 0x{hidden_buf:04X}\n"
        f"output_buf equ 0x{output_buf:04X}\n"
        f"result     equ 0x{result_addr:04X}\n"
    )

    # Entry point: jump to inference
    parts.append("    jp inference\n")

    # Subroutines
    parts.append(_SIGNED_MUL_SUB)
    parts.append(_MATMUL_SUB)
    parts.append(_RELU_SUB)
    parts.append(_ARGMAX_SUB)

    # Main inference routine
    parts.append(
        _INFERENCE_TEMPLATE.format(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            input_buf=input_buf,
            hidden_buf=hidden_buf,
            output_buf=output_buf,
            result_addr=result_addr,
        )
    )

    # Weight data blocks
    parts.append(_emit_defb("fc1_weights", [int(v) for v in w1_flat]))
    parts.append("")
    parts.append(_emit_defw_bias("fc1_bias", [int(v) for v in b1_flat]))
    parts.append("")
    parts.append(_emit_defb("fc2_weights", [int(v) for v in w2_flat]))
    parts.append("")
    parts.append(_emit_defw_bias("fc2_bias", [int(v) for v in b2_flat]))
    parts.append("")

    return "\n".join(parts)


def run_z80_inference(
    quantised: dict,
    x_int8_bytes: bytes,
    ticks: int = 10_000_000,
) -> int:
    """Assemble and run the Z80 inference engine on a single input.

    Args:
        quantised:    output of quantise_model.
        x_int8_bytes: input as raw INT8 bytes (length == in_dim).
        ticks:        Z80 tick budget (default 10M — enough for small models).

    Returns:
        Argmax index from the Z80 output buffer.
    """
    import z80 as z80lib

    from .targets.z80 import assemble as z80_assemble

    asm_source = generate_z80_inference(quantised)
    ok, binary, err = z80_assemble(asm_source)
    if not ok:
        raise RuntimeError(f"Z80 assembly failed: {err}")

    linear_layers = [l for l in quantised["layers"] if l["type"] == "linear"]
    in_dim = linear_layers[0]["in"]
    out_dim = linear_layers[-1]["out"]

    input_buf = _INPUT_BASE
    hidden_dim = linear_layers[0]["out"]
    hidden_buf = (input_buf + in_dim + 15) & ~15
    output_buf = (hidden_buf + hidden_dim + 15) & ~15
    result_addr = (output_buf + out_dim + 1) & ~1

    machine = z80lib.Z80Machine()
    machine.set_memory_block(0, binary)
    machine.set_memory_block(input_buf, x_int8_bytes)
    machine.ticks_to_stop = ticks
    machine.run()

    if not machine.halted:
        raise TimeoutError(f"Z80 inference did not halt within {ticks} ticks")

    # Read result byte
    return int(machine.memory[result_addr])
