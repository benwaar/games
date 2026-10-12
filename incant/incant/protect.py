"""Binary protection — self-verifying code wrappers for Z80 and WASM."""

from __future__ import annotations

import subprocess
import tempfile
from functools import reduce
from operator import xor
from pathlib import Path


VERIFY_PREFIX_LEN = 26
HALT_OPCODE = 0x76
CHECKSUM_OFFSET = 19  # byte offset of the expected checksum in the prefix
JP_NZ_OFFSET = 20     # byte offset of the JP NZ instruction in the prefix


def _lo(val: int) -> int:
    return val & 0xFF


def _hi(val: int) -> int:
    return (val >> 8) & 0xFF


def _xor_checksum(data: bytes) -> int:
    return reduce(xor, data, 0)


def wrap_z80_checksum(payload: bytes, load_addr: int = 0x8000) -> bytes:
    """Wrap Z80 binary with an XOR checksum verifier prefix.

    The returned binary has this layout at `load_addr`:
        [26-byte verifier prefix] [payload] [HALT (crash target)]

    The verifier uses register E as an XOR accumulator over every byte
    of [payload], then compares against a patched-in expected value.
    If intact → JP to payload.  If tampered → JP to the trailing HALT.

    Returns the complete binary ready to load at `load_addr`.
    """
    payload_start = load_addr + VERIFY_PREFIX_LEN
    payload_len = len(payload)
    crash_addr = payload_start + payload_len

    expected_checksum = _xor_checksum(payload)

    prefix = bytes([
        0x21, _lo(payload_start), _hi(payload_start),  # LD HL, payload_start
        0x01, _lo(payload_len), _hi(payload_len),       # LD BC, payload_len
        0x1E, 0x00,                                      # LD E, 0
        0x7E,                                            # LD A, (HL)
        0xAB,                                            # XOR E
        0x5F,                                            # LD E, A
        0x23,                                            # INC HL
        0x0B,                                            # DEC BC
        0x78,                                            # LD A, B
        0xB1,                                            # OR C
        0x20, 0xF7,                                      # JR NZ, -9 (back to LD A,(HL))
        0x7B,                                            # LD A, E
        0xFE, expected_checksum,                         # CP expected
        0xC2, _lo(crash_addr), _hi(crash_addr),          # JP NZ, crash
        0xC3, _lo(payload_start), _hi(payload_start),    # JP payload_start
    ])

    assert len(prefix) == VERIFY_PREFIX_LEN
    return prefix + payload + bytes([HALT_OPCODE])


def tamper_byte(protected: bytes, offset: int) -> bytes:
    """Flip one byte in a protected binary (for testing)."""
    flipped = bytearray(protected)
    flipped[offset] ^= 0x01
    return bytes(flipped)


def bypass_nop_branch(protected: bytes) -> bytes:
    """Attacker bypass: NOP out the JP NZ instruction (3 bytes at JP_NZ_OFFSET)."""
    patched = bytearray(protected)
    patched[JP_NZ_OFFSET] = 0x00      # NOP
    patched[JP_NZ_OFFSET + 1] = 0x00  # NOP
    patched[JP_NZ_OFFSET + 2] = 0x00  # NOP
    return bytes(patched)


def bypass_fix_checksum(protected: bytes) -> bytes:
    """Attacker bypass: recompute the checksum after tampering."""
    patched = bytearray(protected)
    payload = patched[VERIFY_PREFIX_LEN:-1]
    patched[CHECKSUM_OFFSET] = _xor_checksum(payload)
    return bytes(patched)


# ---------------------------------------------------------------------------
# WASM self-verifying module
# ---------------------------------------------------------------------------

def generate_self_verify_wat(
    payload_bytes: bytes,
    expected_result: int = 42,
) -> str:
    """Generate a WAT module that verifies a data segment before running.

    The module stores `payload_bytes` in a data segment at offset 1024.
    At startup, it XOR-hashes that region and compares to the expected
    checksum.  If valid → returns `expected_result`.  If tampered → unreachable trap.

    The exported function is `run` with signature () -> i32.
    """
    data_offset = 1024
    data_len = len(payload_bytes)
    expected_checksum = _xor_checksum(payload_bytes)

    hex_payload = "".join(f"\\{b:02x}" for b in payload_bytes)

    return f"""\
(module
  (memory (export "memory") 1)
  (data (i32.const {data_offset}) "{hex_payload}")

  (func $verify (result i32)
    (local $i i32)
    (local $end i32)
    (local $acc i32)
    (local.set $i (i32.const {data_offset}))
    (local.set $end (i32.const {data_offset + data_len}))
    (local.set $acc (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $end)))
        (local.set $acc
          (i32.xor (local.get $acc) (i32.load8_u (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (local.get $acc)
  )

  (func (export "run") (result i32)
    (if (result i32) (i32.ne (call $verify) (i32.const {expected_checksum}))
      (then (unreachable))
      (else (i32.const {expected_result}))
    )
  )
)"""


def assemble_wat(wat_source: str) -> Path:
    """Assemble WAT to WASM using wat2wasm. Returns path to .wasm file."""
    tmp_dir = tempfile.mkdtemp(prefix="incant_protect_")
    wat_path = Path(tmp_dir) / "verify.wat"
    wat_path.write_text(wat_source)
    wasm_path = wat_path.with_suffix(".wasm")

    result = subprocess.run(
        ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wat2wasm failed: {result.stderr.strip()}")
    return wasm_path


def run_wasm_module(wasm_path: Path) -> int:
    """Run the exported `run` function and return its i32 result."""
    result = subprocess.run(
        ["wasm-interp", str(wasm_path), "-r", "run"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    output = result.stdout.strip()
    if "unreachable" in output.lower():
        raise RuntimeError("unreachable executed")
    if "=>" not in output:
        raise RuntimeError(f"Unexpected output: {output}")
    val_str = output.split("=>")[-1].strip()
    if ":" in val_str:
        val_str = val_str.split(":")[-1]
    return int(val_str)


def tamper_wasm_data_segment(wasm_bytes: bytes, payload_bytes: bytes) -> bytes:
    """Flip one byte in the data segment of a compiled WASM binary.

    Searches for the payload bytes in the binary and flips the first byte.
    """
    patched = bytearray(wasm_bytes)
    idx = patched.find(payload_bytes)
    if idx < 0:
        raise ValueError("Payload bytes not found in WASM binary")
    patched[idx] ^= 0x01
    return bytes(patched)
