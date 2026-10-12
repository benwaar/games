"""Tests for the unified harness — no LLM required."""

import pytest
from pathlib import Path
import tempfile

from incant.harness import Z80Harness, WasmHarness
from incant.targets.z80 import assemble as z80_assemble
from incant.targets.wat import assemble as wat_assemble

# --- Z80 helpers ---

# "hi " + copy-input-to-output (the same fixture used in test_z80_gate.py)
_GREET_ASM = """\
INPUT_BUF  equ 0x8000
OUTPUT_BUF equ 0x9000
  ld de, OUTPUT_BUF
  ld a, 0x68
  ld (de), a
  inc de
  ld a, 0x69
  ld (de), a
  inc de
  ld a, 0x20
  ld (de), a
  inc de
  ld hl, INPUT_BUF
copy:
  ld a, (hl)
  or a
  jr z, done
  ld (de), a
  inc hl
  inc de
  jr copy
done:
  xor a
  ld (de), a
  halt
"""

# Minimal WASI WAT: echo stdin back to stdout
_ECHO_WAT = """\
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_read"
    (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (local $nread i32)
    (i32.store (i32.const 0) (i32.const 100))
    (i32.store (i32.const 4) (i32.const 200))
    (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
    (local.set $nread (i32.load (i32.const 8)))
    (i32.store (i32.const 0) (i32.const 100))
    (i32.store (i32.const 4) (local.get $nread))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))
  )
)
"""


@pytest.fixture(scope="module")
def greet_bin(tmp_path_factory):
    ok, data, err = z80_assemble(_GREET_ASM)
    assert ok, f"Z80 assembly failed: {err}"
    p = tmp_path_factory.mktemp("z80") / "greet.bin"
    p.write_bytes(data)
    return p


@pytest.fixture(scope="module")
def echo_wasm(tmp_path_factory):
    ok, wasm_path, err = wat_assemble(_ECHO_WAT)
    assert ok, f"WAT assembly failed: {err}"
    dest = tmp_path_factory.mktemp("wasm") / "echo.wasm"
    dest.write_bytes(wasm_path.read_bytes())
    return dest


# --- Z80Harness tests ---

def test_z80_harness_basic(greet_bin):
    h = Z80Harness(greet_bin)
    assert h.run("Ben") == "hi Ben"


def test_z80_harness_empty_input(greet_bin):
    h = Z80Harness(greet_bin)
    assert h.run("") == "hi "


def test_z80_harness_timeout():
    # Infinite loop — no HALT
    src = "loop:\njr loop"
    ok, data, _ = z80_assemble(src)
    assert ok
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(data)
        p = Path(f.name)
    try:
        h = Z80Harness(p, ticks=500)
        with pytest.raises(TimeoutError):
            h.run("")
    finally:
        p.unlink(missing_ok=True)


# --- WasmHarness tests ---

def test_wasm_harness_echo(echo_wasm):
    h = WasmHarness(echo_wasm)
    assert h.run("hello") == "hello"


def test_wasm_harness_empty_input(echo_wasm):
    h = WasmHarness(echo_wasm)
    assert h.run("") == ""


def test_wasm_harness_missing_file():
    with pytest.raises(FileNotFoundError):
        WasmHarness("/tmp/does_not_exist.wasm").run("")
