"""Tests for WAT and Z80 stitching."""

import shutil

import pytest

from incant.stitch import stitch_wat, stitch_z80
from incant.targets.wat import assemble as wat_assemble


ADD_WAT = """(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  (export "add" (func $add))
)"""

DOUBLE_WAT = """(module
  (func $double (param $x i32) (result i32)
    local.get $x
    i32.const 2
    i32.mul
  )
  (export "double" (func $double))
)"""

MEMORY_WAT = """(module
  (memory 1)
  (export "memory" (memory 0))
  (func $store (param $v i32) (result i32)
    i32.const 0
    local.get $v
    i32.store
    i32.const 0
    i32.load
  )
  (export "store" (func $store))
)"""


@pytest.fixture
def has_wabt():
    if not shutil.which("wat2wasm") or not shutil.which("wasm-interp"):
        pytest.skip("wabt not installed")


class TestStitchWat:
    def test_two_modules(self):
        result = stitch_wat([ADD_WAT, DOUBLE_WAT])
        assert "(module" in result
        assert "$add" in result
        assert "$double" in result
        assert '"add"' in result
        assert '"double"' in result

    def test_dedup_memory(self):
        result = stitch_wat([ADD_WAT, MEMORY_WAT])
        # one memory declaration, one memory export
        lines = [l.strip() for l in result.splitlines()]
        mem_decls = [l for l in lines if l.startswith("(memory")]
        assert len(mem_decls) == 1
        assert "(memory 1)" in mem_decls[0]

    def test_assembled_combined_runs(self, has_wabt):
        combined = stitch_wat([ADD_WAT, DOUBLE_WAT])
        ok, wasm_path, err = wat_assemble(combined)
        assert ok, f"Combined assembly failed: {err}"

        import subprocess
        result = subprocess.run(
            ["wasm-interp", str(wasm_path), "-r", "add", "-a", "i32:2", "-a", "i32:3"],
            capture_output=True, text=True,
        )
        assert "i32:5" in result.stdout

        result = subprocess.run(
            ["wasm-interp", str(wasm_path), "-r", "double", "-a", "i32:7"],
            capture_output=True, text=True,
        )
        assert "i32:14" in result.stdout

    def test_single_module(self):
        result = stitch_wat([ADD_WAT])
        assert "$add" in result
        assert "(module" in result


class TestStitchZ80:
    def test_two_sources(self):
        s1 = "ld a, 0\nhalt"
        s2 = "add a, b\nhalt"
        result = stitch_z80([s1, s2], ["init", "add"])
        assert "; --- init ---" in result
        assert "; --- add ---" in result
        # first halt removed, second kept
        lines = [l.strip().lower() for l in result.splitlines() if l.strip()]
        halts = [l for l in lines if l == "halt"]
        assert len(halts) == 1

    def test_single_source(self):
        s = "add a, b\nhalt"
        result = stitch_z80([s], ["add"])
        assert "halt" in result.lower()

    def test_preserves_labels(self):
        s1 = "loop:\n  inc a\n  djnz loop\nhalt"
        s2 = "ld a, 0\nhalt"
        result = stitch_z80([s1, s2], ["counter", "reset"])
        assert "loop:" in result
        assert "ld a, 0" in result
