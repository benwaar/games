"""Tests for WAT target — assemble and validate known-good WAT."""

import shutil

import pytest

from incant.targets.wat import assemble, extract_code


@pytest.fixture
def has_wabt():
    if not shutil.which("wat2wasm"):
        pytest.skip("wabt not installed")


def test_assemble_add(has_wabt):
    source = """(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  (export "add" (func $add))
)"""
    ok, wasm_path, err = assemble(source)
    assert ok, f"Assembly failed: {err}"
    assert wasm_path is not None
    assert wasm_path.exists()
    assert wasm_path.stat().st_size > 0


def test_assemble_invalid(has_wabt):
    source = "(module (func INVALID))"
    ok, _, err = assemble(source)
    assert not ok
    assert err is not None


def test_extract_code_with_fences():
    raw = """Here's the WAT:
```wat
(module
  (func $x (result i32) i32.const 1)
  (export "x" (func $x))
)
```"""
    code = extract_code(raw)
    assert code.startswith("(module")
    assert code.endswith(")")


def test_extract_code_module_detection():
    raw = """Some explanation before.
(module
  (func $y (result i32) i32.const 42)
  (export "y" (func $y))
)
Some text after."""
    code = extract_code(raw)
    assert code.startswith("(module")
    assert code.endswith(")")
    assert "Some explanation" not in code
