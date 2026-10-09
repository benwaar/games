"""Tests for WAT target — assemble and validate known-good WAT."""

import shutil

import pytest

from incant.sigil import Sigil, SigilParam, SigilSignature, SigilTest
from incant.targets.wat import assemble, extract_code, run_test, run_wasi_test


@pytest.fixture
def has_wabt():
    if not shutil.which("wat2wasm") or not shutil.which("wasm-interp"):
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


ADD_WAT = """(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  (export "add" (func $add))
)"""


def _make_wat_sigil(inputs, output_type="i32", export="add", tests=None, memory=None):
    return Sigil(
        name="test",
        description="test",
        target="wat",
        signature=SigilSignature(
            inputs=inputs,
            output_type=output_type,
        ),
        tests=tests or [],
        export=export,
        memory=memory,
    )


def test_run_test_add(has_wabt):
    ok, wasm_path, err = assemble(ADD_WAT)
    assert ok, err

    sigil = _make_wat_sigil(
        inputs=[
            SigilParam(name="a", type="i32"),
            SigilParam(name="b", type="i32"),
        ],
        export="add",
    )
    test = SigilTest(inputs={"a": 2, "b": 3}, expect=5)
    passed, msg = run_test(wasm_path, test, sigil)
    assert passed, msg


def test_run_test_negative(has_wabt):
    ok, wasm_path, err = assemble(ADD_WAT)
    assert ok, err

    sigil = _make_wat_sigil(
        inputs=[
            SigilParam(name="a", type="i32"),
            SigilParam(name="b", type="i32"),
        ],
        export="add",
    )
    test = SigilTest(inputs={"a": -1, "b": 1}, expect=0)
    passed, msg = run_test(wasm_path, test, sigil)
    assert passed, msg


def test_run_test_wrong_result(has_wabt):
    ok, wasm_path, err = assemble(ADD_WAT)
    assert ok, err

    sigil = _make_wat_sigil(
        inputs=[
            SigilParam(name="a", type="i32"),
            SigilParam(name="b", type="i32"),
        ],
        export="add",
    )
    test = SigilTest(inputs={"a": 2, "b": 3}, expect=99)
    passed, msg = run_test(wasm_path, test, sigil)
    assert not passed
    assert "Expected 99" in msg


WASI_HELLO_WAT = """(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 8) "hello world\\n")
  (data (i32.const 0) "\\08\\00\\00\\00\\0c\\00\\00\\00")
  (func (export "_start")
    (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20))
    drop
  )
)"""


def test_wasi_assemble_and_run(has_wabt):
    ok, wasm_path, err = assemble(WASI_HELLO_WAT)
    assert ok, f"Assembly failed: {err}"
    passed, msg = run_wasi_test(wasm_path, "", "hello world\n")
    assert passed, msg


def test_wasi_wrong_output(has_wabt):
    ok, wasm_path, err = assemble(WASI_HELLO_WAT)
    assert ok, err
    passed, msg = run_wasi_test(wasm_path, "", "wrong output")
    assert not passed
    assert "Expected stdout" in msg
