"""Tests for Z80 target — assemble and run known-good code."""

from incant.sigil import Sigil, SigilParam, SigilSignature, SigilTest
from incant.targets.z80 import assemble, run_test, extract_code


def _make_sigil(inputs, output_reg="A", tests=None):
    return Sigil(
        name="test",
        description="test",
        target="z80",
        signature=SigilSignature(
            inputs=inputs,
            output_type="u8",
            output_register=output_reg,
        ),
        tests=tests or [],
    )


def test_assemble_add():
    source = "add a, b\nhalt"
    ok, data, err = assemble(source)
    assert ok, f"Assembly failed: {err}"
    assert data is not None
    assert len(data) > 0


def test_assemble_invalid():
    source = "INVALID_INSTRUCTION"
    ok, data, err = assemble(source)
    assert not ok
    assert err is not None


def test_run_add():
    source = "add a, b\nhalt"
    ok, data, _ = assemble(source)
    assert ok

    sigil = _make_sigil(
        inputs=[
            SigilParam(name="a", type="u8", register="A"),
            SigilParam(name="b", type="u8", register="B"),
        ],
    )
    test = SigilTest(inputs={"a": 2, "b": 3}, expect={"a": 5})
    passed, msg = run_test(data, test, sigil)
    assert passed, msg


def test_run_add_overflow():
    source = "add a, b\nhalt"
    ok, data, _ = assemble(source)
    assert ok

    sigil = _make_sigil(
        inputs=[
            SigilParam(name="a", type="u8", register="A"),
            SigilParam(name="b", type="u8", register="B"),
        ],
    )
    test = SigilTest(inputs={"a": 255, "b": 1}, expect={"a": 0})
    passed, msg = run_test(data, test, sigil)
    assert passed, msg


def test_extract_code_with_fences():
    raw = "Here is the code:\n```asm\nadd a, b\nhalt\n```\nDone."
    code = extract_code(raw)
    assert code == "add a, b\nhalt"


def test_extract_code_plain():
    raw = "add a, b\nhalt"
    code = extract_code(raw)
    assert code == raw


def test_assemble_with_labels():
    source = "ld b, 5\nloop:\ninc a\ndjnz loop\nhalt"
    ok, data, err = assemble(source)
    assert ok, f"Assembly with labels failed: {err}"
    assert data is not None


def test_run_counter_loop():
    source = "ld a, 0\nloop:\ninc a\ndjnz loop\nhalt"
    ok, data, _ = assemble(source)
    assert ok

    sigil = _make_sigil(
        inputs=[SigilParam(name="count", type="u8", register="B")],
    )
    test = SigilTest(inputs={"count": 5}, expect={"a": 5})
    passed, msg = run_test(data, test, sigil)
    assert passed, msg


def test_assemble_lowercases_uppercase():
    source = "ADD A, B\nHALT"
    ok, data, err = assemble(source)
    assert ok, f"Uppercase assembly failed: {err}"
