"""Tests for sigil parser."""

from pathlib import Path

from incant.sigil import parse_sigil


EXAMPLES_DIR = Path(__file__).parent.parent / "sigils" / "examples"


def test_parse_z80_sigil():
    sigil = parse_sigil(EXAMPLES_DIR / "z80_add.sigil.yaml")
    assert sigil.name == "add"
    assert sigil.target == "z80"
    assert len(sigil.signature.inputs) == 2
    assert sigil.signature.inputs[0].register == "A"
    assert sigil.signature.inputs[1].register == "B"
    assert sigil.signature.output_type == "u8"
    assert len(sigil.tests) == 3


def test_parse_wat_sigil():
    sigil = parse_sigil(EXAMPLES_DIR / "wat_add.sigil.yaml")
    assert sigil.name == "add"
    assert sigil.target == "wat"
    assert sigil.export == "add"
    assert len(sigil.signature.inputs) == 2
    assert sigil.signature.inputs[0].type == "i32"
    assert sigil.signature.output_type == "i32"


def test_parse_wat_memory_sigil():
    sigil = parse_sigil(EXAMPLES_DIR / "wat_memory_swap.sigil.yaml")
    assert sigil.memory == 1
    assert sigil.export == "memory_swap"


def test_parse_z80_counter():
    sigil = parse_sigil(EXAMPLES_DIR / "z80_counter.sigil.yaml")
    assert sigil.name == "counter"
    assert sigil.signature.inputs[0].register == "B"
    assert sigil.tests[0].inputs == {"count": 5}
    assert sigil.tests[0].expect == {"a": 5}
