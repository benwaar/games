"""Tests for the glyph-audio Z80 beeper converter."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "glyph"))
sys.path.insert(0, str(ROOT / "incant"))

from glyphz80.beeper import (
    CPU_CLOCK,
    HALF_CYCLE_OVERHEAD_T,
    compile_glyph_audio,
    duration_to_cycles,
    freq_to_half_period,
    half_period_to_hl,
    load_spec,
    midi_to_freq,
    note_to_freq,
    note_to_midi,
)
from incant.tap import write_spectrum_tap, read_tap

# --- note_to_midi ---


def test_c3_midi():
    assert note_to_midi("C3") == 48


def test_c4_midi():
    assert note_to_midi("C4") == 60


def test_a4_midi():
    assert note_to_midi("A4") == 69


def test_c5_midi():
    assert note_to_midi("C5") == 72


def test_asharp3_midi():
    assert note_to_midi("A#3") == 58


def test_rest_midi():
    assert note_to_midi("REST") == -1


# --- midi_to_freq / note_to_freq ---


def test_a4_freq():
    assert midi_to_freq(69) == pytest.approx(440.0, abs=0.01)


def test_c4_freq():
    assert note_to_freq("C4") == pytest.approx(261.63, abs=0.01)


def test_c3_freq():
    assert note_to_freq("C3") == pytest.approx(130.81, abs=0.01)


def test_rest_freq():
    assert note_to_freq("REST") == 0.0


# --- freq_to_half_period ---


def test_a4_half_period():
    hp = freq_to_half_period(440.0)
    expected = round(CPU_CLOCK / (2 * 440))
    assert hp == expected


def test_c3_half_period():
    freq = note_to_freq("C3")
    hp = freq_to_half_period(freq)
    assert 13000 < hp < 14000


def test_zero_freq_half_period():
    assert freq_to_half_period(0.0) == 0


# --- duration_to_cycles ---


def test_a4_1sec_cycles():
    """440 Hz for 1000ms = 440 cycles."""
    cycles = duration_to_cycles(440.0, 1000)
    assert cycles == 440


def test_c3_500ms_cycles():
    freq = note_to_freq("C3")
    cycles = duration_to_cycles(freq, 500)
    assert cycles == round(freq * 0.5)


def test_rest_zero_cycles():
    assert duration_to_cycles(0.0, 1000) == 0


# --- half_period_to_hl ---


def test_hl_positive():
    hp = freq_to_half_period(440.0)
    hl = half_period_to_hl(hp)
    assert hl > 0


def test_hl_round_trip():
    """HL * 26 + 131 should approximate the target half-period."""
    hp = freq_to_half_period(440.0)
    hl = half_period_to_hl(hp)
    reconstructed = hl * 26 + HALF_CYCLE_OVERHEAD_T
    assert abs(reconstructed - hp) / hp < 0.02


# --- compile_glyph_audio ---


def _make_spec(notes: list[dict]) -> dict:
    return {
        "type": "glyph-audio",
        "version": 1,
        "pieces": [
            {
                "id": "test",
                "kind": "melody",
                "notes": notes,
            }
        ],
    }


def test_single_note_produces_binary():
    spec = _make_spec([{"note": "A4", "ms": 500}])
    binary = compile_glyph_audio(spec)
    assert len(binary) > 0


def test_melody_produces_binary():
    spec = _make_spec([
        {"note": "C4", "ms": 300},
        {"note": "E4", "ms": 300},
        {"note": "G4", "ms": 300},
    ])
    binary = compile_glyph_audio(spec)
    assert len(binary) > 20


def test_rest_no_out_instructions():
    """REST segments should not contain OUT (0xFE) = D3 FE."""
    spec = _make_spec([{"note": "REST", "ms": 500}])
    binary = compile_glyph_audio(spec)
    rest_stub = binary[:9]
    for i in range(len(rest_stub) - 1):
        assert not (rest_stub[i] == 0xD3 and rest_stub[i + 1] == 0xFE), (
            f"Found OUT (0xFE) in REST stub at offset {i}"
        )


def test_binary_starts_with_valid_opcodes():
    """First instruction should be LD HL,nn (0x21) for a note or rest."""
    spec = _make_spec([{"note": "D3", "ms": 200}])
    binary = compile_glyph_audio(spec)
    assert binary[0] == 0x21, f"Expected LD HL (0x21), got 0x{binary[0]:02X}"


def test_binary_contains_halt():
    """Binary should contain HALT (0x76) after the note stubs."""
    spec = _make_spec([{"note": "A4", "ms": 100}])
    binary = compile_glyph_audio(spec)
    assert 0x76 in binary


def test_mixed_notes_and_rests():
    spec = _make_spec([
        {"note": "C4", "ms": 300},
        {"note": "REST", "ms": 200},
        {"note": "E4", "ms": 300},
    ])
    binary = compile_glyph_audio(spec)
    assert len(binary) > 30


def test_invalid_type_raises():
    spec = {"type": "wrong", "pieces": []}
    with pytest.raises(ValueError, match="glyph-audio"):
        compile_glyph_audio(spec)


def test_no_notes_raises():
    spec = {"type": "glyph-audio", "pieces": []}
    with pytest.raises(ValueError, match="No melody notes"):
        compile_glyph_audio(spec)


# --- load_spec ---


def test_load_spec_from_file():
    spec = _make_spec([{"note": "A4", "ms": 100}])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        f.flush()
        loaded = load_spec(f.name)
    assert loaded["type"] == "glyph-audio"


# --- a1.json integration ---


A1_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "a1.json"


@pytest.mark.skipif(not A1_PATH.exists(), reason="a1.json not found")
def test_a1_compiles():
    """The real a1.json spec compiles to a non-trivial binary."""
    spec = load_spec(A1_PATH)
    binary = compile_glyph_audio(spec)
    assert len(binary) > 50


# --- TAP generation (using incant.tap) ---


def test_tap_has_four_blocks(tmp_path):
    """TAP should have 4 blocks: BASIC header, BASIC data, CODE header, CODE data."""
    spec = _make_spec([{"note": "A4", "ms": 500}])
    binary = compile_glyph_audio(spec)
    tap_path = str(tmp_path / "test.tap")
    write_spectrum_tap(binary, "test", 0x8000, tap_path)
    blocks = read_tap(tap_path)
    assert len(blocks) == 4


def test_tap_header_flags(tmp_path):
    """First byte of each block is the flag: 0x00 for headers, 0xFF for data."""
    spec = _make_spec([{"note": "A4", "ms": 500}])
    binary = compile_glyph_audio(spec)
    tap_path = str(tmp_path / "test.tap")
    write_spectrum_tap(binary, "test", 0x8000, tap_path)
    blocks = read_tap(tap_path)
    flags = [b["flag"] for b in blocks]
    assert flags == [0x00, 0xFF, 0x00, 0xFF]


def test_tap_code_block_contains_binary(tmp_path):
    """The last data block should contain the original Z80 binary."""
    spec = _make_spec([{"note": "C4", "ms": 200}, {"note": "E4", "ms": 200}])
    binary = compile_glyph_audio(spec)
    tap_path = str(tmp_path / "test.tap")
    write_spectrum_tap(binary, "test", 0x8000, tap_path)
    blocks = read_tap(tap_path)
    code_data = blocks[3]["payload"]
    assert code_data == binary


def test_tap_checksums_valid(tmp_path):
    """Each block's checksum should be valid."""
    spec = _make_spec([{"note": "A4", "ms": 100}])
    binary = compile_glyph_audio(spec)
    tap_path = str(tmp_path / "test.tap")
    write_spectrum_tap(binary, "test", 0x8000, tap_path)
    blocks = read_tap(tap_path)
    for b in blocks:
        assert b["valid"], f"Checksum invalid for block flag=0x{b['flag']:02X}"


def test_tap_code_header_has_load_address(tmp_path):
    """CODE header (3rd block) should encode the load address."""
    spec = _make_spec([{"note": "A4", "ms": 100}])
    binary = compile_glyph_audio(spec)
    addr = 0x8000
    tap_path = str(tmp_path / "test.tap")
    write_spectrum_tap(binary, "test", addr, tap_path)
    blocks = read_tap(tap_path)
    code_header = blocks[2]
    assert code_header["param1"] == addr


@pytest.mark.skipif(not A1_PATH.exists(), reason="a1.json not found")
def test_a1_tap(tmp_path):
    """a1.json produces a valid TAP file."""
    spec = load_spec(A1_PATH)
    binary = compile_glyph_audio(spec)
    tap_path = str(tmp_path / "a1.tap")
    write_spectrum_tap(binary, "glyph", 0x8000, tap_path)
    blocks = read_tap(tap_path)
    assert len(blocks) == 4
