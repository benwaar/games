"""Tests for glyph/z80/scr_loader.asm and build_scr.py."""

import sys
from pathlib import Path

import pytest
import z80 as z80lib

# Ensure glyph/z80 is importable from the tests directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_scr import assemble_loader, build  # noqa: E402

SCR_SIZE = 6912
SCREEN_START = 0x4000
CODE_ORG = 0x8000


def test_assemble_produces_bytes() -> None:
    """Assembler must return a non-empty bytes object."""
    data = assemble_loader()
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_assembled_code_is_small() -> None:
    """The loader is a handful of instructions — sanity-check it isn't huge."""
    data = assemble_loader()
    # ldir + surrounding ld instructions fit in under 20 bytes
    assert len(data) <= 20, f"Loader too large: {len(data)} bytes"


def test_build_concatenates_correctly(tmp_path: Path) -> None:
    """build() concatenates loader + SCR data and writes the correct byte count."""
    scr_bytes = bytes(SCR_SIZE)  # all zeros
    scr_file = tmp_path / "test.scr"
    scr_file.write_bytes(scr_bytes)

    out_file = tmp_path / "out.bin"
    build(scr_file, out_file)

    combined = out_file.read_bytes()
    loader_len = len(assemble_loader())
    assert len(combined) == loader_len + SCR_SIZE
    assert combined[loader_len:] == scr_bytes


def test_build_rejects_wrong_scr_size(tmp_path: Path) -> None:
    """build() must reject an SCR file that isn't exactly 6912 bytes."""
    bad_scr = tmp_path / "bad.scr"
    bad_scr.write_bytes(bytes(100))
    out_file = tmp_path / "out.bin"

    with pytest.raises(ValueError, match="6912"):
        build(bad_scr, out_file)


def _run_until_halted(machine: z80lib.Z80Machine, max_rounds: int = 20) -> bool:
    """Call machine.run() in a loop until halted or max_rounds exceeded.

    ticks_to_stop in the z80 library is an absolute threshold, not a
    relative duration.  Bumping it each round is the safe way to guarantee
    forward progress for large block operations like ldir over 6912 bytes.
    """
    step = 500_000
    for i in range(1, max_rounds + 1):
        machine.ticks_to_stop = step * i
        machine.run()
        if machine.halted:
            return True
    return False


def test_execution_copies_scr_to_screen_memory() -> None:
    """Run the combined binary in the Z80 emulator; check screen memory matches SCR."""
    # Use a recognisable pattern so any byte-order bug is visible
    scr_bytes = bytes(range(256)) * (SCR_SIZE // 256) + bytes(range(SCR_SIZE % 256))
    assert len(scr_bytes) == SCR_SIZE

    loader = assemble_loader()
    combined = loader + scr_bytes

    machine = z80lib.Z80Machine()
    machine.set_memory_block(CODE_ORG, combined)
    machine.pc = CODE_ORG

    halted = _run_until_halted(machine)
    assert halted, "Program did not halt within round limit"

    actual = bytes(machine.memory[SCREEN_START : SCREEN_START + SCR_SIZE])
    assert actual == scr_bytes, "Screen memory does not match SCR data after execution"


def test_execution_with_all_zeros_scr() -> None:
    """All-zeros SCR should leave screen memory zeroed."""
    scr_bytes = bytes(SCR_SIZE)

    loader = assemble_loader()
    combined = loader + scr_bytes

    machine = z80lib.Z80Machine()
    # Pre-fill screen memory with 0xFF to prove the copy happens
    machine.set_memory_block(SCREEN_START, bytes([0xFF] * SCR_SIZE))
    machine.set_memory_block(CODE_ORG, combined)
    machine.pc = CODE_ORG

    halted = _run_until_halted(machine)
    assert halted, "Program did not halt within round limit"

    actual = bytes(machine.memory[SCREEN_START : SCREEN_START + SCR_SIZE])
    assert actual == scr_bytes, "Screen memory should be all zeros after copying zero SCR"
