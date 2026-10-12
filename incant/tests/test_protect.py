"""Tests for protect.py — self-verifying binary protection."""

import z80 as z80lib
import pytest

from incant.protect import (
    VERIFY_PREFIX_LEN,
    HALT_OPCODE,
    CHECKSUM_OFFSET,
    JP_NZ_OFFSET,
    wrap_z80_checksum,
    tamper_byte,
    bypass_nop_branch,
    bypass_fix_checksum,
    generate_self_verify_wat,
    assemble_wat,
    run_wasm_module,
    tamper_wasm_data_segment,
    _xor_checksum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simple Z80 payload: LD A, 0x42 ; HALT — puts 0x42 in A and halts
SIMPLE_PAYLOAD = bytes([0x3E, 0x42, 0x76])
LOAD_ADDR = 0x8000


def run_z80(binary: bytes, load_addr: int = LOAD_ADDR, ticks: int = 50_000) -> z80lib.Z80Machine:
    machine = z80lib.Z80Machine()
    machine.set_memory_block(load_addr, binary)
    machine.pc = load_addr
    machine.ticks_to_stop = ticks
    machine.run()
    return machine


# ---------------------------------------------------------------------------
# Z80: basic structure
# ---------------------------------------------------------------------------

def test_z80_prefix_length():
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    assert len(protected) == VERIFY_PREFIX_LEN + len(SIMPLE_PAYLOAD) + 1


def test_z80_ends_with_halt():
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    assert protected[-1] == HALT_OPCODE


def test_z80_payload_preserved():
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    assert protected[VERIFY_PREFIX_LEN:-1] == SIMPLE_PAYLOAD


# ---------------------------------------------------------------------------
# Z80: intact binary runs correctly
# ---------------------------------------------------------------------------

def test_z80_intact_runs():
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    machine = run_z80(protected)
    assert machine.halted
    # Payload sets A = 0x42, then HALTs
    assert machine.a == 0x42


def test_z80_intact_reaches_payload():
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    machine = run_z80(protected)
    # Payload HALT is at load_addr + prefix + 2 (LD A,0x42 is 2 bytes, then HALT)
    payload_halt_addr = LOAD_ADDR + VERIFY_PREFIX_LEN + 2
    assert machine.pc == payload_halt_addr + 1


# ---------------------------------------------------------------------------
# Z80: tamper detection
# ---------------------------------------------------------------------------

def test_z80_tamper_detected():
    """Flip one byte in the payload region — verifier should catch it."""
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    tampered = tamper_byte(protected, VERIFY_PREFIX_LEN)  # first payload byte
    machine = run_z80(tampered)
    assert machine.halted
    # Should halt at the crash HALT, not the payload HALT
    crash_addr = LOAD_ADDR + VERIFY_PREFIX_LEN + len(SIMPLE_PAYLOAD)
    assert machine.pc == crash_addr + 1


def test_z80_tamper_every_payload_byte():
    """Every single-byte flip in the payload is detected."""
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    crash_addr = LOAD_ADDR + VERIFY_PREFIX_LEN + len(SIMPLE_PAYLOAD)

    for i in range(len(SIMPLE_PAYLOAD)):
        offset = VERIFY_PREFIX_LEN + i
        tampered = tamper_byte(protected, offset)
        machine = run_z80(tampered)
        assert machine.halted, f"Byte {i}: did not halt"
        assert machine.pc == crash_addr + 1, f"Byte {i}: halted at wrong address"


def test_z80_tamper_does_not_reach_payload():
    """Tampered binary never executes the payload (A should not be 0x42)."""
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    tampered = tamper_byte(protected, VERIFY_PREFIX_LEN)
    machine = run_z80(tampered)
    assert machine.a != 0x42


# ---------------------------------------------------------------------------
# Z80: attacker bypasses
# ---------------------------------------------------------------------------

def test_z80_bypass_nop_branch():
    """Attacker NOPs the JP NZ → tampered binary runs anyway."""
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    tampered = tamper_byte(protected, VERIFY_PREFIX_LEN)
    bypassed = bypass_nop_branch(tampered)
    machine = run_z80(bypassed)
    assert machine.halted
    payload_halt_addr = LOAD_ADDR + VERIFY_PREFIX_LEN + 2
    assert machine.pc == payload_halt_addr + 1


def test_z80_bypass_fix_checksum():
    """Attacker recomputes checksum after tampering → check passes."""
    protected = wrap_z80_checksum(SIMPLE_PAYLOAD, LOAD_ADDR)
    tampered = tamper_byte(protected, VERIFY_PREFIX_LEN)
    bypassed = bypass_fix_checksum(tampered)
    machine = run_z80(bypassed)
    assert machine.halted
    payload_halt_addr = LOAD_ADDR + VERIFY_PREFIX_LEN + 2
    assert machine.pc == payload_halt_addr + 1


# ---------------------------------------------------------------------------
# Z80: edge cases
# ---------------------------------------------------------------------------

def test_z80_minimum_payload():
    """Single-byte payload works correctly."""
    payload = bytes([0x76])  # just HALT
    protected = wrap_z80_checksum(payload, LOAD_ADDR)
    machine = run_z80(protected)
    assert machine.halted


def test_z80_large_payload():
    """256-byte payload works correctly."""
    payload = bytes(range(256))
    protected = wrap_z80_checksum(payload + bytes([0x76]), LOAD_ADDR)
    machine = run_z80(protected)
    assert machine.halted


# ---------------------------------------------------------------------------
# WASM: intact module runs correctly
# ---------------------------------------------------------------------------

def test_wasm_intact_runs():
    payload = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    wat = generate_self_verify_wat(payload, expected_result=42)
    wasm_path = assemble_wat(wat)
    result = run_wasm_module(wasm_path)
    assert result == 42


def test_wasm_different_payloads():
    for data in [b"\x00", b"\xFF" * 100, bytes(range(50))]:
        wat = generate_self_verify_wat(data, expected_result=99)
        wasm_path = assemble_wat(wat)
        assert run_wasm_module(wasm_path) == 99


# ---------------------------------------------------------------------------
# WASM: tamper detection
# ---------------------------------------------------------------------------

def test_wasm_tamper_detected():
    payload = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    wat = generate_self_verify_wat(payload, expected_result=42)
    wasm_path = assemble_wat(wat)
    wasm_bytes = wasm_path.read_bytes()

    tampered = tamper_wasm_data_segment(wasm_bytes, payload)
    tampered_path = wasm_path.with_name("tampered.wasm")
    tampered_path.write_bytes(tampered)

    with pytest.raises(RuntimeError):
        run_wasm_module(tampered_path)


# ---------------------------------------------------------------------------
# WASM: bypass (patch the expected checksum constant in the binary)
# ---------------------------------------------------------------------------

def test_wasm_bypass_patch_checksum():
    """Attacker finds and patches the expected checksum in the WASM binary."""
    payload = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    original_checksum = _xor_checksum(payload)
    wat = generate_self_verify_wat(payload, expected_result=42)
    wasm_path = assemble_wat(wat)
    wasm_bytes = bytearray(wasm_path.read_bytes())

    # Tamper the data segment
    idx = wasm_bytes.find(payload)
    wasm_bytes[idx] ^= 0x01
    new_checksum = _xor_checksum(bytes(wasm_bytes[idx:idx + len(payload)]))

    # Find and replace the i32.const for the expected checksum
    # In WASM binary, i32.const is opcode 0x41 followed by LEB128 value
    # For small values (< 128), LEB128 is just the byte itself
    # We need to find the comparison: i32.const <original_checksum>
    # and change it to i32.const <new_checksum>
    #
    # Simpler approach: scan for 0x41 followed by original_checksum
    # and replace with new_checksum (works for checksums < 128)
    if original_checksum < 128 and new_checksum < 128:
        target = bytes([0x41, original_checksum])
        # Find the LAST occurrence (the comparison constant, not loop init)
        last_idx = wasm_bytes.rfind(target)
        if last_idx >= 0:
            wasm_bytes[last_idx + 1] = new_checksum

    bypassed_path = wasm_path.with_name("bypassed.wasm")
    bypassed_path.write_bytes(bytes(wasm_bytes))
    result = run_wasm_module(bypassed_path)
    assert result == 42
