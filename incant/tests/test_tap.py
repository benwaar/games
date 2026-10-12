"""Tests for tap.py — TAP writer, reader, disassembler."""

import os
import tempfile
from pathlib import Path

import pytest

from incant.tap import (
    _xor_checksum,
    _make_header_block,
    _make_data_block,
    write_tap,
    read_tap,
    extract_code_blocks,
    disassemble,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_tap(binary_data: bytes, filename: str = "test", start_addr: int = 0x8000) -> str:
    """Write a TAP to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".tap")
    os.close(fd)
    write_tap(binary_data, filename, start_addr, path)
    return path


# ---------------------------------------------------------------------------
# XOR checksum
# ---------------------------------------------------------------------------

def test_xor_checksum_empty():
    assert _xor_checksum(b"") == 0


def test_xor_checksum_single():
    assert _xor_checksum(bytes([0xAB])) == 0xAB


def test_xor_checksum_cancels():
    assert _xor_checksum(bytes([0xFF, 0xFF])) == 0


def test_xor_checksum_known():
    # 0x00 (flag) ^ 0x03 (type) ^ 10 name bytes ^ lengths ^ params
    # tested implicitly via round-trip; check a trivial case here
    assert _xor_checksum(bytes([0x01, 0x02, 0x03])) == 0x01 ^ 0x02 ^ 0x03


# ---------------------------------------------------------------------------
# Header block structure
# ---------------------------------------------------------------------------

def test_header_block_length():
    block = _make_header_block(3, "test", 16, 0x8000, 0x8000)
    # 2-byte prefix + 1 flag + 17 payload + 1 checksum = 21 bytes total
    length_field = int.from_bytes(block[:2], "little")
    assert length_field == 19  # 1 flag + 17 payload + 1 checksum
    assert len(block) == 21


def test_header_block_flag():
    block = _make_header_block(3, "test", 0, 0, 0)
    # byte 2 (after 2-byte length prefix) is the flag
    assert block[2] == 0x00


def test_header_block_type():
    block = _make_header_block(3, "test", 0, 0, 0)
    # byte 3 = block type
    assert block[3] == 3


def test_header_block_filename_padded():
    block = _make_header_block(3, "hi", 0, 0, 0)
    # bytes 4..13 = filename (10 chars)
    name_bytes = block[4:14]
    assert name_bytes == b"hi        "


def test_header_block_filename_truncated():
    block = _make_header_block(3, "toolongname_extra", 0, 0, 0)
    name_bytes = block[4:14]
    assert name_bytes == b"toolongnam"


# ---------------------------------------------------------------------------
# Data block structure
# ---------------------------------------------------------------------------

def test_data_block_flag():
    block = _make_data_block(b"\x01\x02")
    assert block[2] == 0xFF


def test_data_block_length_prefix():
    payload = b"\x01\x02\x03"
    block = _make_data_block(payload)
    length_field = int.from_bytes(block[:2], "little")
    # 1 flag + 3 payload + 1 checksum = 5
    assert length_field == 5


def test_data_block_checksum():
    payload = bytes([0xAA, 0xBB])
    block = _make_data_block(payload)
    content = block[2:-1]  # flag + payload
    expected_checksum = _xor_checksum(content)
    assert block[-1] == expected_checksum


# ---------------------------------------------------------------------------
# Round-trip: write → read
# ---------------------------------------------------------------------------

def test_round_trip_basic():
    data = bytes(range(16))
    path = make_temp_tap(data, "roundtrip", 0x8000)
    try:
        blocks = read_tap(path)
        assert len(blocks) == 2

        header = blocks[0]
        assert header["flag"] == 0x00
        assert header["valid"] is True
        assert header["type"] == 3
        assert header["name"] == "roundtrip"
        assert header["data_len"] == 16
        assert header["param1"] == 0x8000
        assert header["param2"] == 0x8000

        data_block = blocks[1]
        assert data_block["flag"] == 0xFF
        assert data_block["valid"] is True
        assert data_block["payload"] == data
    finally:
        os.unlink(path)


def test_round_trip_checksums_valid():
    data = bytes([0x3E, 0x42, 0x76])  # LD A,0x42 ; HALT
    path = make_temp_tap(data, "chk", 0x9000)
    try:
        blocks = read_tap(path)
        assert all(b["valid"] for b in blocks)
    finally:
        os.unlink(path)


def test_round_trip_empty_binary():
    data = b""
    path = make_temp_tap(data, "empty", 0x8000)
    try:
        blocks = read_tap(path)
        assert blocks[1]["payload"] == b""
    finally:
        os.unlink(path)


def test_round_trip_large_binary():
    data = bytes(range(256)) * 16  # 4096 bytes
    path = make_temp_tap(data, "large", 0x6000)
    try:
        code_blocks = extract_code_blocks(path)
        assert len(code_blocks) == 1
        assert code_blocks[0]["data"] == data
        assert code_blocks[0]["start_addr"] == 0x6000
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# extract_code_blocks
# ---------------------------------------------------------------------------

def test_extract_code_blocks_returns_name_and_addr():
    data = b"\x00\x01\x02"
    path = make_temp_tap(data, "mycode", 0xC000)
    try:
        blocks = extract_code_blocks(path)
        assert len(blocks) == 1
        assert blocks[0]["name"] == "mycode"
        assert blocks[0]["start_addr"] == 0xC000
        assert blocks[0]["data"] == data
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Disassembler
# ---------------------------------------------------------------------------

def test_disassemble_halt():
    # 0x76 = HALT
    result = disassemble(bytes([0x76]), 0x8000)
    assert "8000" in result
    assert "76" in result.upper()
    assert "HALT" in result.upper()


def test_disassemble_nop():
    result = disassemble(bytes([0x00]), 0x8000)
    assert "NOP" in result.upper()


def test_disassemble_ld_a_n():
    # 0x3E 0x05 = LD A,5
    result = disassemble(bytes([0x3E, 0x05]), 0x8000)
    lines = result.strip().splitlines()
    assert len(lines) == 1
    assert "8000" in lines[0]
    assert "LD" in lines[0].upper()
    assert "A" in lines[0].upper()


def test_disassemble_known_sequence():
    # LD A,0x42 ; HALT
    code = bytes([0x3E, 0x42, 0x76])
    result = disassemble(code, 0x8000)
    lines = result.strip().splitlines()
    assert len(lines) == 2
    # First instruction at 0x8000
    assert "8000" in lines[0]
    # Second at 0x8002
    assert "8002" in lines[1]
    assert "HALT" in lines[1].upper()


def test_disassemble_address_offset():
    code = bytes([0x00])  # NOP
    result = disassemble(code, 0x4000)
    assert "4000" in result


def test_disassemble_returns_string():
    result = disassemble(b"\x00", 0x8000)
    assert isinstance(result, str)
    assert len(result) > 0


def test_disassemble_empty():
    result = disassemble(b"", 0x8000)
    assert result == ""


# ---------------------------------------------------------------------------
# TAP file path
# ---------------------------------------------------------------------------

def test_write_tap_creates_file():
    fd, path = tempfile.mkstemp(suffix=".tap")
    os.close(fd)
    os.unlink(path)
    try:
        write_tap(b"\x00\x76", "test", 0x8000, path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
    finally:
        if Path(path).exists():
            os.unlink(path)
