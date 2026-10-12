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
    write_spectrum_tap,
    _int5,
    _basic_line,
    _make_basic_loader,
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


# ---------------------------------------------------------------------------
# 5-byte integer encoding
# ---------------------------------------------------------------------------

def test_int5_positive():
    result = _int5(32768)
    assert len(result) == 5
    assert result[0] == 0x00  # marker
    assert result[1] == 0x00  # sign (positive)
    assert result[2] == 0x00  # lo byte of 32768
    assert result[3] == 0x80  # hi byte of 32768
    assert result[4] == 0x00  # terminator


def test_int5_small():
    result = _int5(10)
    assert result == bytes([0x00, 0x00, 10, 0x00, 0x00])


def test_int5_zero():
    result = _int5(0)
    assert result == bytes([0x00, 0x00, 0x00, 0x00, 0x00])


def test_int5_negative():
    result = _int5(-1)
    assert result[1] == 0xFF  # sign byte for negative


def test_int5_boundary():
    result = _int5(32767)
    assert result == bytes([0x00, 0x00, 0xFF, 0x7F, 0x00])


# ---------------------------------------------------------------------------
# BASIC line encoding
# ---------------------------------------------------------------------------

def test_basic_line_structure():
    body = bytes([0xF9, 0xC0])  # RANDOMIZE USR
    line = _basic_line(10, body)
    assert line[0:2] == (10).to_bytes(2, "big")  # line number big-endian
    length = int.from_bytes(line[2:4], "little")
    assert length == len(body) + 1  # body + CR
    assert line[-1] == 0x0D  # CR terminator


def test_basic_line_ten():
    body = bytes([0xEF, 0x22, 0x22])  # LOAD ""
    line = _basic_line(10, body)
    assert line[0] == 0x00
    assert line[1] == 0x0A  # line 10


# ---------------------------------------------------------------------------
# BASIC auto-loader
# ---------------------------------------------------------------------------

def test_basic_loader_contains_tokens():
    loader = _make_basic_loader(32768)
    assert 0xFD in loader  # CLEAR
    assert 0xEF in loader  # LOAD
    assert 0xAF in loader  # CODE
    assert 0xF9 in loader  # RANDOMIZE
    assert 0xC0 in loader  # USR


def test_basic_loader_contains_addresses():
    loader = _make_basic_loader(32768)
    # Should contain "32767" (CLEAR addr-1) and "32768" (USR addr) as ASCII
    assert b"32767" in loader
    assert b"32768" in loader


def test_basic_loader_ends_with_cr():
    loader = _make_basic_loader(32768)
    assert loader[-1] == 0x0D


def test_basic_loader_is_line_10():
    loader = _make_basic_loader(32768)
    assert loader[0:2] == (10).to_bytes(2, "big")


def test_basic_loader_has_int5_markers():
    loader = _make_basic_loader(32768)
    # 0x0E marks the start of a 5-byte integer literal
    count = loader.count(0x0E)
    assert count == 2  # one for CLEAR value, one for USR value


# ---------------------------------------------------------------------------
# write_spectrum_tap — full TAP with BASIC loader + CODE
# ---------------------------------------------------------------------------

def make_temp_spectrum_tap(
    binary_data: bytes, filename: str = "test", start_addr: int = 0x8000
) -> str:
    fd, path = tempfile.mkstemp(suffix=".tap")
    os.close(fd)
    write_spectrum_tap(binary_data, filename, start_addr, path)
    return path


def test_spectrum_tap_four_blocks():
    path = make_temp_spectrum_tap(b"\x3E\x42\x76")
    try:
        blocks = read_tap(path)
        assert len(blocks) == 4
    finally:
        os.unlink(path)


def test_spectrum_tap_block_types():
    path = make_temp_spectrum_tap(b"\x3E\x42\x76")
    try:
        blocks = read_tap(path)
        assert blocks[0]["flag"] == 0x00  # BASIC header
        assert blocks[0]["type"] == 0     # Program
        assert blocks[1]["flag"] == 0xFF  # BASIC data
        assert blocks[2]["flag"] == 0x00  # CODE header
        assert blocks[2]["type"] == 3     # Code
        assert blocks[3]["flag"] == 0xFF  # CODE data
    finally:
        os.unlink(path)


def test_spectrum_tap_autostart_line():
    path = make_temp_spectrum_tap(b"\x00")
    try:
        blocks = read_tap(path)
        assert blocks[0]["param1"] == 10  # autostart line 10
    finally:
        os.unlink(path)


def test_spectrum_tap_code_address():
    path = make_temp_spectrum_tap(b"\x76", start_addr=0xC000)
    try:
        blocks = read_tap(path)
        assert blocks[2]["param1"] == 0xC000
    finally:
        os.unlink(path)


def test_spectrum_tap_code_data_roundtrip():
    code = bytes(range(64))
    path = make_temp_spectrum_tap(code, start_addr=0x8000)
    try:
        code_blocks = extract_code_blocks(path)
        assert len(code_blocks) == 1
        assert code_blocks[0]["data"] == code
        assert code_blocks[0]["start_addr"] == 0x8000
    finally:
        os.unlink(path)


def test_spectrum_tap_all_checksums_valid():
    path = make_temp_spectrum_tap(b"\x3E\x42\x76")
    try:
        blocks = read_tap(path)
        assert all(b["valid"] for b in blocks)
    finally:
        os.unlink(path)


def test_spectrum_tap_basic_is_parseable():
    """The BASIC data block should contain valid tokenised BASIC."""
    path = make_temp_spectrum_tap(b"\x76", start_addr=0x8000)
    try:
        blocks = read_tap(path)
        basic_data = blocks[1]["payload"]
        # Should start with line number 0x00 0x0A (line 10, big-endian)
        assert basic_data[0] == 0x00
        assert basic_data[1] == 0x0A
        # Should end with 0x0D (CR)
        assert basic_data[-1] == 0x0D
    finally:
        os.unlink(path)
