#!/usr/bin/env python3
"""TAP file analyser — parse blocks, check checksums, report corruption."""

import sys
from pathlib import Path


def xor_checksum(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result


BLOCK_TYPES = {0: "Program", 1: "Number array", 2: "Character array", 3: "Code"}


def parse_header(payload: bytes) -> dict:
    if len(payload) != 17:
        return {"error": f"header payload is {len(payload)} bytes, expected 17"}
    block_type = payload[0]
    filename = payload[1:11].decode("ascii", errors="replace").rstrip()
    data_len = int.from_bytes(payload[11:13], "little")
    param1 = int.from_bytes(payload[13:15], "little")
    param2 = int.from_bytes(payload[15:17], "little")
    return {
        "type": BLOCK_TYPES.get(block_type, f"Unknown({block_type})"),
        "type_id": block_type,
        "filename": filename,
        "data_length": data_len,
        "param1": param1,
        "param2": param2,
    }


def analyse_tap(filepath: str) -> None:
    data = Path(filepath).read_bytes()
    filesize = len(data)
    print(f"File: {filepath}")
    print(f"Size: {filesize} bytes ({filesize:,})")
    print(f"{'=' * 60}")

    pos = 0
    block_num = 0
    issues = []
    pending_header = None

    while pos < filesize:
        block_num += 1

        # Check we have enough bytes for the length field
        if pos + 2 > filesize:
            issues.append(f"Block {block_num}: truncated — only {filesize - pos} byte(s) left, need 2 for length")
            break

        block_len = int.from_bytes(data[pos:pos+2], "little")
        block_start = pos
        pos += 2

        print(f"\nBlock {block_num} at offset {block_start} (0x{block_start:04X})")
        print(f"  Declared length: {block_len}")

        if block_len == 0:
            issues.append(f"Block {block_num}: zero-length block")
            print(f"  ** ISSUE: zero-length block")
            continue

        # Check we have enough bytes for the block
        if pos + block_len > filesize:
            actual = filesize - pos
            issues.append(f"Block {block_num}: truncated — declared {block_len} bytes but only {actual} remain")
            print(f"  ** ISSUE: truncated — need {block_len} bytes, only {actual} available")
            print(f"  Available bytes (hex): {data[pos:pos+min(actual, 32)].hex(' ')}")
            break

        block_data = data[pos:pos+block_len]
        pos += block_len

        flag = block_data[0]
        payload = block_data[1:-1]
        stored_checksum = block_data[-1]
        computed_checksum = xor_checksum(block_data[:-1])

        flag_name = "header" if flag == 0x00 else "data" if flag == 0xFF else f"unknown(0x{flag:02X})"
        print(f"  Flag: 0x{flag:02X} ({flag_name})")
        print(f"  Payload: {len(payload)} bytes")
        print(f"  Checksum: stored=0x{stored_checksum:02X}, computed=0x{computed_checksum:02X}", end="")

        if stored_checksum == computed_checksum:
            print(" OK")
        else:
            print(" ** MISMATCH **")
            issues.append(f"Block {block_num}: checksum mismatch — stored 0x{stored_checksum:02X}, computed 0x{computed_checksum:02X}")

        if flag not in (0x00, 0xFF):
            issues.append(f"Block {block_num}: unexpected flag byte 0x{flag:02X}")

        # Parse header blocks
        if flag == 0x00:
            hdr = parse_header(payload)
            if "error" in hdr:
                print(f"  Header: {hdr['error']}")
                issues.append(f"Block {block_num}: {hdr['error']}")
            else:
                print(f"  Header type: {hdr['type']}")
                print(f"  Filename: \"{hdr['filename']}\"")
                print(f"  Data length: {hdr['data_length']}")
                if hdr["type_id"] == 3:
                    print(f"  Start address: {hdr['param1']} (0x{hdr['param1']:04X})")
                elif hdr["type_id"] == 0:
                    print(f"  Autostart line: {hdr['param1']}")
                    print(f"  Variable offset: {hdr['param2']}")
                pending_header = hdr

        elif flag == 0xFF and pending_header:
            expected_len = pending_header["data_length"]
            if len(payload) != expected_len:
                issues.append(
                    f"Block {block_num}: data length {len(payload)} doesn't match "
                    f"header's declared {expected_len}"
                )
                print(f"  ** ISSUE: expected {expected_len} bytes from header, got {len(payload)}")
            else:
                print(f"  Data length matches header: {expected_len} bytes")
            pending_header = None

        # Show first bytes of payload
        preview_len = min(len(payload), 32)
        if preview_len > 0:
            print(f"  First {preview_len} bytes: {payload[:preview_len].hex(' ')}")

    print(f"\n{'=' * 60}")
    print(f"Total blocks: {block_num}")
    if pos < filesize:
        trailing = filesize - pos
        issues.append(f"{trailing} trailing bytes after last block")
        print(f"Trailing bytes: {trailing}")

    if issues:
        print(f"\n** {len(issues)} issue(s) found: **")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo issues found — TAP file looks valid.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.tap>")
        sys.exit(1)
    analyse_tap(sys.argv[1])
