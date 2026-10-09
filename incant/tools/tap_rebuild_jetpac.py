#!/usr/bin/env python3
"""Rebuild Jetpac TAP with a simple single-stage loader.

Extracts the screen and game code from the original multi-stage TAP
and repackages with a straightforward BASIC loader that doesn't rely
on system variable patching or multi-block FOR loops.
"""

import sys
from pathlib import Path


def xor_checksum(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result


def make_header_block(block_type: int, filename: str, data_len: int, param1: int, param2: int) -> bytes:
    """Build a TAP header block (type 0x00)."""
    name_bytes = filename.encode("ascii")[:10].ljust(10, b" ")
    payload = bytes([block_type]) + name_bytes
    payload += data_len.to_bytes(2, "little")
    payload += param1.to_bytes(2, "little")
    payload += param2.to_bytes(2, "little")

    flag = 0x00
    content = bytes([flag]) + payload
    checksum = xor_checksum(content)
    block = content + bytes([checksum])
    return len(block).to_bytes(2, "little") + block


def make_data_block(payload: bytes) -> bytes:
    """Build a TAP data block (type 0xFF)."""
    flag = 0xFF
    content = bytes([flag]) + payload
    checksum = xor_checksum(content)
    block = content + bytes([checksum])
    return len(block).to_bytes(2, "little") + block


def extract_blocks(tap_path: str) -> list[dict]:
    """Parse TAP into list of {flag, payload} dicts."""
    data = Path(tap_path).read_bytes()
    blocks = []
    pos = 0
    while pos < len(data):
        length = int.from_bytes(data[pos:pos+2], "little")
        pos += 2
        block_data = data[pos:pos+length]
        pos += length
        blocks.append({
            "flag": block_data[0],
            "payload": block_data[1:-1],
            "checksum": block_data[-1],
        })
    return blocks


def make_basic_loader(clear_addr: int, code_name: str, usr_addr: int) -> bytes:
    """Build a minimal BASIC program:
        10 CLEAR VAL "24575"
        20 LOAD ""CODE
        30 LOAD ""CODE
        40 RANDOMIZE USR VAL "24576"

    We load screen (unnamed) then code (unnamed) then run.
    Using VAL "number" is standard Spectrum practice to save tokeniser issues.
    """
    lines = []

    # Line 10: CLEAR VAL "24575"
    clear_str = str(clear_addr).encode("ascii")
    # CLEAR = 0xFD, VAL = 0xB0
    line10_body = bytes([0xFD, 0xB0, 0x22]) + clear_str + bytes([0x22, 0x0D])
    line10 = (10).to_bytes(2, "big") + len(line10_body).to_bytes(2, "little") + line10_body
    lines.append(line10)

    # Line 20: LOAD ""SCREEN$
    # LOAD = 0xEF, SCREEN$ = 0xAA
    line20_body = bytes([0xEF, 0x22, 0x22, 0xAA, 0x0D])
    line20 = (20).to_bytes(2, "big") + len(line20_body).to_bytes(2, "little") + line20_body
    lines.append(line20)

    # Line 30: LOAD ""CODE
    # LOAD = 0xEF, CODE = 0xAF
    line30_body = bytes([0xEF, 0x22, 0x22, 0xAF, 0x0D])
    line30 = (30).to_bytes(2, "big") + len(line30_body).to_bytes(2, "little") + line30_body
    lines.append(line30)

    # Line 40: RANDOMIZE USR VAL "24576"
    # RANDOMIZE = 0xF9, USR = 0xC0, VAL = 0xB0
    usr_str = str(usr_addr).encode("ascii")
    line40_body = bytes([0xF9, 0xC0, 0xB0, 0x22]) + usr_str + bytes([0x22, 0x0D])
    line40 = (40).to_bytes(2, "big") + len(line40_body).to_bytes(2, "little") + line40_body
    lines.append(line40)

    return b"".join(lines)


def rebuild_jetpac(src_path: str, dst_path: str) -> None:
    blocks = extract_blocks(src_path)

    # Original structure:
    # blocks[0] = header: Program "Jetpac1"
    # blocks[1] = data: BASIC loader
    # blocks[2] = header: Code "Jetpac2" at 0x4000 (screen)
    # blocks[3] = data: 6912 bytes screen
    # blocks[4] = header: Code "Jetpac3" at 0x6000 (game)
    # blocks[5] = data: 8192 bytes game code
    # blocks[6] = header: Code "Jetpac4" at 0x5B80 (relocation stub)
    # blocks[7] = data: 15 bytes stub
    # blocks[8] = header: Code "Jetpac5" at 0x5CB0 (JP HL)
    # blocks[9] = data: 1 byte
    # blocks[10] = header: Code "Jetpac6" at 0x5C78 (ERR_SP patch)
    # blocks[11] = data: 2 bytes

    screen_data = blocks[3]["payload"]  # 6912 bytes
    game_code = bytearray(blocks[5]["payload"])  # 8192 bytes
    stub_code = blocks[7]["payload"]  # 15 bytes: relocation + JP 61E5h

    print(f"Screen: {len(screen_data)} bytes")
    print(f"Game code: {len(game_code)} bytes")
    print(f"Stub: {len(stub_code)} bytes")

    # The stub does:
    #   LD HL, 6004h
    #   LD DE, 6000h
    #   LD BC, 2000h
    #   LDIR              ; copy 0x6004→0x6000, 8192 bytes (4-byte overlap shift)
    #   JP 61E5h          ; entry point
    #
    # We can apply this relocation ourselves: shift game code left by 4 bytes.
    # The LDIR copies byte-by-byte forward with a 4-byte overlap,
    # so byte[0] = byte[4], byte[1] = byte[5], ..., byte[8188] = byte[8188+4]
    # But bytes beyond 8192 don't exist — the last 4 bytes repeat.
    # Actually LDIR with BC=2000h copies exactly 8192 bytes:
    #   dest[i] = src[i] for i in 0..8191, but src starts 4 bytes later
    #   so effectively: game[0..8187] = game[4..8191], game[8188..8191] = game[8188..8191]

    relocated = bytearray(len(game_code))
    for i in range(len(game_code)):
        src_offset = i + 4
        if src_offset < len(game_code):
            relocated[i] = game_code[src_offset]
        else:
            # LDIR reads from already-written destination for the overlap tail
            relocated[i] = relocated[i - 4]

    # Entry point after relocation: JP 61E5h
    # But we shifted code left by 4, so 0x61E5 in the original = 0x61E1 in relocated?
    # No — the stub copies FROM 0x6004 TO 0x6000 so addresses STAY the same.
    # The code at 0x61E5 (offset 0x01E5 from 0x6000) now has the byte that WAS at 0x01E9.
    # But wait, addresses in the code are absolute — they reference 0x6000-based addrs.
    # The LDIR doesn't rebase anything, it just shifts bytes.
    #
    # Actually, let's think again. The original game code is loaded at 0x6000.
    # The stub copies game[4..8195] to game[0..8191] (shifting left 4 bytes).
    # Then jumps to 0x61E5.
    # The code at 0x61E5 after the shift is whatever was at offset 0x01E5 in the
    # relocated buffer, which is the original byte at offset 0x01E9 (0x61E9 absolute).
    #
    # This means the game was ASSEMBLED expecting the 4-byte shift to happen.
    # The original bytes 0-3 are garbage/placeholder that get overwritten.
    # After LDIR, the code is in its intended final layout.
    #
    # So our relocated buffer IS the correct final game state.
    # Entry point: 0x61E5 absolute = offset 0x01E5 in our buffer.
    # We load at 0x6000, so USR 0x61E5 = USR 25061.
    #
    # But actually we want to also handle the system var patches:
    # - ERR_SP (0x5C78) = 0x835A — this is copy protection, skip it
    # - JP (HL) at 0x5CB0 — also copy protection, skip it
    # These prevent BREAK from working. We don't need them for a clean load.

    # Actually, let's reconsider: maybe the entry should be via the stub.
    # The simplest approach: include the stub in our code block.
    # Load everything from 0x5B80, run USR 23424 (0x5B80).
    # The stub relocates and jumps. This preserves the original behaviour.
    #
    # But simpler: just load the pre-relocated code at 0x6000 and USR 0x61E5.

    entry_point = 0x61E5  # from JP 61E5h in the stub
    load_addr = 0x6000

    # Build new TAP
    output = bytearray()

    # 1. BASIC loader
    basic = make_basic_loader(
        clear_addr=load_addr - 1,  # CLEAR 24575 (0x5FFF)
        code_name="",
        usr_addr=entry_point,
    )
    # Program header: type=0, autostart line=10
    output += make_header_block(0, "Jetpac", len(basic), 10, len(basic))
    output += make_data_block(basic)

    # 2. Screen (LOAD ""SCREEN$ expects a Code block at 0x4000, 6912 bytes)
    output += make_header_block(3, "Jetpac", len(screen_data), 0x4000, 0x8000)
    output += make_data_block(screen_data)

    # 3. Game code (pre-relocated)
    output += make_header_block(3, "Jetpac", len(relocated), load_addr, 0x8000)
    output += make_data_block(bytes(relocated))

    Path(dst_path).write_bytes(bytes(output))
    print(f"\nWritten: {dst_path} ({len(output)} bytes)")
    print(f"Loader: CLEAR {load_addr - 1}, LOAD SCREEN$, LOAD CODE at 0x{load_addr:04X}, USR {entry_point}")
    print(f"\nBlocks: BASIC loader + screen (6912) + code ({len(relocated)})")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "tools/jetpac.tap"
    dst = sys.argv[2] if len(sys.argv) > 2 else "tools/jetpac_fixed.tap"
    rebuild_jetpac(src, dst)
