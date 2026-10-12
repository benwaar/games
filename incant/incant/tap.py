"""TAP file writer, reader, and Z80 disassembler.

ZX Spectrum TAP format: each block is a 2-byte little-endian length prefix
followed by the block data. Block data is: flag byte + payload + XOR checksum.

Header block (flag=0x00): type (1) + 10-char filename + data_len (2) + param1 (2) + param2 (2)
  = 17 bytes payload, so block data = 1 + 17 + 1 = 19 bytes, length prefix = 19.

Data block (flag=0xFF): raw bytes.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# XOR checksum
# ---------------------------------------------------------------------------

def _xor_checksum(data: bytes) -> int:
    result = 0
    for b in data:
        result ^= b
    return result


# ---------------------------------------------------------------------------
# TAP writer
# ---------------------------------------------------------------------------

def _make_header_block(
    block_type: int,
    filename: str,
    data_len: int,
    param1: int,
    param2: int,
) -> bytes:
    """Build a TAP header block (flag=0x00, type byte + 10-char name + 2+2+2 params)."""
    name_bytes = filename.encode("ascii")[:10].ljust(10, b" ")
    payload = bytes([block_type]) + name_bytes
    payload += data_len.to_bytes(2, "little")
    payload += param1.to_bytes(2, "little")
    payload += param2.to_bytes(2, "little")

    flag = 0x00
    content = bytes([flag]) + payload
    checksum = _xor_checksum(content)
    block = content + bytes([checksum])
    return len(block).to_bytes(2, "little") + block


def _make_data_block(payload: bytes) -> bytes:
    """Build a TAP data block (flag=0xFF + raw bytes)."""
    flag = 0xFF
    content = bytes([flag]) + payload
    checksum = _xor_checksum(content)
    block = content + bytes([checksum])
    return len(block).to_bytes(2, "little") + block


def write_tap(
    binary_data: bytes,
    filename: str,
    start_addr: int,
    output_path: str,
) -> None:
    """Write a ZX Spectrum TAP file wrapping binary_data.

    Produces two blocks:
      - Header block: type=3 (Code), filename (10 chars), length, start_addr, 0x8000
      - Data block:   raw binary_data

    Args:
        binary_data: The machine code bytes to wrap.
        filename:    TAP filename (truncated/padded to 10 chars).
        start_addr:  Load address in Spectrum RAM (e.g. 0x8000).
        output_path: Path to write the .tap file.
    """
    tap = bytearray()
    tap += _make_header_block(3, filename, len(binary_data), start_addr, 0x8000)
    tap += _make_data_block(binary_data)
    Path(output_path).write_bytes(bytes(tap))


def _int5(value: int) -> bytes:
    """Spectrum 5-byte integer: 0x00 + sign + lo + hi + 0x00."""
    sign = 0xFF if value < 0 else 0x00
    unsigned = value & 0xFFFF
    return bytes([0x00, sign, unsigned & 0xFF, (unsigned >> 8) & 0xFF, 0x00])


def _basic_line(line_num: int, body: bytes) -> bytes:
    """Wrap a tokenised BASIC line body with number + length + CR."""
    terminated = body + bytes([0x0D])
    return (
        line_num.to_bytes(2, "big")
        + len(terminated).to_bytes(2, "little")
        + terminated
    )


def _make_basic_loader(start_addr: int) -> bytes:
    """Build tokenised BASIC auto-loader:
        10 CLEAR <addr-1>: LOAD ""CODE: RANDOMIZE USR <addr>

    Spectrum BASIC tokens: CLEAR=0xFD, LOAD=0xEF, CODE=0xAF,
    RANDOMIZE=0xF9, USR=0xC0, colon=0x3A, quote=0x22.
    Number literal = ASCII digits + 0x0E + 5-byte integer.
    """
    clear_val = start_addr - 1
    clear_str = str(clear_val).encode("ascii")
    addr_str = str(start_addr).encode("ascii")

    body = bytearray()
    # CLEAR <addr-1>
    body += bytes([0xFD])
    body += clear_str + bytes([0x0E]) + _int5(clear_val)
    # : LOAD ""CODE
    body += bytes([0x3A, 0xEF, 0x22, 0x22, 0xAF])
    # : RANDOMIZE USR <addr>
    body += bytes([0x3A, 0xF9, 0xC0])
    body += addr_str + bytes([0x0E]) + _int5(start_addr)

    return _basic_line(10, bytes(body))


def write_spectrum_tap(
    binary_data: bytes,
    filename: str,
    start_addr: int,
    output_path: str,
) -> None:
    """Write a full Spectrum TAP: BASIC auto-loader + CODE block.

    Produces four blocks:
      1. Header (type=0 Program): BASIC loader, autostart line 10
      2. Data: tokenised BASIC — 10 RANDOMIZE USR <start_addr>
      3. Header (type=3 Code): machine code at start_addr
      4. Data: raw binary_data

    The ROM auto-runs line 10, which calls USR to jump into the code.

    Args:
        binary_data: The machine code bytes.
        filename:    TAP filename (truncated/padded to 10 chars).
        start_addr:  Load address for the CODE block (e.g. 32768).
        output_path: Path to write the .tap file.
    """
    basic = _make_basic_loader(start_addr)

    tap = bytearray()
    # Block 1+2: BASIC program — type=0, param1=autostart line (10), param2=length
    tap += _make_header_block(0, filename, len(basic), 10, len(basic))
    tap += _make_data_block(basic)
    # Block 3+4: CODE block
    tap += _make_header_block(3, filename, len(binary_data), start_addr, 0x8000)
    tap += _make_data_block(binary_data)
    Path(output_path).write_bytes(bytes(tap))


# ---------------------------------------------------------------------------
# TAP reader
# ---------------------------------------------------------------------------

def read_tap(tap_path: str) -> list[dict]:
    """Parse a TAP file into a list of block dicts.

    Each dict has:
      - flag (int): 0x00 = header, 0xFF = data
      - payload (bytes): block content (flag and checksum stripped)
      - checksum (int): stored XOR byte
      - valid (bool): whether stored checksum matches computed
      - For header blocks (flag == 0x00):
          - type (int): block type (3 = Code)
          - name (str): 10-char filename stripped of trailing spaces
          - data_len (int): expected data length
          - param1 (int): start address (for Code blocks)
          - param2 (int): always 0x8000 for Code blocks
    """
    data = Path(tap_path).read_bytes()
    blocks: list[dict] = []
    pos = 0

    while pos < len(data):
        if pos + 2 > len(data):
            break
        length = int.from_bytes(data[pos:pos + 2], "little")
        pos += 2
        if pos + length > len(data):
            break
        block_data = data[pos:pos + length]
        pos += length

        flag = block_data[0]
        payload = block_data[1:-1]
        stored_checksum = block_data[-1]
        computed_checksum = _xor_checksum(block_data[:-1])

        block: dict = {
            "flag": flag,
            "payload": payload,
            "checksum": stored_checksum,
            "valid": stored_checksum == computed_checksum,
        }

        if flag == 0x00 and len(payload) >= 17:
            block["type"] = payload[0]
            block["name"] = payload[1:11].decode("ascii", errors="replace").rstrip()
            block["data_len"] = int.from_bytes(payload[11:13], "little")
            block["param1"] = int.from_bytes(payload[13:15], "little")
            block["param2"] = int.from_bytes(payload[15:17], "little")

        blocks.append(block)

    return blocks


def extract_code_blocks(tap_path: str) -> list[dict]:
    """Return only Code (type=3) header+data pairs from a TAP.

    Returns a list of dicts with:
      - name (str)
      - start_addr (int)
      - data (bytes)
    """
    blocks = read_tap(tap_path)
    results = []
    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk["flag"] == 0x00 and blk.get("type") == 3 and i + 1 < len(blocks):
            data_block = blocks[i + 1]
            if data_block["flag"] == 0xFF:
                results.append({
                    "name": blk["name"],
                    "start_addr": blk["param1"],
                    "data": data_block["payload"],
                })
                i += 2
                continue
        i += 1
    return results


# ---------------------------------------------------------------------------
# Disassembler
# ---------------------------------------------------------------------------

def disassemble(code_bytes: bytes, start_addr: int = 0x8000) -> str:
    """Disassemble Z80 machine code bytes to annotated assembly text.

    Uses z80dis if available, otherwise falls back to a minimal built-in
    disassembler covering the most common opcodes.

    Args:
        code_bytes:  Raw Z80 machine code.
        start_addr:  Load address (used for address labels in output).

    Returns:
        Multi-line string of annotated Z80 assembly.
    """
    try:
        return _disassemble_z80dis(code_bytes, start_addr)
    except ImportError:
        return _disassemble_minimal(code_bytes, start_addr)


def _disassemble_z80dis(code_bytes: bytes, start_addr: int) -> str:
    """Disassemble using the z80dis library."""
    from z80dis import z80  # type: ignore[import]

    lines = []
    offset = 0
    buf = bytes(code_bytes)

    while offset < len(buf):
        addr = start_addr + offset
        remaining = buf[offset:]

        try:
            decoded = z80.decode(remaining, addr)
            if decoded.status.name != "OK" or decoded.len == 0:
                mnemonic = f"DB 0x{remaining[0]:02X}"
                step = 1
            else:
                mnemonic = z80.disasm(remaining, addr)
                step = decoded.len
        except Exception:
            mnemonic = f"DB 0x{remaining[0]:02X}"
            step = 1

        raw_bytes = " ".join(f"{b:02X}" for b in buf[offset:offset + step])
        lines.append(f"{addr:04X}  {raw_bytes:<12}  {mnemonic}")
        offset += step

    return "\n".join(lines)


# Minimal single-byte opcode table for fallback disassembler
_SIMPLE_OPCODES: dict[int, str] = {
    0x00: "NOP",
    0x76: "HALT",
    0x3F: "CCF",
    0x37: "SCF",
    0x2F: "CPL",
    0x3C: "INC A",
    0x04: "INC B",
    0x0C: "INC C",
    0x14: "INC D",
    0x1C: "INC E",
    0x24: "INC H",
    0x2C: "INC L",
    0x3D: "DEC A",
    0x05: "DEC B",
    0x0D: "DEC C",
    0x15: "DEC D",
    0x1D: "DEC E",
    0x25: "DEC H",
    0x2D: "DEC L",
    0xAF: "XOR A",
    0xB7: "OR A",
    0xA7: "AND A",
    0x87: "ADD A,A",
    0x80: "ADD A,B",
    0x81: "ADD A,C",
    0x82: "ADD A,D",
    0x83: "ADD A,E",
    0x84: "ADD A,H",
    0x85: "ADD A,L",
    0x90: "SUB B",
    0x91: "SUB C",
    0x97: "SUB A",
    0x78: "LD A,B",
    0x79: "LD A,C",
    0x7A: "LD A,D",
    0x7B: "LD A,E",
    0x7C: "LD A,H",
    0x7D: "LD A,L",
    0x7F: "LD A,A",
    0x47: "LD B,A",
    0x4F: "LD C,A",
    0x57: "LD D,A",
    0x5F: "LD E,A",
    0x67: "LD H,A",
    0x6F: "LD L,A",
    0xC9: "RET",
    0xE9: "JP (HL)",
    0xF3: "DI",
    0xFB: "EI",
    0x08: "EX AF,AF'",
    0xD9: "EXX",
    0xEB: "EX DE,HL",
}


def _disassemble_minimal(code_bytes: bytes, start_addr: int) -> str:
    """Fallback disassembler: handles common single/two/three-byte opcodes."""
    lines = []
    i = 0
    buf = bytes(code_bytes)
    n = len(buf)

    while i < n:
        addr = start_addr + i
        op = buf[i]

        if op in _SIMPLE_OPCODES:
            mnemonic = _SIMPLE_OPCODES[op]
            raw = f"{op:02X}"
            i += 1
        elif op == 0x3E and i + 1 < n:       # LD A, n
            imm = buf[i + 1]
            mnemonic = f"LD A,0x{imm:02X}"
            raw = f"{op:02X} {imm:02X}"
            i += 2
        elif op == 0x06 and i + 1 < n:       # LD B, n
            imm = buf[i + 1]
            mnemonic = f"LD B,0x{imm:02X}"
            raw = f"{op:02X} {imm:02X}"
            i += 2
        elif op == 0x0E and i + 1 < n:       # LD C, n
            imm = buf[i + 1]
            mnemonic = f"LD C,0x{imm:02X}"
            raw = f"{op:02X} {imm:02X}"
            i += 2
        elif op == 0xD6 and i + 1 < n:       # SUB n
            imm = buf[i + 1]
            mnemonic = f"SUB 0x{imm:02X}"
            raw = f"{op:02X} {imm:02X}"
            i += 2
        elif op == 0xC6 and i + 1 < n:       # ADD A, n
            imm = buf[i + 1]
            mnemonic = f"ADD A,0x{imm:02X}"
            raw = f"{op:02X} {imm:02X}"
            i += 2
        elif op == 0x18 and i + 1 < n:       # JR e
            disp = buf[i + 1]
            signed = disp if disp < 128 else disp - 256
            target = addr + 2 + signed
            mnemonic = f"JR 0x{target:04X}"
            raw = f"{op:02X} {disp:02X}"
            i += 2
        elif op == 0x20 and i + 1 < n:       # JR NZ, e
            disp = buf[i + 1]
            signed = disp if disp < 128 else disp - 256
            target = addr + 2 + signed
            mnemonic = f"JR NZ,0x{target:04X}"
            raw = f"{op:02X} {disp:02X}"
            i += 2
        elif op == 0x28 and i + 1 < n:       # JR Z, e
            disp = buf[i + 1]
            signed = disp if disp < 128 else disp - 256
            target = addr + 2 + signed
            mnemonic = f"JR Z,0x{target:04X}"
            raw = f"{op:02X} {disp:02X}"
            i += 2
        elif op == 0xC3 and i + 2 < n:       # JP nn
            lo, hi = buf[i + 1], buf[i + 2]
            target = lo | (hi << 8)
            mnemonic = f"JP 0x{target:04X}"
            raw = f"{op:02X} {lo:02X} {hi:02X}"
            i += 3
        elif op == 0xCD and i + 2 < n:       # CALL nn
            lo, hi = buf[i + 1], buf[i + 2]
            target = lo | (hi << 8)
            mnemonic = f"CALL 0x{target:04X}"
            raw = f"{op:02X} {lo:02X} {hi:02X}"
            i += 3
        elif op == 0x21 and i + 2 < n:       # LD HL, nn
            lo, hi = buf[i + 1], buf[i + 2]
            mnemonic = f"LD HL,0x{lo | (hi << 8):04X}"
            raw = f"{op:02X} {lo:02X} {hi:02X}"
            i += 3
        elif op == 0x11 and i + 2 < n:       # LD DE, nn
            lo, hi = buf[i + 1], buf[i + 2]
            mnemonic = f"LD DE,0x{lo | (hi << 8):04X}"
            raw = f"{op:02X} {lo:02X} {hi:02X}"
            i += 3
        elif op == 0x01 and i + 2 < n:       # LD BC, nn
            lo, hi = buf[i + 1], buf[i + 2]
            mnemonic = f"LD BC,0x{lo | (hi << 8):04X}"
            raw = f"{op:02X} {lo:02X} {hi:02X}"
            i += 3
        else:
            mnemonic = f"DB 0x{op:02X}"
            raw = f"{op:02X}"
            i += 1

        lines.append(f"{addr:04X}  {raw:<12}  {mnemonic}")

    return "\n".join(lines)
