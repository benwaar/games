"""Generate a self-verifying Z80 TAP file for the zxw demo.

Payload: writes 'PROTECTED' to screen memory (top-left) then loops.
The binary is wrapped with an XOR checksum verifier prefix.
If any payload byte is tampered, the verifier jumps to HALT instead.

Output: zxw/www/tap/protected.tap

Usage:
    cd incant
    source .venv/bin/activate
    python scripts/make_protected_tap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from incant.protect import wrap_z80_checksum, VERIFY_PREFIX_LEN
from incant.tap import write_tap

LOAD_ADDR = 0x8000
OUT_TAP = Path(__file__).parent.parent.parent / "zxw" / "www" / "tap" / "protected.tap"

# ---------------------------------------------------------------------------
# Payload: write "PROTECTED" chars to top-left screen memory (0x4000+)
# and then loop forever (JR $ = jump to self).
#
# Spectrum character codes: P=80 R=82 O=79 T=84 E=69 C=67 T=84 E=69 D=68
# Screen memory: 0x4000 = top-left pixel row. For text we use 0x5800 (attrs)
# but for simplicity we write to 0x4000..0x4008 directly — visible as bright
# pixel rows, not text, but clearly "something happened".
#
# To show a readable message: use LD (HL),n for each screen address.
# This is a demo — we just write a visible pattern.
# ---------------------------------------------------------------------------

def build_payload() -> bytes:
    # Write the byte 0xFF (all pixels set) to 9 consecutive screen bytes
    # starting at 0x4000 — makes a bright white stripe across the top.
    # Then loop forever with JR $ (0x18, 0xFE).
    ops = []
    addr = 0x4000
    for i in range(9):
        a = addr + i
        ops += [
            0x3E, 0xFF,                          # LD A, 0xFF
            0x32, a & 0xFF, (a >> 8) & 0xFF,     # LD (addr), A
        ]
    ops += [0x18, 0xFE]  # JR $ (infinite loop)
    return bytes(ops)


def main() -> None:
    payload = build_payload()
    protected = wrap_z80_checksum(payload, LOAD_ADDR)

    tamper_addr = LOAD_ADDR + VERIFY_PREFIX_LEN   # first byte of payload
    tamper_byte = protected[VERIFY_PREFIX_LEN] ^ 0x01  # flipped value

    OUT_TAP.parent.mkdir(parents=True, exist_ok=True)
    write_tap(protected, "protected", LOAD_ADDR, str(OUT_TAP))

    print(f"Written: {OUT_TAP}")
    print(f"Load address:  0x{LOAD_ADDR:04X}")
    print(f"Payload bytes: {len(payload)}")
    print(f"Total binary:  {len(protected)} bytes")
    print()
    print("To run:   load 'protected' in zxw, then RANDOMIZE USR 32768")
    print()
    print("To tamper (paste in DevTools console while emulator runs):")
    print(f"  zxw.poke(0x{tamper_addr:04X}, 0x{tamper_byte:02X})")
    print()
    print("Expected: intact → white stripe appears; tampered → emulator halts")


if __name__ == "__main__":
    main()
