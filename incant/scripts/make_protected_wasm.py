"""Generate a self-verifying WASM binary for the zxw browser demo.

The module XOR-hashes its own data segment at startup.
  intact  → run() returns 42
  tampered → unreachable trap fires

Output: zxw/www/protect.wasm

Usage:
    cd incant
    source .venv/bin/activate
    python scripts/make_protected_wasm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from incant.protect import generate_self_verify_wat, assemble_wat, _xor_checksum

PAYLOAD = b"ZX Spectrum binary protection demo payload v1"
EXPECTED_RESULT = 42
OUT_WASM = Path(__file__).parent.parent.parent / "zxw" / "www" / "protect.wasm"


def main() -> None:
    wat = generate_self_verify_wat(PAYLOAD, expected_result=EXPECTED_RESULT)
    wasm_path = assemble_wat(wat)
    wasm_bytes = wasm_path.read_bytes()

    OUT_WASM.parent.mkdir(parents=True, exist_ok=True)
    OUT_WASM.write_bytes(wasm_bytes)

    checksum = _xor_checksum(PAYLOAD)
    print(f"Written: {OUT_WASM} ({len(wasm_bytes)} bytes)")
    print(f"Payload: {PAYLOAD!r}")
    print(f"XOR checksum: 0x{checksum:02X}")
    print(f"Expected result: {EXPECTED_RESULT}")
    print()
    print("Tamper test: flip first payload byte in the binary")
    print(f"  First payload byte offset in WASM: search for {PAYLOAD[:4]!r}")


if __name__ == "__main__":
    main()
