"""build_scr.py — assemble scr_loader.asm and concatenate with an SCR file.

Usage:
    python build_scr.py <scr_file> [--output <out.bin>]

The output binary loads at 0x8000 and, when executed, copies the SCR data
into screen memory at 0x4000–0x5AFF then halts.
"""

import argparse
import sys
from pathlib import Path

import z80 as z80lib

SCR_SIZE = 6912
ASM_PATH = Path(__file__).parent / "scr_loader.asm"


def assemble_loader() -> bytes:
    """Assemble scr_loader.asm and return the machine-code bytes."""
    source = ASM_PATH.read_text()
    asm = z80lib.Asm()
    sf = z80lib.SourceFile(str(ASM_PATH), source.lower())
    code = asm.assemble(sf)
    code.resolve()
    chunks = code.encode()
    if not chunks:
        raise RuntimeError("Assembly produced no output")
    _, data = chunks[0]
    return data


def build(scr_path: Path, output_path: Path) -> None:
    """Concatenate assembled loader bytes with SCR data and write output."""
    scr_bytes = scr_path.read_bytes()
    if len(scr_bytes) != SCR_SIZE:
        raise ValueError(
            f"SCR file must be exactly {SCR_SIZE} bytes, got {len(scr_bytes)}"
        )

    loader_bytes = assemble_loader()
    combined = loader_bytes + scr_bytes
    output_path.write_bytes(combined)
    print(
        f"Written {len(combined)} bytes to {output_path} "
        f"(loader={len(loader_bytes)}B, scr={len(scr_bytes)}B)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SCR loader binary")
    parser.add_argument("scr_file", type=Path, help="Input .scr file (6912 bytes)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scr_loader.bin"),
        help="Output binary path (default: scr_loader.bin)",
    )
    args = parser.parse_args(argv)

    try:
        build(args.scr_file, args.output)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
