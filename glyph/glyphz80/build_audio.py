"""CLI entry point: convert glyph-audio JSON to Z80 beeper TAP file.

Usage:
    python glyph/glyphz80/build_audio.py assets/a1.json --output assets/a1.tap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "glyph"))
sys.path.insert(0, str(ROOT / "incant"))

from glyphz80.beeper import compile_glyph_audio, load_spec
from incant.tap import write_spectrum_tap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert glyph-audio JSON to Z80 beeper TAP file"
    )
    parser.add_argument("input", help="Path to glyph-audio JSON spec")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .tap path (default: input with .tap extension)",
    )
    parser.add_argument(
        "--base-addr",
        type=lambda x: int(x, 0),
        default=0x8000,
        help="Load address (default: 0x8000 = 32768)",
    )
    parser.add_argument(
        "--name",
        default="glyph",
        help="TAP filename header (max 10 chars, default: glyph)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".tap")

    spec = load_spec(input_path)
    binary = compile_glyph_audio(spec, base_addr=args.base_addr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_spectrum_tap(binary, args.name, args.base_addr, str(output_path))

    pieces = spec.get("pieces", [])
    total_notes = sum(len(p.get("notes", [])) for p in pieces)
    print(f"Compiled {total_notes} notes -> {len(binary)} bytes code")
    print(f"Load address: 0x{args.base_addr:04X} ({args.base_addr})")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
