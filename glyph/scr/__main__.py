"""
CLI entry point for glyph.scr.

Usage:
  python -m glyph.scr convert input.png --output output.scr
  python -m glyph.scr render output.scr --output preview.png
  python -m glyph.scr compare input.png output.scr --output comparison.png
"""
import argparse
import sys
from pathlib import Path

from PIL import Image


def cmd_convert(args: argparse.Namespace) -> None:
    from .quantise import quantise_image
    from .pack import pack_scr

    img = Image.open(args.input)
    print(f"Loaded {args.input} ({img.size[0]}x{img.size[1]} {img.mode})")

    quantised = quantise_image(img)
    out_path = args.output or str(Path(args.input).with_suffix(".scr"))
    pack_scr(quantised, out_path)
    print(f"Written {out_path} (6912 bytes)")


def cmd_render(args: argparse.Namespace) -> None:
    from .render import render_scr

    rendered = render_scr(args.input)
    out_path = args.output or str(Path(args.input).with_suffix(".png"))
    rendered.save(out_path)
    print(f"Rendered to {out_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    from .render import render_scr

    original = Image.open(args.input).convert("RGB").resize((256, 192))
    rendered = render_scr(args.scr)

    out = Image.new("RGB", (512, 192))
    out.paste(original, (0, 0))
    out.paste(rendered, (256, 0))

    out_path = args.output or "comparison.png"
    out.save(out_path)
    print(f"Comparison saved to {out_path} (original left, SCR right)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m glyph.scr",
        description="ZX Spectrum SCR file tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="Convert image to .scr")
    p_convert.add_argument("input", help="Input image path")
    p_convert.add_argument("--output", "-o", help="Output .scr path")

    p_render = sub.add_parser("render", help="Render .scr to PNG")
    p_render.add_argument("input", help="Input .scr path")
    p_render.add_argument("--output", "-o", help="Output PNG path")

    p_compare = sub.add_parser("compare", help="Side-by-side comparison")
    p_compare.add_argument("input", help="Original image path")
    p_compare.add_argument("scr", help="SCR file path")
    p_compare.add_argument("--output", "-o", help="Output PNG path")

    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "render":
        cmd_render(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
