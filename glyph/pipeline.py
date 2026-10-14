"""pipeline.py — image/audio → debug.wasm + debug.tap

Two parallel paths from one input:

Image (.png/.jpg):
  - WASM: full-colour RGBA at original resolution → wasm/debug.wasm
  - TAP:  Spectrum-quantised (256x192, k-means) → zxw/www/tap/debug.tap

Audio (.json glyph-audio):
  - WASM: Web Audio oscillator via Rust + wasm-pack → wasm/debug.wasm
  - TAP:  Z80 beeper binary → zxw/www/tap/debug.tap

Usage:
    python glyph/pipeline.py glyph/assets/i1.png
    python glyph/pipeline.py glyph/assets/a1.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "glyph"))
sys.path.insert(0, str(ROOT / "incant"))

from scr.quantise import quantise_image
from scr.pack import pack_scr
from incant.tap import write_spectrum_tap

WASM_OUT = ROOT / "wasm" / "debug.wasm"
WAT_OUT = ROOT / "wasm" / "debug.wat"
TAP_OUT = ROOT / "zxw" / "www" / "tap" / "debug.tap"
GLYPHWASM_DIR = ROOT / "glyph" / "glyphwasm"


# ── Image pipeline ──────────────────────────────────────────────────


def image_to_rgba(img: Image.Image) -> tuple[bytes, int, int]:
    """Convert a PIL Image to raw RGBA bytes at its original resolution."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    return bytes(np.array(rgba, dtype=np.uint8).tobytes()), w, h


def rgba_to_wat(rgba: bytes, width: int, height: int) -> str:
    """Generate a WAT module with pixel data embedded as a data segment."""
    hex_str = "".join(f"\\{b:02x}" for b in rgba)
    pages = (len(rgba) + 65535) // 65536

    return f"""(module
  (memory (export "memory") {pages})
  (data (i32.const 0) "{hex_str}")
  (func (export "render") (nop))
  (func (export "width") (result i32) (i32.const {width}))
  (func (export "height") (result i32) (i32.const {height}))
)
"""


def build_image_wasm(img: Image.Image) -> None:
    """Image → RGBA → WAT → WASM at original resolution."""
    rgba, w, h = image_to_rgba(img)
    print(f"WASM: {w}x{h} ({len(rgba)} bytes RGBA)")

    wat = rgba_to_wat(rgba, w, h)
    WAT_OUT.write_text(wat)

    result = subprocess.run(
        ["wat2wasm", str(WAT_OUT), "-o", str(WASM_OUT)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"wat2wasm failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Written {WASM_OUT} ({WASM_OUT.stat().st_size} bytes)")


def build_image_tap(img: Image.Image) -> None:
    """Image → SCR (quantise) → Z80 binary → TAP."""
    from glyphz80.build_scr import build

    quantised = quantise_image(img)
    print(f"TAP: quantised to 256x192 Spectrum palette")

    with tempfile.NamedTemporaryFile(suffix=".scr", delete=False) as tmp:
        scr_path = tmp.name
    pack_scr(quantised, scr_path)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        bin_path = Path(tmp.name)
    build(Path(scr_path), bin_path)
    binary = bin_path.read_bytes()

    write_spectrum_tap(binary, "debug", 0x8000, str(TAP_OUT))
    print(f"Written {TAP_OUT} ({TAP_OUT.stat().st_size} bytes)")

    Path(scr_path).unlink()
    bin_path.unlink()


# ── Audio pipeline ──────────────────────────────────────────────────


def generate_data_rs(spec: dict) -> str:
    """Generate Rust data module from glyph-audio JSON spec."""
    from glyphz80.beeper import note_to_freq

    freqs: list[float] = []
    durations: list[float] = []

    for piece in spec.get("pieces", []):
        if piece.get("kind") != "melody":
            continue
        for entry in piece.get("notes", []):
            freqs.append(note_to_freq(entry["note"]))
            durations.append(float(entry["ms"]))

    lines = [
        f"pub const NOTE_COUNT: usize = {len(freqs)};",
        "",
        f"pub const FREQS: [f64; {len(freqs)}] = [",
    ]
    for f in freqs:
        lines.append(f"    {f:.4f},")
    lines.append("];")
    lines.append("")
    lines.append(f"pub const DURATIONS_MS: [f64; {len(durations)}] = [")
    for d in durations:
        lines.append(f"    {d:.1f},")
    lines.append("];")
    lines.append("")

    return "\n".join(lines)


def build_audio_wasm(spec: dict) -> None:
    """glyph-audio JSON → data.rs → wasm-pack → debug.wasm."""
    data_rs = generate_data_rs(spec)
    data_rs_path = GLYPHWASM_DIR / "src" / "data.rs"
    data_rs_path.write_text(data_rs)
    print(f"Generated {data_rs_path}")

    result = subprocess.run(
        ["wasm-pack", "build", "--target", "web", "--out-dir", "pkg"],
        cwd=str(GLYPHWASM_DIR),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"wasm-pack failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    pkg_wasm = GLYPHWASM_DIR / "pkg" / "glyph_audio_bg.wasm"
    if not pkg_wasm.exists():
        wasm_files = list((GLYPHWASM_DIR / "pkg").glob("*.wasm"))
        if not wasm_files:
            print("No .wasm file found in pkg/", file=sys.stderr)
            sys.exit(1)
        pkg_wasm = wasm_files[0]

    import shutil
    shutil.copy2(pkg_wasm, WASM_OUT)
    print(f"Written {WASM_OUT} ({WASM_OUT.stat().st_size} bytes)")

    # Also copy the JS glue for the HTML to import
    pkg_js = pkg_wasm.with_suffix("").with_suffix(".js")
    if not pkg_js.exists():
        js_files = list((GLYPHWASM_DIR / "pkg").glob("*.js"))
        if js_files:
            pkg_js = js_files[0]
    if pkg_js.exists():
        js_out = WASM_OUT.parent / pkg_js.name
        shutil.copy2(pkg_js, js_out)
        print(f"Written {js_out}")


def build_audio_tap(spec: dict) -> None:
    """glyph-audio JSON → Z80 beeper binary → TAP."""
    from glyphz80.beeper import compile_glyph_audio

    binary = compile_glyph_audio(spec)
    write_spectrum_tap(binary, "debug", 0x8000, str(TAP_OUT))
    print(f"Written {TAP_OUT} ({TAP_OUT.stat().st_size} bytes)")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Image/audio → debug.wasm + debug.tap")
    parser.add_argument("input", help="Input image (.png/.jpg) or audio spec (.json)")
    args = parser.parse_args()

    path = Path(args.input)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg"):
        img = Image.open(path)
        print(f"Image: {path.name} ({img.size[0]}x{img.size[1]} {img.mode})")
        build_image_wasm(img)
        build_image_tap(img)
    elif suffix == ".json":
        spec = json.loads(path.read_text())
        if spec.get("type") != "glyph-audio":
            print(f"Unknown JSON type: {spec.get('type')}", file=sys.stderr)
            sys.exit(1)
        notes = sum(len(p.get("notes", [])) for p in spec.get("pieces", []))
        print(f"Audio: {path.name} ({notes} notes)")
        build_audio_wasm(spec)
        build_audio_tap(spec)
    else:
        print(f"Unknown file type: {suffix}", file=sys.stderr)
        sys.exit(1)

    print("\nDone. Refresh :8999 (WASM) and :8998 (TAP).")


if __name__ == "__main__":
    main()
