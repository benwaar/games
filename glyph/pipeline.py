"""pipeline.py — image/audio → debug.wasm + debug.tap

Two parallel paths from one input:

Image (.png/.jpg):
  - WASM: full-colour RGBA at original resolution → wasm/debug.wasm
  - TAP:  Spectrum-quantised (256x192, k-means) → zxw/www/tap/debug.tap

Audio (.json glyph-audio):
  - WASM: freq/duration arrays embedded as WAT data segments → wasm/debug.wasm
  - TAP:  Z80 beeper binary → zxw/www/tap/debug.tap

Usage:
    python glyph/pipeline.py glyph/assets/i1.png
    python glyph/pipeline.py glyph/assets/a1.json
"""

import argparse
import json
import struct
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
  (func (export "type") (result i32) (i32.const 1))
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


def _f64_hex(value: float) -> str:
    """Encode an f64 as WAT hex escape string (8 bytes, little-endian)."""
    return "".join(f"\\{b:02x}" for b in struct.pack("<d", value))


def audio_to_wat(freqs: list[float], durations: list[float]) -> str:
    """Generate a WAT module with freq/duration arrays as data segments.

    Layout:
      offset 0              — f64[n] frequencies  (n * 8 bytes)
      offset n * 8          — f64[n] durations    (n * 8 bytes)

    Exports:
      type()         → i32  2  (bit 1 = has audio, no image)
      audio_count()  → i32  n
      audio_offset() → i32  0
      memory
    """
    count = len(freqs)
    freq_bytes = "".join(_f64_hex(f) for f in freqs)
    dur_bytes = "".join(_f64_hex(d) for d in durations)
    dur_offset = count * 8
    pages = max(1, (count * 8 * 2 + 65535) // 65536)

    return f"""(module
  (memory (export "memory") {pages})
  (data (i32.const 0) "{freq_bytes}")
  (data (i32.const {dur_offset}) "{dur_bytes}")
  (func (export "type") (result i32) (i32.const 2))
  (func (export "audio_count") (result i32) (i32.const {count}))
  (func (export "audio_offset") (result i32) (i32.const 0))
)
"""


def build_audio_wasm(spec: dict) -> None:
    """glyph-audio JSON → freq/duration arrays → WAT → WASM."""
    from glyphz80.beeper import note_to_freq

    freqs: list[float] = []
    durations: list[float] = []

    for piece in spec.get("pieces", []):
        if piece.get("kind") != "melody":
            continue
        for entry in piece.get("notes", []):
            freqs.append(note_to_freq(entry["note"]))
            durations.append(float(entry["ms"]))

    count = len(freqs)
    print(f"WASM: {count} notes → WAT data segments")

    wat = audio_to_wat(freqs, durations)
    WAT_OUT.write_text(wat)

    result = subprocess.run(
        ["wat2wasm", str(WAT_OUT), "-o", str(WASM_OUT)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"wat2wasm failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Written {WASM_OUT} ({WASM_OUT.stat().st_size} bytes)")


def build_audio_tap(spec: dict) -> None:
    """glyph-audio JSON → Z80 beeper binary → TAP."""
    from glyphz80.beeper import compile_glyph_audio

    binary = compile_glyph_audio(spec)
    write_spectrum_tap(binary, "debug", 0x8000, str(TAP_OUT))
    print(f"Written {TAP_OUT} ({TAP_OUT.stat().st_size} bytes)")


# ── Combined image + audio pipeline ─────────────────────────────────


def combined_to_wat(
    rgba: bytes,
    width: int,
    height: int,
    freqs: list[float],
    durations: list[float],
) -> str:
    """Generate a WAT module with both pixel data and audio arrays embedded.

    Memory layout:
      offset 0                     — RGBA pixels (width * height * 4 bytes)
      offset width * height * 4    — f64[n] frequencies (n * 8 bytes)
      offset audio_offset + n * 8  — f64[n] durations   (n * 8 bytes)

    Exports:
      type()         → i32  3  (bits 0+1 = image + audio)
      width()        → i32
      height()       → i32
      render()            nop — pixels already in memory
      audio_count()  → i32  n
      audio_offset() → i32  pixel end offset
      memory
    """
    count = len(freqs)
    pixel_bytes = len(rgba)
    audio_offset = pixel_bytes  # audio starts immediately after pixels

    pixel_hex = "".join(f"\\{b:02x}" for b in rgba)
    freq_hex = "".join(_f64_hex(f) for f in freqs)
    dur_hex = "".join(_f64_hex(d) for d in durations)
    dur_offset = audio_offset + count * 8

    total_bytes = pixel_bytes + count * 8 * 2
    pages = (total_bytes + 65535) // 65536

    return f"""(module
  (memory (export "memory") {pages})
  (data (i32.const 0) "{pixel_hex}")
  (data (i32.const {audio_offset}) "{freq_hex}")
  (data (i32.const {dur_offset}) "{dur_hex}")
  (func (export "type") (result i32) (i32.const 3))
  (func (export "render") (nop))
  (func (export "width") (result i32) (i32.const {width}))
  (func (export "height") (result i32) (i32.const {height}))
  (func (export "audio_count") (result i32) (i32.const {count}))
  (func (export "audio_offset") (result i32) (i32.const {audio_offset}))
)
"""


def build_combined_wasm(img: Image.Image, audio_spec: dict) -> None:
    """Image + audio → combined WAT → WASM."""
    from glyphz80.beeper import note_to_freq

    rgba, w, h = image_to_rgba(img)

    freqs: list[float] = []
    durations: list[float] = []
    for piece in audio_spec.get("pieces", []):
        if piece.get("kind") != "melody":
            continue
        for entry in piece.get("notes", []):
            freqs.append(note_to_freq(entry["note"]))
            durations.append(float(entry["ms"]))

    count = len(freqs)
    print(f"WASM: {w}x{h} image + {count} audio notes → combined WAT")

    wat = combined_to_wat(rgba, w, h, freqs, durations)
    WAT_OUT.write_text(wat)

    result = subprocess.run(
        ["wat2wasm", str(WAT_OUT), "-o", str(WASM_OUT)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"wat2wasm failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Written {WASM_OUT} ({WASM_OUT.stat().st_size} bytes)")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Image/audio → debug.wasm + debug.tap")
    parser.add_argument("input", help="Input image (.png/.jpg) or audio spec (.json)")
    parser.add_argument(
        "--audio",
        metavar="JSON",
        help="Audio spec (.json) to combine with an image input (produces type=3 WASM)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg"):
        img = Image.open(path)
        print(f"Image: {path.name} ({img.size[0]}x{img.size[1]} {img.mode})")
        if args.audio:
            audio_path = Path(args.audio)
            spec = json.loads(audio_path.read_text())
            if spec.get("type") != "glyph-audio":
                print(f"Unknown JSON type: {spec.get('type')}", file=sys.stderr)
                sys.exit(1)
            notes = sum(len(p.get("notes", [])) for p in spec.get("pieces", []))
            print(f"Audio: {audio_path.name} ({notes} notes)")
            build_combined_wasm(img, spec)
            build_image_tap(img)
        else:
            build_image_wasm(img)
            build_image_tap(img)
    elif suffix == ".json":
        if args.audio:
            print("--audio flag is only valid with an image input", file=sys.stderr)
            sys.exit(1)
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
