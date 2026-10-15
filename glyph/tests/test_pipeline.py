"""Tests for glyph/pipeline.py — audio_to_wat, rgba_to_wat, image_to_rgba."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "glyph"))

from pipeline import _a11y_segment, _f64_hex, audio_to_wat, combined_to_wat, image_to_rgba, rgba_to_wat


# ── Helpers ──────────────────────────────────────────────────────────


def _wat2wasm_available() -> bool:
    return subprocess.run(
        ["wat2wasm", "--version"], capture_output=True
    ).returncode == 0


def _compile_wat(wat: str) -> tuple[bool, str]:
    """Compile WAT string with wat2wasm. Returns (success, stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".wat", mode="w", delete=False) as f:
        f.write(wat)
        wat_path = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as f:
        wasm_path = Path(f.name)
    result = subprocess.run(
        ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
        capture_output=True,
        text=True,
    )
    wat_path.unlink(missing_ok=True)
    wasm_path.unlink(missing_ok=True)
    return result.returncode == 0, result.stderr


# ── audio_to_wat ─────────────────────────────────────────────────────


class TestAudioToWat:
    def test_returns_valid_wat_string(self):
        wat = audio_to_wat([440.0], [500.0])
        assert "(module" in wat

    def test_exports_type(self):
        wat = audio_to_wat([440.0], [500.0])
        assert '"type"' in wat

    def test_exports_audio_count(self):
        wat = audio_to_wat([440.0], [500.0])
        assert '"audio_count"' in wat

    def test_exports_audio_offset(self):
        wat = audio_to_wat([440.0], [500.0])
        assert '"audio_offset"' in wat

    def test_exports_memory(self):
        wat = audio_to_wat([440.0], [500.0])
        assert '"memory"' in wat

    def test_type_returns_2(self):
        """type() bitmask: bit 1 = audio only → value 2."""
        wat = audio_to_wat([440.0], [500.0])
        assert "i32.const 2" in wat

    def test_audio_count_correct(self):
        freqs = [440.0, 880.0, 220.0]
        durs = [100.0, 200.0, 300.0]
        wat = audio_to_wat(freqs, durs)
        assert f"i32.const {len(freqs)}" in wat

    def test_audio_count_single_note(self):
        wat = audio_to_wat([440.0], [500.0])
        assert "i32.const 1" in wat

    def test_f64_encoding_440hz(self):
        """440.0 Hz must be encoded as little-endian f64 bytes."""
        expected_hex = "".join(f"\\{b:02x}" for b in struct.pack("<d", 440.0))
        wat = audio_to_wat([440.0], [500.0])
        assert expected_hex in wat

    def test_f64_encoding_known_bytes(self):
        """struct.pack('<d', 440.0) == b'\\x00\\x00\\x00\\x00\\x00\\x1a\\x7b\\x40'."""
        raw = struct.pack("<d", 440.0)
        assert _f64_hex(440.0) == "".join(f"\\{b:02x}" for b in raw)

    def test_duration_encoded(self):
        dur = 250.0
        expected_hex = "".join(f"\\{b:02x}" for b in struct.pack("<d", dur))
        wat = audio_to_wat([440.0], [dur])
        assert expected_hex in wat

    def test_empty_arrays_produces_valid_module(self):
        """Empty freq/dur arrays: count=0 is a degenerate but allowed case."""
        wat = audio_to_wat([], [])
        assert "(module" in wat
        assert "i32.const 0" in wat  # audio_count = 0

    def test_memory_pages_minimum_1(self):
        """Even empty arrays must allocate at least 1 page."""
        wat = audio_to_wat([], [])
        assert "(memory" in wat
        # Grab the page count — must be >= 1
        import re
        match = re.search(r'\(memory \(export "memory"\) (\d+)\)', wat)
        assert match is not None
        assert int(match.group(1)) >= 1

    def test_memory_pages_scales_with_count(self):
        """8192 notes × 8 bytes × 2 arrays = 131072 bytes = 2 pages."""
        n = 8192
        freqs = [440.0] * n
        durs = [100.0] * n
        wat = audio_to_wat(freqs, durs)
        import re
        match = re.search(r'\(memory \(export "memory"\) (\d+)\)', wat)
        assert match is not None
        assert int(match.group(1)) >= 2

    @pytest.mark.skipif(not _wat2wasm_available(), reason="wat2wasm not found")
    def test_compiles_with_wat2wasm_single_note(self):
        wat = audio_to_wat([440.0], [500.0])
        ok, stderr = _compile_wat(wat)
        assert ok, f"wat2wasm failed:\n{stderr}"

    @pytest.mark.skipif(not _wat2wasm_available(), reason="wat2wasm not found")
    def test_compiles_with_wat2wasm_multi_note(self):
        freqs = [261.63, 293.66, 329.63, 349.23]
        durs = [250.0, 250.0, 250.0, 500.0]
        wat = audio_to_wat(freqs, durs)
        ok, stderr = _compile_wat(wat)
        assert ok, f"wat2wasm failed:\n{stderr}"

    @pytest.mark.skipif(not _wat2wasm_available(), reason="wat2wasm not found")
    def test_compiles_with_wat2wasm_empty(self):
        wat = audio_to_wat([], [])
        ok, stderr = _compile_wat(wat)
        assert ok, f"wat2wasm failed:\n{stderr}"


# ── rgba_to_wat ──────────────────────────────────────────────────────


class TestRgbaToWat:
    def test_type_returns_1(self):
        """type() bitmask: bit 0 = image only → value 1."""
        rgba = bytes([255, 0, 0, 255] * 4)  # 2x2 red image
        wat = rgba_to_wat(rgba, 2, 2)
        assert "i32.const 1" in wat

    def test_exports_type(self):
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        assert '"type"' in wat

    def test_exports_width(self):
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        assert '"width"' in wat

    def test_exports_height(self):
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        assert '"height"' in wat

    def test_exports_memory(self):
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        assert '"memory"' in wat

    def test_width_value_correct(self):
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        assert "i32.const 2" in wat

    def test_height_value_correct(self):
        rgba = bytes([0] * 32)
        wat = rgba_to_wat(rgba, 2, 4)
        assert "i32.const 4" in wat

    def test_contains_module(self):
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        assert "(module" in wat

    def test_2x2_data_segment_encoded(self):
        """A known pixel must appear as its hex escape in the data segment."""
        # RGBA = (0xAB, 0xCD, 0xEF, 0xFF)
        pixel = bytes([0xAB, 0xCD, 0xEF, 0xFF])
        rgba = pixel * 4  # 2×2
        wat = rgba_to_wat(rgba, 2, 2)
        assert "\\ab\\cd\\ef\\ff" in wat

    def test_memory_pages_256x256(self):
        """256×256×4 = 262144 bytes → needs 4 pages (262144 / 65536 = 4)."""
        rgba = bytes([0] * (256 * 256 * 4))
        wat = rgba_to_wat(rgba, 256, 256)
        import re
        match = re.search(r'\(memory \(export "memory"\) (\d+)\)', wat)
        assert match is not None
        assert int(match.group(1)) == 4

    def test_memory_pages_small(self):
        """2×2×4 = 16 bytes → 1 page is enough."""
        rgba = bytes([0] * 16)
        wat = rgba_to_wat(rgba, 2, 2)
        import re
        match = re.search(r'\(memory \(export "memory"\) (\d+)\)', wat)
        assert match is not None
        assert int(match.group(1)) == 1

    @pytest.mark.skipif(not _wat2wasm_available(), reason="wat2wasm not found")
    def test_compiles_2x2(self):
        rgba = bytes([255, 128, 0, 255] * 4)
        wat = rgba_to_wat(rgba, 2, 2)
        ok, stderr = _compile_wat(wat)
        assert ok, f"wat2wasm failed:\n{stderr}"


# ── image_to_rgba ────────────────────────────────────────────────────


class TestImageToRgba:
    def test_returns_bytes_and_dimensions(self):
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        raw, w, h = image_to_rgba(img)
        assert isinstance(raw, bytes)
        assert w == 4
        assert h == 4

    def test_byte_length_is_width_times_height_times_4(self):
        img = Image.new("RGB", (8, 6), color=(0, 255, 0))
        raw, w, h = image_to_rgba(img)
        assert len(raw) == w * h * 4

    def test_rgba_image_passthrough(self):
        """An RGBA source keeps its alpha channel."""
        img = Image.new("RGBA", (2, 2), color=(10, 20, 30, 128))
        raw, w, h = image_to_rgba(img)
        assert len(raw) == 2 * 2 * 4
        assert raw[3] == 128  # alpha of first pixel

    def test_pixel_values_correct_red(self):
        img = Image.new("RGB", (1, 1), color=(200, 100, 50))
        raw, w, h = image_to_rgba(img)
        assert w == 1 and h == 1
        r, g, b, a = raw[0], raw[1], raw[2], raw[3]
        assert r == 200
        assert g == 100
        assert b == 50
        assert a == 255  # RGB → RGBA fills alpha as 255

    def test_greyscale_converted_to_rgba(self):
        img = Image.new("L", (2, 2), color=128)
        raw, w, h = image_to_rgba(img)
        assert len(raw) == 2 * 2 * 4
        # Each channel of greyscale maps to R=G=B=128, A=255
        assert raw[0] == 128
        assert raw[3] == 255


# ── _a11y_segment ────────────────────────────────────────────────────


class TestA11ySegment:
    def test_a11y_segment_encoding(self):
        """Hex in data segment must round-trip back to original JSON."""
        import re
        nodes = [{"role": "img", "label": "test"}]
        fragment, size = _a11y_segment(nodes, 100)
        # Extract the hex string from the data segment line
        match = re.search(r'\(data \(i32\.const 100\) "([^"]+)"\)', fragment)
        assert match is not None
        hex_escaped = match.group(1)
        # Decode \\XX escapes back to bytes
        raw = bytes(int(hex_escaped[i+1:i+3], 16) for i in range(0, len(hex_escaped), 3))
        assert json.loads(raw.decode("utf-8")) == nodes

    def test_a11y_segment_size_matches_json(self):
        nodes = [{"role": "region", "label": "test area"}]
        raw = json.dumps(nodes, separators=(',', ':')).encode('utf-8')
        _, size = _a11y_segment(nodes, 0)
        assert size == len(raw)

    def test_a11y_segment_exports_offset_and_size(self):
        nodes = [{"role": "img", "label": "x"}]
        fragment, _ = _a11y_segment(nodes, 256)
        assert '"a11y_offset"' in fragment
        assert '"a11y_size"' in fragment
        assert "i32.const 256" in fragment


# ── a11y in rgba_to_wat ──────────────────────────────────────────────


class TestRgbaToWatA11y:
    def test_rgba_to_wat_with_a11y(self):
        """type bitmask: 1 | 8 = 9. a11y exports present."""
        rgba = bytes([255, 0, 0, 255] * 4)  # 2x2 red
        a11y = [{"role": "img", "label": "test"}]
        wat = rgba_to_wat(rgba, 2, 2, a11y=a11y)
        assert "i32.const 9" in wat
        assert '"a11y_offset"' in wat
        assert '"a11y_size"' in wat

    def test_backward_compat_no_a11y(self):
        """Without a11y: type=1, no a11y exports."""
        rgba = bytes([255, 0, 0, 255] * 4)
        wat = rgba_to_wat(rgba, 2, 2)
        assert "i32.const 1" in wat
        assert "a11y" not in wat


# ── a11y in audio_to_wat ─────────────────────────────────────────────


class TestAudioToWatA11y:
    def test_audio_to_wat_with_a11y(self):
        """type bitmask: 2 | 8 = 10. a11y exports present."""
        a11y = [{"role": "img", "label": "test"}]
        wat = audio_to_wat([440.0], [500.0], a11y=a11y)
        assert "i32.const 10" in wat
        assert '"a11y_offset"' in wat
        assert '"a11y_size"' in wat


# ── a11y in combined_to_wat ──────────────────────────────────────────


class TestCombinedToWatA11y:
    def test_combined_to_wat_with_a11y(self):
        """type bitmask: 3 | 8 = 11. a11y exports present."""
        rgba = bytes([255, 0, 0, 255] * 4)  # 2x2
        a11y = [{"role": "img", "label": "test"}]
        wat = combined_to_wat(rgba, 2, 2, [440.0], [500.0], a11y=a11y)
        assert "i32.const 11" in wat
        assert '"a11y_offset"' in wat
        assert '"a11y_size"' in wat
