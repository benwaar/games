"""Tests for glyph.scr.pack."""
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from glyph.scr.pack import pixel_addr, pack_scr, unpack_scr, SCR_SIZE, BITMAP_BYTES
from glyph.scr.quantise import quantise_image, CellData, COLS, ROWS


class TestPixelAddr:
    def test_origin(self):
        # (0, 0) is the first byte of the bitmap
        assert pixel_addr(0, 0) == 0

    def test_row_0_col_1(self):
        # x=8, y=0: same row, next byte
        assert pixel_addr(8, 0) == 1

    def test_row_8_interleaved(self):
        # y=8 is the start of character row 1 (scan line 0 of char row 1).
        # Using formula: ((8 & 0xC0) << 5) | ((8 & 0x07) << 8) | ((8 & 0x38) << 2) | 0
        # = 0 | 0 | (8 << 2) | 0 = 32
        assert pixel_addr(0, 8) == 32

    def test_row_1_scan_1(self):
        # y=1 is scan line 1 of char row 0, within first third.
        # ((1 & 0xC0) << 5) | ((1 & 0x07) << 8) | ((1 & 0x38) << 2) | 0
        # = 0 | (1 << 8) | 0 | 0 = 256
        assert pixel_addr(0, 1) == 256

    def test_second_third(self):
        # y=64 is first row of second third.
        # ((64 & 0xC0) << 5) | ((64 & 0x07) << 8) | ((64 & 0x38) << 2) | 0
        # = (64 << 5) | 0 | 0 | 0 = 2048
        assert pixel_addr(0, 64) == 2048

    def test_all_addresses_in_range(self):
        addrs = set()
        for y in range(192):
            for col in range(32):
                x = col * 8
                addr = pixel_addr(x, y)
                assert 0 <= addr < BITMAP_BYTES, f"Out of range: pixel_addr({x},{y}) = {addr}"
                addrs.add(addr)
        # Each of the 6144 byte addresses should be hit exactly once (one byte per 8 pixels)
        assert len(addrs) == BITMAP_BYTES


class TestPackUnpackRoundTrip:
    def _make_checkerboard_image(self) -> Image.Image:
        """256x192 checkerboard: alternating black and bright white 8x8 cells."""
        img = Image.new("RGB", (256, 192), (0, 0, 0))
        for row in range(ROWS):
            for col in range(COLS):
                if (row + col) % 2 == 0:
                    for y in range(8):
                        for x in range(8):
                            img.putpixel((col * 8 + x, row * 8 + y), (255, 255, 255))
        return img

    def test_round_trip_preserves_ink_paper(self):
        img = Image.new("RGB", (256, 192), (205, 0, 0))  # solid normal red
        quantised = quantise_image(img)

        with tempfile.NamedTemporaryFile(suffix=".scr", delete=False) as f:
            path = f.name
        try:
            pack_scr(quantised, path)
            cells = unpack_scr(path)

            assert len(cells) == 768
            for orig, loaded in zip(quantised.cells, cells):
                assert orig.ink == loaded.ink
                assert orig.paper == loaded.paper
                assert orig.bright == loaded.bright
                assert np.array_equal(orig.pixel_bits, loaded.pixel_bits)
        finally:
            os.unlink(path)

    def test_scr_file_is_6912_bytes(self):
        img = Image.new("RGB", (256, 192), (0, 0, 0))
        quantised = quantise_image(img)

        with tempfile.NamedTemporaryFile(suffix=".scr", delete=False) as f:
            path = f.name
        try:
            pack_scr(quantised, path)
            assert os.path.getsize(path) == SCR_SIZE
        finally:
            os.unlink(path)

    def test_black_image_attr_bytes_are_zero(self):
        img = Image.new("RGB", (256, 192), (0, 0, 0))
        quantised = quantise_image(img)

        with tempfile.NamedTemporaryFile(suffix=".scr", delete=False) as f:
            path = f.name
        try:
            pack_scr(quantised, path)
            with open(path, "rb") as f:
                data = f.read()
            attr_bytes = data[BITMAP_BYTES:]
            # All cells: black ink (0) and black paper (0), no bright
            # Attr = 0b00_000_000 = 0
            assert all(b == 0 for b in attr_bytes)
        finally:
            os.unlink(path)

    def test_bright_flag_round_trip(self):
        # Bright red image
        img = Image.new("RGB", (256, 192), (255, 0, 0))
        quantised = quantise_image(img)

        with tempfile.NamedTemporaryFile(suffix=".scr", delete=False) as f:
            path = f.name
        try:
            pack_scr(quantised, path)
            cells = unpack_scr(path)
            # All cells should have bright=True and ink or paper = 2 (red)
            for cell in cells:
                assert cell.bright is True
                assert cell.ink == 2 or cell.paper == 2
        finally:
            os.unlink(path)
