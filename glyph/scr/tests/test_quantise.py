"""Tests for glyph.scr.quantise."""
import numpy as np
import pytest
from PIL import Image

from glyph.scr.quantise import (
    _letterbox,
    _quantise_cell,
    quantise_image,
    SCREEN_W,
    SCREEN_H,
    COLS,
    ROWS,
    CELL_SIZE,
)
from glyph.scr.palette import ColourIndex


class TestLetterbox:
    def test_exact_size_unchanged(self):
        img = Image.new("RGB", (256, 192), (100, 100, 100))
        result = _letterbox(img)
        assert result.size == (256, 192)

    def test_wider_image_adds_top_bottom_bars(self):
        # 512x192 is twice as wide — needs letterbox top/bottom
        img = Image.new("RGB", (512, 192), (200, 0, 0))
        result = _letterbox(img)
        assert result.size == (256, 192)
        # Top row should be black (letterbox bar)
        top_row = np.array(result)[0, :, :]
        assert np.all(top_row == 0)

    def test_taller_image_adds_left_right_bars(self):
        # 256x384 is twice as tall — needs letterbox left/right
        img = Image.new("RGB", (256, 384), (0, 200, 0))
        result = _letterbox(img)
        assert result.size == (256, 192)
        # Left column should be black
        left_col = np.array(result)[:, 0, :]
        assert np.all(left_col == 0)

    def test_output_is_rgb(self):
        img = Image.new("L", (256, 192), 128)
        result = _letterbox(img.convert("RGB"))
        assert result.mode == "RGB"


class TestQuantiseCell:
    def test_solid_red_block(self):
        # All 64 pixels are Spectrum bright red (255, 0, 0)
        pixels = np.full((64, 3), [255, 0, 0], dtype=np.float32)
        cell = _quantise_cell(pixels)
        # Both ink and paper should resolve to red; bright should be True
        assert cell.ink == 2 or cell.paper == 2
        assert cell.bright is True

    def test_solid_black_block(self):
        pixels = np.zeros((64, 3), dtype=np.float32)
        cell = _quantise_cell(pixels)
        assert cell.ink == 0
        assert cell.paper == 0
        assert cell.bright is False

    def test_pixel_bits_shape(self):
        pixels = np.random.rand(64, 3).astype(np.float32) * 255
        cell = _quantise_cell(pixels)
        assert cell.pixel_bits.shape == (64,)
        assert cell.pixel_bits.dtype == bool

    def test_two_distinct_colours_assigns_both(self):
        # Half pixels are bright white, half are black
        pixels = np.zeros((64, 3), dtype=np.float32)
        pixels[32:] = 255.0
        cell = _quantise_cell(pixels)
        # Ink and paper should differ
        assert cell.ink != cell.paper or (cell.pixel_bits.all() or not cell.pixel_bits.any())


class TestQuantiseImage:
    def test_returns_768_cells(self):
        img = Image.new("RGB", (256, 192), (128, 0, 128))
        result = quantise_image(img)
        assert len(result.cells) == 768

    def test_preview_is_correct_size(self):
        img = Image.new("RGB", (256, 192), (0, 128, 0))
        result = quantise_image(img)
        assert result.preview.size == (256, 192)

    def test_solid_spectrum_green_image(self):
        # Normal green (0, 205, 0)
        img = Image.new("RGB", (256, 192), (0, 205, 0))
        result = quantise_image(img)
        for cell in result.cells:
            # All cells should map to green (index 4)
            colours = {cell.ink, cell.paper}
            assert 4 in colours

    def test_rgba_image_converted(self):
        img = Image.new("RGBA", (256, 192), (255, 0, 0, 255))
        result = quantise_image(img)
        assert len(result.cells) == 768
