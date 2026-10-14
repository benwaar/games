"""Tests for glyph.scr.palette."""
import pytest
import numpy as np

from glyph.scr.palette import (
    ColourIndex,
    PALETTE,
    nearest_colour,
    colour_to_rgb,
    _NORMAL_RGB,
    _BRIGHT_RGB,
)


class TestPaletteStructure:
    def test_palette_has_15_entries(self):
        # 8 normal + 7 bright (bright black excluded as duplicate)
        assert len(PALETTE) == 15

    def test_all_entries_are_colour_index_rgb_pairs(self):
        for ci, rgb in PALETTE:
            assert isinstance(ci, ColourIndex)
            assert len(rgb) == 3
            assert all(0 <= v <= 255 for v in rgb)

    def test_normal_black_in_palette(self):
        first_ci, first_rgb = PALETTE[0]
        assert first_ci == ColourIndex(0, False)
        assert first_rgb == (0, 0, 0)

    def test_no_bright_black_in_palette(self):
        # Bright black would be ColourIndex(0, True)
        palette_indices = [ci for ci, _ in PALETTE]
        assert ColourIndex(0, True) not in palette_indices


class TestNearestColour:
    def test_exact_normal_red(self):
        result = nearest_colour((205, 0, 0))
        assert result == ColourIndex(2, False)

    def test_exact_bright_red(self):
        result = nearest_colour((255, 0, 0))
        assert result == ColourIndex(2, True)

    def test_exact_black(self):
        result = nearest_colour((0, 0, 0))
        assert result == ColourIndex(0, False)

    def test_near_black(self):
        result = nearest_colour((10, 10, 10))
        assert result == ColourIndex(0, False)

    def test_exact_bright_white(self):
        result = nearest_colour((255, 255, 255))
        assert result == ColourIndex(7, True)

    def test_exact_normal_white(self):
        result = nearest_colour((205, 205, 205))
        assert result == ColourIndex(7, False)

    def test_bright_green(self):
        result = nearest_colour((0, 255, 0))
        assert result == ColourIndex(4, True)

    def test_normal_cyan(self):
        result = nearest_colour((0, 205, 205))
        assert result == ColourIndex(5, False)

    def test_all_exact_palette_colours_round_trip(self):
        for ci, rgb in PALETTE:
            result = nearest_colour(rgb)
            assert result == ci, f"Expected {ci}, got {result} for rgb={rgb}"


class TestColourToRgb:
    def test_normal_black(self):
        assert colour_to_rgb(ColourIndex(0, False)) == (0, 0, 0)

    def test_bright_red(self):
        assert colour_to_rgb(ColourIndex(2, True)) == (255, 0, 0)

    def test_normal_blue(self):
        assert colour_to_rgb(ColourIndex(1, False)) == (0, 0, 205)

    def test_bright_yellow(self):
        assert colour_to_rgb(ColourIndex(6, True)) == (255, 255, 0)
