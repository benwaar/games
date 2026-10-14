"""
ZX Spectrum colour palette.

The Spectrum has 15 usable colours: black + 7 normal + 7 bright.
Bright black is the same as normal black (both map to RGB 0,0,0).

Each colour has an index 0-7 and a bright flag. The attribute byte
encodes FBPPPIII (Flash, Bright, Paper 0-7, Ink 0-7).

Palette layout (index, bright) -> RGB:
  (0, False) = black      (0,   0,   0)
  (1, False) = blue       (0,   0, 205)
  (2, False) = red        (205, 0,   0)
  (3, False) = magenta    (205, 0, 205)
  (4, False) = green      (0, 205,   0)
  (5, False) = cyan       (0, 205, 205)
  (6, False) = yellow     (205, 205,  0)
  (7, False) = white      (205, 205, 205)
  (0, True)  = black      (0,   0,   0)   (bright black = black)
  (1, True)  = blue       (0,   0, 255)
  (2, True)  = red        (255, 0,   0)
  (3, True)  = magenta    (255, 0, 255)
  (4, True)  = green      (0, 255,   0)
  (5, True)  = cyan       (0, 255, 255)
  (6, True)  = yellow     (255, 255,  0)
  (7, True)  = white      (255, 255, 255)
"""
from typing import NamedTuple

import numpy as np


class ColourIndex(NamedTuple):
    index: int   # 0-7
    bright: bool


# All 15 usable Spectrum colours as (ColourIndex, RGB tuple).
# Bright black duplicates normal black, but we only need one entry
# for palette matching — it's included in BRIGHT_ENTRIES for completeness
# but excluded from PALETTE_ENTRIES so nearest_colour always maps to
# normal black (index 0, bright False).
_NORMAL_RGB: tuple[tuple[int, int, int], ...] = (
    (0,   0,   0),    # 0 black
    (0,   0, 205),    # 1 blue
    (205, 0,   0),    # 2 red
    (205, 0, 205),    # 3 magenta
    (0, 205,   0),    # 4 green
    (0, 205, 205),    # 5 cyan
    (205, 205,  0),   # 6 yellow
    (205, 205, 205),  # 7 white
)

_BRIGHT_RGB: tuple[tuple[int, int, int], ...] = (
    (0,   0,   0),    # 0 bright black = black
    (0,   0, 255),    # 1 bright blue
    (255, 0,   0),    # 2 bright red
    (255, 0, 255),    # 3 bright magenta
    (0, 255,   0),    # 4 bright green
    (0, 255, 255),    # 5 bright cyan
    (255, 255,  0),   # 6 bright yellow
    (255, 255, 255),  # 7 bright white
)

# PALETTE: list of (ColourIndex, rgb) pairs used for nearest-colour lookup.
# Bright black is omitted — it's identical to normal black.
PALETTE: list[tuple[ColourIndex, tuple[int, int, int]]] = [
    (ColourIndex(i, False), _NORMAL_RGB[i]) for i in range(8)
] + [
    (ColourIndex(i, True), _BRIGHT_RGB[i]) for i in range(1, 8)  # skip bright black
]

# Pre-computed array for vectorised distance calculations.
# Shape: (15, 3) — one RGB row per palette entry.
_PALETTE_ARRAY: np.ndarray = np.array([rgb for _, rgb in PALETTE], dtype=np.float32)


def nearest_colour(rgb: tuple[int, int, int]) -> ColourIndex:
    """Return the Spectrum ColourIndex closest to the given RGB value.

    Uses Euclidean distance in RGB space.

    Args:
        rgb: (R, G, B) tuple, values 0-255.

    Returns:
        ColourIndex with the nearest palette entry.
    """
    point = np.array(rgb, dtype=np.float32)
    dists = np.sum((_PALETTE_ARRAY - point) ** 2, axis=1)
    best = int(np.argmin(dists))
    return PALETTE[best][0]


def colour_to_rgb(colour: ColourIndex) -> tuple[int, int, int]:
    """Return the RGB tuple for a ColourIndex."""
    if colour.bright:
        return _BRIGHT_RGB[colour.index]
    return _NORMAL_RGB[colour.index]
