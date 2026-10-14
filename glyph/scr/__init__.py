"""
glyph.scr — ZX Spectrum SCR file tools.

Converts images to SCR format via per-cell k-means quantisation,
packs into the Spectrum's interleaved bitmap format, and renders
SCR files back to RGB images.
"""
from .palette import PALETTE, nearest_colour, ColourIndex
from .quantise import quantise_image, CellData, QuantisedImage
from .pack import pack_scr, unpack_scr
from .render import render_scr

__all__ = [
    "PALETTE",
    "nearest_colour",
    "ColourIndex",
    "quantise_image",
    "CellData",
    "QuantisedImage",
    "pack_scr",
    "unpack_scr",
    "render_scr",
]
