"""
SCR renderer: decode a .scr file back to a PIL RGB image.

Reads the 6144-byte interleaved bitmap and 768-byte attribute block,
decodes each cell's ink/paper/bright assignment, and paints pixels.
"""
import numpy as np
from PIL import Image

from .pack import unpack_scr
from .palette import ColourIndex, colour_to_rgb
from .quantise import CELL_SIZE, COLS, ROWS, SCREEN_H, SCREEN_W


def render_scr(path: str) -> Image.Image:
    """Render a .scr file to a 256x192 PIL RGB Image.

    Args:
        path: path to a 6912-byte .scr file.

    Returns:
        256x192 PIL Image in RGB mode.
    """
    cells = unpack_scr(path)

    out = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    for row in range(ROWS):
        for col in range(COLS):
            cell = cells[row * COLS + col]
            ink_rgb = colour_to_rgb(ColourIndex(cell.ink, cell.bright))
            paper_rgb = colour_to_rgb(ColourIndex(cell.paper, cell.bright))

            y0 = row * CELL_SIZE
            x0 = col * CELL_SIZE
            block = out[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE]

            bits = cell.pixel_bits.reshape(CELL_SIZE, CELL_SIZE)
            block[bits] = ink_rgb
            block[~bits] = paper_rgb

    return Image.fromarray(out, "RGB")
