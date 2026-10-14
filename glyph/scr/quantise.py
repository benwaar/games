"""
Image-to-SCR quantiser.

Converts a PIL Image to per-cell Spectrum colour assignments via k-means.

Pipeline:
1. Letterbox-resize image to 256x192 (black padding, not crop).
2. For each 8x8 cell (768 total, 32 cols x 24 rows):
   a. Extract 64 pixels as RGB vectors.
   b. Run k-means k=2 to get two cluster centroids.
   c. Map each centroid to nearest Spectrum palette colour.
   d. Determine ink (colour 1), paper (colour 0), bright flag.
3. Return cell assignments and attribute data.

The bright flag is set if either mapped colour is in the bright set.
When bright is set both ink and paper use their bright variants,
matching the Spectrum hardware constraint.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from .palette import ColourIndex, PALETTE, colour_to_rgb, nearest_colour

# Spectrum screen dimensions.
SCREEN_W = 256
SCREEN_H = 192
CELL_SIZE = 8
COLS = SCREEN_W // CELL_SIZE   # 32
ROWS = SCREEN_H // CELL_SIZE   # 24


@dataclass
class CellData:
    """Quantised data for one 8x8 cell.

    pixel_bits: 64-element bool array. True = ink, False = paper.
    ink: Spectrum colour index for ink (bit=1).
    paper: Spectrum colour index for paper (bit=0).
    bright: bright flag (applies to both ink and paper).
    """
    pixel_bits: np.ndarray   # shape (64,), dtype bool
    ink: int                 # colour index 0-7
    paper: int               # colour index 0-7
    bright: bool


@dataclass
class QuantisedImage:
    """Result of quantising a full 256x192 image.

    cells: flat list of 768 CellData, row-major (cell[row*32+col]).
    preview: 256x192 PIL Image reconstructed from Spectrum colours (for comparison).
    """
    cells: list[CellData]
    preview: Image.Image


def _letterbox(image: Image.Image) -> Image.Image:
    """Resize image to 256x192 with black letterboxing (no crop).

    Scales the image to fit within 256x192 while preserving aspect ratio,
    then centres it on a black background.
    """
    src_w, src_h = image.size
    scale = min(SCREEN_W / src_w, SCREEN_H / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))
    offset_x = (SCREEN_W - new_w) // 2
    offset_y = (SCREEN_H - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _quantise_cell(pixels: np.ndarray) -> CellData:
    """Quantise 64 pixels (shape 64x3) to two Spectrum colours.

    Args:
        pixels: float32 array, shape (64, 3), values 0-255.

    Returns:
        CellData with ink, paper, bright, and pixel bit assignments.
    """
    # If all pixels are identical, skip k-means to avoid degenerate clusters.
    if np.all(pixels == pixels[0]):
        colour = nearest_colour((int(pixels[0, 0]), int(pixels[0, 1]), int(pixels[0, 2])))
        return CellData(
            pixel_bits=np.zeros(64, dtype=bool),
            ink=colour.index,
            paper=colour.index,
            bright=colour.bright,
        )

    km = KMeans(n_clusters=2, n_init=4, random_state=0)
    labels = km.fit_predict(pixels)   # shape (64,), values 0 or 1
    centroids = km.cluster_centers_   # shape (2, 3)

    c0 = nearest_colour((int(centroids[0, 0]), int(centroids[0, 1]), int(centroids[0, 2])))
    c1 = nearest_colour((int(centroids[1, 0]), int(centroids[1, 1]), int(centroids[1, 2])))

    # Bright flag: set if either colour is bright (and they're not both black).
    # When bright is set, both colours shift to their bright variants.
    bright = c0.bright or c1.bright

    # Resolve ink/paper. Convention: paper = cluster 0, ink = cluster 1.
    # If bright flag is set, use the colour index in bright context.
    paper = c0.index
    ink = c1.index

    # pixel_bits: True where label == 1 (ink).
    pixel_bits = labels.astype(bool)

    return CellData(pixel_bits=pixel_bits, ink=ink, paper=paper, bright=bright)


def quantise_image(image: Image.Image) -> QuantisedImage:
    """Convert a PIL Image to a QuantisedImage with per-cell Spectrum assignments.

    Args:
        image: Any-size PIL Image (mode RGB or convertible).

    Returns:
        QuantisedImage with 768 CellData entries and a preview image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = _letterbox(image)
    pixels = np.array(resized, dtype=np.float32)   # shape (192, 256, 3)

    cells: list[CellData] = []

    for row in range(ROWS):
        for col in range(COLS):
            y0 = row * CELL_SIZE
            x0 = col * CELL_SIZE
            block = pixels[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE]   # (8, 8, 3)
            flat = block.reshape(64, 3)
            cells.append(_quantise_cell(flat))

    preview = _build_preview(cells)
    return QuantisedImage(cells=cells, preview=preview)


def _build_preview(cells: list[CellData]) -> Image.Image:
    """Reconstruct a 256x192 PIL Image from quantised cell data."""
    out = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    for row in range(ROWS):
        for col in range(COLS):
            cell = cells[row * COLS + col]
            ink_rgb = colour_to_rgb(ColourIndex(cell.ink, cell.bright))
            paper_rgb = colour_to_rgb(ColourIndex(cell.paper, cell.bright))
            y0 = row * CELL_SIZE
            x0 = col * CELL_SIZE
            block = out[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE]   # view

            bits = cell.pixel_bits.reshape(CELL_SIZE, CELL_SIZE)
            block[bits] = ink_rgb
            block[~bits] = paper_rgb

    return Image.fromarray(out, "RGB")
