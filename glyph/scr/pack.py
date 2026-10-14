"""
SCR file packer and unpacker.

SCR format: 6912 bytes total.
  Bytes 0-6143 (0x4000-0x57FF): bitmap, 1 bit per pixel, 256x192.
  Bytes 6144-6911 (0x5800-0x5AFF): attributes, 1 byte per 8x8 cell.

Bitmap layout is interleaved (not row-sequential). The pixel address
for (x, y) within the 6144-byte bitmap block is:

  addr = ((y & 0xC0) << 5) | ((y & 0x07) << 8) | ((y & 0x38) << 2) | (x >> 3)

This splits the 192 rows into three thirds (0-63, 64-127, 128-191),
then within each third interleaves scan lines by character row.

Attribute byte format: FBPPPIII
  F = flash (bit 7)
  B = bright (bit 6)
  PPP = paper colour index (bits 5-3)
  III = ink colour index (bits 2-0)
"""
import numpy as np

from .quantise import CellData, QuantisedImage, COLS, ROWS, SCREEN_H, SCREEN_W, CELL_SIZE

# SCR file size constants.
BITMAP_BYTES = 6144
ATTR_BYTES = 768
SCR_SIZE = BITMAP_BYTES + ATTR_BYTES   # 6912


def pixel_addr(x: int, y: int) -> int:
    """Return byte offset within the 6144-byte bitmap block for pixel (x, y).

    x: 0-255 (column)
    y: 0-191 (row)
    """
    return (
        ((y & 0xC0) << 5)
        | ((y & 0x07) << 8)
        | ((y & 0x38) << 2)
        | (x >> 3)
    )


def pack_scr(quantised: QuantisedImage, path: str) -> None:
    """Pack a QuantisedImage to a 6912-byte SCR file.

    Args:
        quantised: QuantisedImage from quantise_image().
        path: output file path.
    """
    data = bytearray(SCR_SIZE)

    for row in range(ROWS):
        for col in range(COLS):
            cell = quantised.cells[row * COLS + col]

            # Pack bitmap: 8 rows of 8 pixels = 8 bytes.
            bits = cell.pixel_bits.reshape(CELL_SIZE, CELL_SIZE)
            for scan in range(CELL_SIZE):
                y = row * CELL_SIZE + scan
                x = col * CELL_SIZE
                addr = pixel_addr(x, y)
                byte_val = 0
                for bit in range(8):
                    if bits[scan, bit]:
                        byte_val |= (0x80 >> bit)
                data[addr] = byte_val

            # Pack attribute byte: FBPPPIII
            attr_idx = row * COLS + col
            bright_bit = 1 if cell.bright else 0
            attr = (bright_bit << 6) | ((cell.paper & 0x07) << 3) | (cell.ink & 0x07)
            data[BITMAP_BYTES + attr_idx] = attr

    with open(path, "wb") as f:
        f.write(data)


def unpack_scr(path: str) -> list[CellData]:
    """Read a 6912-byte SCR file and return 768 CellData entries.

    Args:
        path: path to .scr file.

    Returns:
        List of 768 CellData entries in row-major order.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) != SCR_SIZE:
        raise ValueError(f"Expected {SCR_SIZE} bytes, got {len(data)}")

    cells: list[CellData] = []

    for row in range(ROWS):
        for col in range(COLS):
            attr_idx = row * COLS + col
            attr = data[BITMAP_BYTES + attr_idx]

            bright = bool((attr >> 6) & 1)
            paper = (attr >> 3) & 0x07
            ink = attr & 0x07

            pixel_bits = np.zeros(64, dtype=bool)

            for scan in range(CELL_SIZE):
                y = row * CELL_SIZE + scan
                x = col * CELL_SIZE
                addr = pixel_addr(x, y)
                byte_val = data[addr]
                for bit in range(8):
                    if byte_val & (0x80 >> bit):
                        pixel_bits[scan * CELL_SIZE + bit] = True

            cells.append(CellData(
                pixel_bits=pixel_bits,
                ink=ink,
                paper=paper,
                bright=bright,
            ))

    return cells
