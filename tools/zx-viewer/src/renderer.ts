// ULA decode: ZX Spectrum pixel address, attributes, palette, and canvas rendering helpers.
// These are pure data-transform functions — no VS Code or DOM dependencies.

export type RgbTuple = [number, number, number];

// Normal and bright Spectrum palette (index 0-7, 8-15 for bright).
// Bright black is still (0,0,0).
const PALETTE: RgbTuple[] = [
  [0, 0, 0],       // 0: black
  [0, 0, 0xCD],    // 1: blue
  [0xCD, 0, 0],    // 2: red
  [0xCD, 0, 0xCD], // 3: magenta
  [0, 0xCD, 0],    // 4: green
  [0, 0xCD, 0xCD], // 5: cyan
  [0xCD, 0xCD, 0], // 6: yellow
  [0xCD, 0xCD, 0xCD], // 7: white
  // bright variants (0-7 duplicated but with 0xFF, except bright black stays 0)
  [0, 0, 0],       // 8: bright black = still black
  [0, 0, 0xFF],    // 9: bright blue
  [0xFF, 0, 0],    // 10: bright red
  [0xFF, 0, 0xFF], // 11: bright magenta
  [0, 0xFF, 0],    // 12: bright green
  [0, 0xFF, 0xFF], // 13: bright cyan
  [0xFF, 0xFF, 0], // 14: bright yellow
  [0xFF, 0xFF, 0xFF], // 15: bright white
];

export const SCREEN_WIDTH = 256;
export const SCREEN_HEIGHT = 192;
export const SCR_SIZE = 6912;
export const BITMAP_SIZE = 6144;

/**
 * Returns the byte offset within the 6144-byte bitmap for the byte containing pixel (x, y).
 * ZX Spectrum rows are interleaved: the memory layout splits the screen into thirds,
 * then char rows within each third, then scan lines within each char row.
 */
export function pixelOffset(x: number, y: number): number {
  return ((y & 0xC0) << 5)  // which third (0/64/128 → bits 12-11)
       | ((y & 0x07) << 8)  // scan line within char row (bits 10-8)
       | ((y & 0x38) << 2)  // char row within third (bits 7-5)
       | (x >> 3);           // byte column (bits 4-0)
}

/** Decode a single attribute byte into ink and paper colour indices (0-15 for bright). */
export function decodeAttr(attr: number): { ink: number; paper: number; flash: boolean } {
  const flash = (attr & 0x80) !== 0;
  const bright = (attr & 0x40) !== 0;
  const paper = (attr >> 3) & 0x07;
  const ink = attr & 0x07;
  const offset = bright ? 8 : 0;
  return { ink: ink + offset, paper: paper + offset, flash };
}

/** Look up an RGB tuple from a 0-15 palette index. */
export function paletteColour(index: number): RgbTuple {
  return PALETTE[index] ?? [0, 0, 0];
}

/**
 * Decode 6912 bytes of SCR data into a flat RGBA Uint8ClampedArray (256x192 pixels).
 * Each pixel is 4 bytes: R, G, B, A (A always 255).
 */
export function decodeScrToRgba(scr: Uint8Array): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SCREEN_WIDTH * SCREEN_HEIGHT * 4);

  for (let y = 0; y < SCREEN_HEIGHT; y++) {
    for (let x = 0; x < SCREEN_WIDTH; x++) {
      const byteOffset = pixelOffset(x, y);
      const bitmapByte = scr[byteOffset];
      const bitIndex = 7 - (x & 7); // MSB = leftmost pixel
      const pixelSet = (bitmapByte >> bitIndex) & 1;

      // Attribute cell: 32 cols x 24 rows
      const attrRow = Math.floor(y / 8);
      const attrCol = Math.floor(x / 8);
      const attrOffset = BITMAP_SIZE + attrRow * 32 + attrCol;
      const attr = scr[attrOffset];

      const { ink, paper } = decodeAttr(attr);
      const colour = paletteColour(pixelSet ? ink : paper);

      const rgbaIndex = (y * SCREEN_WIDTH + x) * 4;
      rgba[rgbaIndex]     = colour[0];
      rgba[rgbaIndex + 1] = colour[1];
      rgba[rgbaIndex + 2] = colour[2];
      rgba[rgbaIndex + 3] = 255;
    }
  }

  return rgba;
}

/** Count distinct colours used in an SCR buffer (by attribute bytes only — faster than pixel scan). */
export function countColours(scr: Uint8Array): number {
  const seen = new Set<number>();
  for (let i = BITMAP_SIZE; i < SCR_SIZE; i++) {
    const attr = scr[i];
    const bright = (attr & 0x40) !== 0;
    const paper = ((attr >> 3) & 0x07) + (bright ? 8 : 0);
    const ink = (attr & 0x07) + (bright ? 8 : 0);
    seen.add(ink);
    seen.add(paper);
  }
  return seen.size;
}
