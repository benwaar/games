use crate::memory::Memory;

/// ZX Spectrum colour palette: 8 normal + 8 bright.
/// Index = colour (0–7); bright adds 8.
/// Normal intensity uses 0xCD; bright uses 0xFF.
const PALETTE: [(u8, u8, u8); 16] = [
    // Normal
    (0x00, 0x00, 0x00), // 0  Black
    (0x00, 0x00, 0xCD), // 1  Blue
    (0xCD, 0x00, 0x00), // 2  Red
    (0xCD, 0x00, 0xCD), // 3  Magenta
    (0x00, 0xCD, 0x00), // 4  Green
    (0x00, 0xCD, 0xCD), // 5  Cyan
    (0xCD, 0xCD, 0x00), // 6  Yellow
    (0xCD, 0xCD, 0xCD), // 7  White
    // Bright
    (0x00, 0x00, 0x00), // 8  Bright Black (same as black)
    (0x00, 0x00, 0xFF), // 9  Bright Blue
    (0xFF, 0x00, 0x00), // 10 Bright Red
    (0xFF, 0x00, 0xFF), // 11 Bright Magenta
    (0x00, 0xFF, 0x00), // 12 Bright Green
    (0x00, 0xFF, 0xFF), // 13 Bright Cyan
    (0xFF, 0xFF, 0x00), // 14 Bright Yellow
    (0xFF, 0xFF, 0xFF), // 15 Bright White
];

/// Address of the pixel byte containing pixel (x, y).
///
/// The Spectrum screen is divided into three "thirds" of 64 lines each.
/// Within each third the lines are NOT stored sequentially — they are
/// interleaved by character row and scan line:
///
///   bits 12–11 → third (y bits 7–6)
///   bits 10–8  → scan line within character row (y bits 2–0)
///   bits  7–5  → character row within third (y bits 5–3)
///   bits  4–0  → byte column (x / 8)
pub fn pixel_addr(x: u16, y: u16) -> u16 {
    0x4000
        | ((y & 0xC0) << 5)  // third  → bits 12–11
        | ((y & 0x07) << 8)  // scan line within char row → bits 10–8
        | ((y & 0x38) << 2)  // char row within third → bits 7–5
        | (x >> 3)           // byte column
}

/// Address of the attribute byte for the 8×8 cell containing pixel (x, y).
/// Attributes live at 0x5800–0x5AFF: 24 rows × 32 columns = 768 bytes.
pub fn attr_addr(x: u16, y: u16) -> u16 {
    0x5800 + (y / 8) * 32 + (x / 8)
}

/// Render a full 256×192 Spectrum screen into a linear RGBA pixel buffer
/// (256 × 192 × 4 = 196 608 bytes, top-left first, left-to-right).
///
/// `flash_frame` is the running frame counter.  The FLASH attribute toggles
/// INK↔PAPER every 16 frames.
pub fn render_frame(memory: &Memory, flash_frame: u32) -> Vec<u8> {
    let flash_on = (flash_frame / 16) % 2 == 1;
    let mut pixels = vec![0u8; 256 * 192 * 4];

    for y in 0u16..192 {
        for x in 0u16..256 {
            // Read pixel bit
            let byte = memory.read_byte(pixel_addr(x, y));
            let bit_pos = 7 - (x & 7);
            let pixel_set = (byte >> bit_pos) & 1 != 0;

            // Decode attribute byte: FBPPPIII
            let attr = memory.read_byte(attr_addr(x, y));
            let ink_idx   = (attr & 0x07) as usize;
            let paper_idx = ((attr >> 3) & 0x07) as usize;
            let bright    = (attr >> 6) & 1 != 0;
            let flash     = (attr >> 7) & 1 != 0;

            // FLASH swaps INK and PAPER when active
            let (fg, bg) = if flash && flash_on {
                (paper_idx, ink_idx)
            } else {
                (ink_idx, paper_idx)
            };

            let colour_idx = if pixel_set { fg } else { bg } + if bright { 8 } else { 0 };
            let (r, g, b) = PALETTE[colour_idx];

            let i = (y as usize * 256 + x as usize) * 4;
            pixels[i]     = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
            pixels[i + 3] = 0xFF;
        }
    }

    pixels
}
