use zxw::memory::Memory;
use zxw::ula::{pixel_addr, attr_addr, render_frame};

// ------------------------------------------------------------------
// pixel_addr — address of the byte containing pixel (x, y)
// ------------------------------------------------------------------

#[test]
fn pixel_addr_origin() {
    assert_eq!(pixel_addr(0, 0), 0x4000);
}

#[test]
fn pixel_addr_top_right_byte() {
    // y=0, x=255 → column 31 (255/8=31), same row
    assert_eq!(pixel_addr(255, 0), 0x4000 + 31);
}

#[test]
fn pixel_addr_y1_char_line_1() {
    // y=1: third=0, char_line=1, char_row=0
    // 0x4000 | (1 << 8) = 0x4100
    assert_eq!(pixel_addr(0, 1), 0x4100);
}

#[test]
fn pixel_addr_y8_second_row_in_char() {
    // y=8: third=0, char_line=0, char_row=1 → row offset = (8 & 0x38) << 2 = 0x20
    assert_eq!(pixel_addr(0, 8), 0x4020);
}

#[test]
fn pixel_addr_start_of_second_third() {
    // y=64: (64 & 0xC0)=64; 64<<5=0x0800 → 0x4800
    assert_eq!(pixel_addr(0, 64), 0x4800);
}

#[test]
fn pixel_addr_start_of_third_third() {
    // y=128: (128 & 0xC0)=128; 128<<5=0x1000 → 0x5000
    assert_eq!(pixel_addr(0, 128), 0x5000);
}

#[test]
fn pixel_addr_bottom_left() {
    // y=191 = 0b10111111
    // (191 & 0xC0)=0x80 → <<5 = 0x1000
    // (191 & 0x07)=0x07 → <<8 = 0x0700
    // (191 & 0x38)=0x38 → <<2 = 0x00E0
    // x=0 → 0x0000
    // total = 0x4000 | 0x1000 | 0x0700 | 0x00E0 = 0x57E0
    assert_eq!(pixel_addr(0, 191), 0x57E0);
}

// ------------------------------------------------------------------
// attr_addr — attribute byte address for 8×8 cell containing (x, y)
// ------------------------------------------------------------------

#[test]
fn attr_addr_origin() {
    assert_eq!(attr_addr(0, 0), 0x5800);
}

#[test]
fn attr_addr_top_right_cell() {
    assert_eq!(attr_addr(255, 0), 0x5800 + 31);
}

#[test]
fn attr_addr_second_row() {
    // y=8 → row 1 (8/8=1), x=0 → col 0
    assert_eq!(attr_addr(0, 8), 0x5800 + 32);
}

#[test]
fn attr_addr_same_cell_for_all_pixels_in_block() {
    // Every pixel in the top-left 8×8 block shares attribute 0x5800
    for dy in 0u16..8 {
        for dx in 0u16..8 {
            assert_eq!(attr_addr(dx, dy), 0x5800,
                "pixel ({dx},{dy}) should use attr 0x5800");
        }
    }
}

#[test]
fn attr_addr_bottom_right() {
    // y=191 → row 23, x=255 → col 31
    assert_eq!(attr_addr(255, 191), 0x5800 + 23 * 32 + 31);
}

// ------------------------------------------------------------------
// render_frame — pixel buffer correctness
// ------------------------------------------------------------------

#[test]
fn render_blank_memory_all_black() {
    // Zeroed memory: all attrs = 0 (ink=0/black, paper=0/black), all pixels = 0
    let mem = Memory::new();
    let pixels = render_frame(&mem, 0);
    assert_eq!(pixels.len(), 256 * 192 * 4);
    for (i, chunk) in pixels.chunks(4).enumerate() {
        assert_eq!(chunk, &[0, 0, 0, 0xFF],
            "pixel {i} should be black, got {chunk:?}");
    }
}

#[test]
fn render_all_ink_normal_white() {
    let mut mem = Memory::new();
    // All attr cells: ink=7 (white), paper=0 (black), no bright
    for addr in 0x5800u16..0x5B00 {
        mem.write_byte(addr, 0x07);
    }
    // All pixel bytes 0xFF → every pixel is INK
    for addr in 0x4000u16..0x5800 {
        mem.write_byte(addr, 0xFF);
    }
    let pixels = render_frame(&mem, 0);
    for (i, chunk) in pixels.chunks(4).enumerate() {
        assert_eq!(chunk, &[0xCD, 0xCD, 0xCD, 0xFF],
            "pixel {i} should be normal white, got {chunk:?}");
    }
}

#[test]
fn render_bright_increases_intensity() {
    let mut mem = Memory::new();
    // attr: bright=1, ink=2 (red) → bright red = (0xFF, 0, 0)
    for addr in 0x5800u16..0x5B00 {
        mem.write_byte(addr, 0b0100_0010);
    }
    // All pixels INK
    for addr in 0x4000u16..0x5800 {
        mem.write_byte(addr, 0xFF);
    }
    let pixels = render_frame(&mem, 0);
    for (i, chunk) in pixels.chunks(4).enumerate() {
        assert_eq!(&chunk[0..3], &[0xFF, 0x00, 0x00],
            "pixel {i} should be bright red, got {chunk:?}");
    }
}

#[test]
fn render_attr_cell_boundary() {
    let mut mem = Memory::new();

    // Cell (0,0): ink=7 white
    mem.write_byte(0x5800, 0x07);
    // Set the leftmost pixel of y=0 (bit 7 of the first byte)
    mem.write_byte(pixel_addr(0, 0), 0x80);

    // Cell (1,0): ink=2 red
    mem.write_byte(0x5801, 0x02);
    // Set the leftmost pixel of cell (1,0) — x=8, y=0 (bit 7 of second byte)
    mem.write_byte(pixel_addr(8, 0), 0x80);

    let pixels = render_frame(&mem, 0);

    // Pixel (0,0) → white INK
    assert_eq!(&pixels[0..4], &[0xCD, 0xCD, 0xCD, 0xFF]);

    // Pixel (8,0) → red INK
    let i = (0 * 256 + 8) * 4;
    assert_eq!(&pixels[i..i + 4], &[0xCD, 0x00, 0x00, 0xFF]);
}

#[test]
fn render_paper_colour_used_for_unset_pixels() {
    let mut mem = Memory::new();
    // attr: paper=5 (cyan), ink=0 (black)
    for addr in 0x5800u16..0x5B00 {
        mem.write_byte(addr, 0b000_101_000); // paper=5, ink=0
    }
    // All pixel bytes 0x00 → every pixel is PAPER
    // (pixel data already zero from Memory::new)
    let pixels = render_frame(&mem, 0);
    for (i, chunk) in pixels.chunks(4).enumerate() {
        assert_eq!(chunk, &[0x00, 0xCD, 0xCD, 0xFF],
            "pixel {i} should be normal cyan (paper), got {chunk:?}");
    }
}
