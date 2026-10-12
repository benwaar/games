/// ZX Spectrum keyboard half-row matrix.
///
/// The Spectrum reads keys via IN port 0xFE. The high byte of the port address
/// selects one or more half-rows (active low — bit 0 in the high byte activates
/// row 0, bit 1 activates row 1, etc.). Each row returns 5 bits (bits 0–4);
/// 0 = key pressed, 1 = not pressed. Bits 5–7 are always 1.
///
/// Half-row layout (row index, bits 0–4):
///   0 (A8,  0xFEFE): SHIFT  Z  X  C  V
///   1 (A9,  0xFDFE): A      S  D  F  G
///   2 (A10, 0xFBFE): Q      W  E  R  T
///   3 (A11, 0xF7FE): 1      2  3  4  5
///   4 (A12, 0xEFFE): 0      9  8  7  6
///   5 (A13, 0xDFFE): P      O  I  U  Y
///   6 (A14, 0xBFFE): ENTER  L  K  J  H
///   7 (A15, 0x7FFE): SPACE  SYM M  N  B
pub struct Keyboard {
    /// Half-row state. Bits 0–4 per row; 0 = pressed, 1 = not pressed.
    rows: [u8; 8],
    /// Kempston joystick state (port 0x1F).
    /// Bit 0=right, 1=left, 2=down, 3=up, 4=fire; 1 = active.
    kempston: u8,
}

impl Keyboard {
    pub fn new() -> Self {
        Keyboard {
            rows: [0xFF; 8],
            kempston: 0,
        }
    }

    /// Clear a bit in the given row — signals key pressed.
    pub fn key_down(&mut self, row: usize, bit: usize) {
        if row < 8 && bit < 5 {
            self.rows[row] &= !(1u8 << bit);
        }
    }

    /// Set a bit in the given row — signals key released.
    pub fn key_up(&mut self, row: usize, bit: usize) {
        if row < 8 && bit < 5 {
            self.rows[row] |= 1u8 << bit;
        }
    }

    /// Read the keyboard half-rows selected by `high_byte`.
    ///
    /// Each bit of `high_byte` corresponds to an address line (A8–A15).
    /// A bit value of 0 means that address line is LOW, selecting that row.
    /// ANDs together all selected rows, then ORs bits 5–7 to 1.
    pub fn read_port(&self, high_byte: u8) -> u8 {
        let mut result: u8 = 0x1F; // assume all 5 key bits unset
        for row in 0..8usize {
            if (high_byte >> row) & 1 == 0 {
                result &= self.rows[row] & 0x1F;
            }
        }
        result | 0xE0 // bits 5–7 always 1
    }

    /// Set a Kempston joystick bit (bit 0=right, 1=left, 2=down, 3=up, 4=fire).
    pub fn kempston_down(&mut self, bit: usize) {
        if bit < 5 {
            self.kempston |= 1u8 << bit;
        }
    }

    /// Clear a Kempston joystick bit (released).
    pub fn kempston_up(&mut self, bit: usize) {
        if bit < 5 {
            self.kempston &= !(1u8 << bit);
        }
    }

    /// Read the Kempston joystick state (port 0x1F).
    pub fn read_kempston(&self) -> u8 {
        self.kempston
    }

    /// Reset all keys and joystick to not-pressed state.
    pub fn reset(&mut self) {
        self.rows = [0xFF; 8];
        self.kempston = 0;
    }
}

impl Default for Keyboard {
    fn default() -> Self {
        Keyboard::new()
    }
}
