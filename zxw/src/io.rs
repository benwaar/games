/// I/O port trait for the Z80 CPU.
///
/// The Z80's IN and OUT instructions call these methods. Implementors
/// route port reads/writes to the correct hardware: ULA (keyboard +
/// border), AY sound chip (M6+), etc.
pub trait IoHandler {
    /// Read a byte from the given port address.
    fn port_read(&self, port: u16) -> u8;
    /// Write a byte to the given port address.
    fn port_write(&mut self, port: u16, val: u8);
}

/// No-op I/O handler for unit tests — reads return 0xFF, writes are ignored.
pub struct NullIo;

impl IoHandler for NullIo {
    fn port_read(&self, _port: u16) -> u8 { 0xFF }
    fn port_write(&mut self, _port: u16, _val: u8) {}
}

/// Minimal ULA I/O handler — border colour only, no keyboard.
/// Kept for standalone use; `SpectrumIo` in `spectrum.rs` is the full
/// implementation used by the emulator.
pub struct UlaIo {
    pub border: u8,
}

impl UlaIo {
    pub fn new(initial_border: u8) -> Self {
        UlaIo { border: initial_border & 0x07 }
    }
}

impl IoHandler for UlaIo {
    fn port_read(&self, _port: u16) -> u8 {
        // No keyboard wired — report all keys released (bits 0–4 high).
        0xFF
    }

    fn port_write(&mut self, port: u16, val: u8) {
        if port & 0x01 == 0 {
            // Any even port address is the ULA port (bit 0 = 0).
            self.border = val & 0x07;
        }
    }
}
