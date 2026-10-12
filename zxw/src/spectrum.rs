use crate::cpu::{Cpu, FLAG_C};
use crate::io::IoHandler;
use crate::keyboard::Keyboard;
use crate::memory::Memory;
use crate::tap::TapBlock;
use crate::ula;

/// T-states per 50Hz frame at 3.5 MHz.
const TSTATES_PER_FRAME: u32 = 69_888;

/// ROM address of the LD_BYTES routine entry point.
/// When the CPU's PC reaches this address we short-circuit tape loading.
const LD_BYTES: u16 = 0x0556;

/// All peripheral I/O for the Spectrum: keyboard matrix + border colour.
/// Implements `IoHandler` so the CPU can read port 0xFE (keyboard) and
/// port 0x1F (Kempston), and write port 0xFE (border colour).
pub struct SpectrumIo {
    pub keyboard: Keyboard,
    pub border_colour: u8,
}

impl SpectrumIo {
    pub fn new() -> Self {
        SpectrumIo {
            keyboard: Keyboard::new(),
            border_colour: 7, // white border until an OUT to 0xFE changes it
        }
    }
}

impl Default for SpectrumIo {
    fn default() -> Self {
        SpectrumIo::new()
    }
}

impl IoHandler for SpectrumIo {
    fn port_read(&self, port: u16) -> u8 {
        match (port & 0xFF) as u8 {
            0xFE => self.keyboard.read_port((port >> 8) as u8),
            0x1F => self.keyboard.read_kempston(),
            _ => 0xFF,
        }
    }

    fn port_write(&mut self, port: u16, val: u8) {
        // Port 0xFE low 3 bits control border colour.
        if (port & 0xFF) as u8 == 0xFE {
            self.border_colour = val & 0x07;
        }
    }
}

/// Top-level emulator state: CPU + Memory + I/O (keyboard, border) + TAP queue.
pub struct Spectrum {
    pub cpu: Cpu,
    pub memory: Memory,
    pub io: SpectrumIo,
    frame_count: u32,
    /// Pending TAP blocks — popped one at a time by the instant-load trap.
    pub tap_queue: Vec<TapBlock>,
}

impl Spectrum {
    pub fn new() -> Self {
        Spectrum {
            cpu: Cpu::new(),
            memory: Memory::new(),
            io: SpectrumIo::new(),
            frame_count: 0,
            tap_queue: Vec::new(),
        }
    }

    /// Queue TAP blocks for instant loading.
    pub fn queue_tap(&mut self, blocks: Vec<TapBlock>) {
        self.tap_queue = blocks;
    }

    /// Execute CPU instructions for approximately one 50Hz frame (~69 888 T-states).
    pub fn step_frame(&mut self) {
        let mut elapsed = 0u32;
        while elapsed < TSTATES_PER_FRAME {
            if self.cpu.halted {
                break;
            }

            // Instant-load trap: intercept LD_BYTES before executing the ROM routine.
            if self.cpu.pc == LD_BYTES && !self.tap_queue.is_empty() {
                self.instant_load();
                continue;
            }

            // Split borrow: cpu, memory, and io are separate fields — Rust allows this.
            elapsed += self.cpu.step(&mut self.memory, &mut self.io) as u32;
        }
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Short-circuit the ROM's LD_BYTES tape-loading loop.
    ///
    /// At entry to 0x0556 the Z80 calling convention has:
    ///   IX  = destination address in RAM
    ///   DE  = number of bytes to load
    ///   A'  = expected flag byte (0x00 = header, 0xFF = data)
    ///
    /// We pop the front block off the queue that matches the expected flag, copy
    /// its data to memory at IX, set the carry flag (success), then return via the
    /// address already on the stack (put there by the caller's CALL instruction).
    fn instant_load(&mut self) {
        let expected_flag = self.cpu.a2; // alternate A register

        // Find the first queued block whose flag matches what the ROM expects.
        let block_idx = self
            .tap_queue
            .iter()
            .position(|b| b.flag == expected_flag);

        let Some(idx) = block_idx else {
            // No matching block — signal failure (carry clear) and return.
            self.cpu.set_flag(FLAG_C, false);
            let ret = self.memory.read_word(self.cpu.sp);
            self.cpu.sp = self.cpu.sp.wrapping_add(2);
            self.cpu.pc = ret;
            return;
        };

        let block = self.tap_queue.remove(idx);

        // Copy data to IX, honouring the DE byte count.
        let dest = self.cpu.ix;
        let count = self.cpu.de() as usize;
        let src = &block.data;
        let copy_len = count.min(src.len());

        for i in 0..copy_len {
            let addr = dest.wrapping_add(i as u16);
            self.memory.write_byte(addr, src[i]);
        }

        // Set carry = 1 (load succeeded).
        self.cpu.set_flag(FLAG_C, true);

        // Pop the return address that CALL pushed and resume there.
        let ret = self.memory.read_word(self.cpu.sp);
        self.cpu.sp = self.cpu.sp.wrapping_add(2);
        self.cpu.pc = ret;
    }

    /// Render the current screen memory into an RGBA pixel buffer.
    pub fn render(&self) -> Vec<u8> {
        ula::render_frame(&self.memory, self.frame_count)
    }

    /// Border colour index (0–7), sourced from the last OUT to port 0xFE bits 0–2.
    pub fn border_colour(&self) -> u8 {
        self.io.border_colour
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Spectrum::new()
    }
}
