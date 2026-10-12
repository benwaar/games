use crate::cpu::Cpu;
use crate::memory::Memory;
use crate::ula;

/// T-states per 50Hz frame at 3.5 MHz.
const TSTATES_PER_FRAME: u32 = 69_888;

/// Top-level emulator state: CPU + Memory + ULA frame counter.
pub struct Spectrum {
    pub cpu: Cpu,
    pub memory: Memory,
    frame_count: u32,
    border_colour: u8,
}

impl Spectrum {
    pub fn new() -> Self {
        Spectrum {
            cpu: Cpu::new(),
            memory: Memory::new(),
            frame_count: 0,
            border_colour: 7, // white border until an OUT tells us otherwise
        }
    }

    /// Execute CPU instructions for approximately one 50Hz frame (~69 888 T-states).
    pub fn step_frame(&mut self) {
        let mut elapsed = 0u32;
        while elapsed < TSTATES_PER_FRAME {
            if self.cpu.halted {
                break;
            }
            elapsed += self.cpu.step(&mut self.memory) as u32;
        }
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Render the current screen memory into an RGBA pixel buffer.
    pub fn render(&self) -> Vec<u8> {
        ula::render_frame(&self.memory, self.frame_count)
    }

    /// Border colour index (0–7), sourced from the last OUT to port 0xFE bits 0–2.
    pub fn border_colour(&self) -> u8 {
        self.border_colour & 0x07
    }

    /// Called by the CPU OUT handler when port 0xFE is written — captures border colour.
    pub fn set_border(&mut self, val: u8) {
        self.border_colour = val & 0x07;
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Spectrum::new()
    }
}
