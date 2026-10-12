use std::collections::HashSet;

use crate::cpu::{Cpu, FLAG_C};
use crate::io::IoHandler;
use crate::keyboard::Keyboard;
use crate::memory::Memory;
use crate::tap::TapBlock;
use crate::ula;

macro_rules! log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!("[zxw] {}", format!($($t)*)).into())
    };
}

/// T-states per 50Hz frame at 3.5 MHz.
const TSTATES_PER_FRAME: u32 = 69_888;

/// System variable ERR_NR: current error code (0xFF = no error).
const ERR_NR: u16 = 0x5C3A;
/// System variable PPC: current BASIC line number.
const PPC: u16 = 0x5C45;
/// System variable SUBPPC: current statement number within line.
const SUBPPC: u16 = 0x5C47;

const SPECTRUM_ERRORS: &[&str] = &[
    "0 OK",
    "1 NEXT without FOR",
    "2 Variable not found",
    "3 Subscript wrong",
    "4 Out of memory",
    "5 Out of screen",
    "6 Number too big",
    "7 RETURN without GOSUB",
    "8 End of file",
    "9 STOP statement",
    "A Invalid argument",
    "B Integer out of range",
    "C Nonsense in BASIC",
    "D BREAK - CONT repeats",
    "E Out of DATA",
    "F Invalid file name",
    "G No room for line",
    "H STOP in INPUT",
    "I FOR without NEXT",
    "J Invalid I/O device",
    "K Invalid colour",
    "L BREAK into program",
    "M RAMTOP no good",
    "N Statement lost",
    "O Invalid stream",
    "P FN without DEF",
    "Q Parameter error",
    "R Tape loading error",
];

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
    /// When true, step_frame is a no-op (CPU is paused at a breakpoint or by request).
    pub paused: bool,
    /// Breakpoint addresses — if PC matches, step_frame pauses before executing.
    pub breakpoints: HashSet<u16>,
    /// Last ERR_NR value — used to detect new errors.
    last_err_nr: u8,
    /// Detected LD_BYTES entry point — set when a ROM is scanned.
    ld_bytes_addr: Option<u16>,
}

impl Spectrum {
    pub fn new() -> Self {
        Spectrum {
            cpu: Cpu::new(),
            memory: Memory::new(),
            io: SpectrumIo::new(),
            frame_count: 0,
            tap_queue: Vec::new(),
            paused: false,
            breakpoints: HashSet::new(),
            last_err_nr: 0xFF,
            ld_bytes_addr: None,
        }
    }

    /// Queue TAP blocks for instant loading.
    pub fn queue_tap(&mut self, blocks: Vec<TapBlock>) {
        self.tap_queue = blocks;
    }

    /// Hard reset — clear CPU, RAM, and I/O state. ROM and TAP queue are preserved.
    pub fn reset(&mut self) {
        self.cpu = Cpu::new();
        self.memory.clear_ram();
        self.io = SpectrumIo::new();
        self.frame_count = 0;
        self.paused = false;
        self.last_err_nr = 0xFF;
    }

    /// Scan loaded ROM for the LD_BYTES entry point.
    /// Signature: INC D; EX AF,AF'; DEC D; DI; LD A,0x0F; OUT (0xFE),A
    /// Bytes: 14 08 15 F3 3E 0F D3 FE
    pub fn detect_ld_bytes(&mut self) {
        const SIG: [u8; 8] = [0x14, 0x08, 0x15, 0xF3, 0x3E, 0x0F, 0xD3, 0xFE];
        for addr in 0x0000..=0x3FF8u16 {
            let mut matched = true;
            for (i, &expected) in SIG.iter().enumerate() {
                if self.memory.read_byte(addr + i as u16) != expected {
                    matched = false;
                    break;
                }
            }
            if matched {
                self.ld_bytes_addr = Some(addr);
                log!("LD_BYTES detected at 0x{:04X}", addr);
                return;
            }
        }
        log!("LD_BYTES signature not found — instant-load trap disabled");
        self.ld_bytes_addr = None;
    }

    /// Execute CPU instructions for approximately one 50Hz frame (~69 888 T-states).
    /// No-ops when paused. Pauses automatically if a breakpoint is hit.
    pub fn step_frame(&mut self) {
        if self.paused {
            return;
        }
        let mut elapsed = 0u32;
        while elapsed < TSTATES_PER_FRAME {
            if self.cpu.halted {
                break;
            }

            // Breakpoint check: pause before executing the instruction at PC.
            if self.breakpoints.contains(&self.cpu.pc) {
                self.paused = true;
                return;
            }

            // Instant-load trap: intercept LD_BYTES before executing the ROM routine.
            if Some(self.cpu.pc) == self.ld_bytes_addr && !self.tap_queue.is_empty() {
                self.instant_load();
                continue;
            }

            // Split borrow: cpu, memory, and io are separate fields — Rust allows this.
            elapsed += self.cpu.step(&mut self.memory, &mut self.io) as u32;
        }

        // IM 1 interrupt: fire once per frame after ~69888 T-states if interrupts enabled.
        // The ROM's handler at 0x0038 updates the frame counter and scans the keyboard.
        if self.cpu.iff1 && self.cpu.im == 1 {
            self.cpu.fire_interrupt(&mut self.memory);
        }

        self.frame_count = self.frame_count.wrapping_add(1);
        self.check_rom_error();
    }

    /// Check ERR_NR system variable for new BASIC errors and log to console.
    fn check_rom_error(&mut self) {
        let err_nr = self.memory.read_byte(ERR_NR);
        if err_nr == self.last_err_nr {
            return;
        }
        self.last_err_nr = err_nr;

        // 0xFF = no error, 0x00 = "0 OK" (normal completion)
        if err_nr == 0xFF {
            return;
        }

        let msg = SPECTRUM_ERRORS
            .get(err_nr as usize)
            .unwrap_or(&"? Unknown error");

        let line = self.memory.read_word(PPC);
        let stmt = self.memory.read_byte(SUBPPC);

        if err_nr == 0x00 {
            // "0 OK" is not really an error — just log briefly
            log!("BASIC: {} at {}:{}", msg, line, stmt);
            return;
        }

        // Real error — full dump
        log!("══════════════════════════════════════════");
        log!("BASIC ERROR: {} at line {}:{}", msg, line, stmt);
        log!("──────────────────────────────────────────");
        log!(
            "  PC=0x{:04X}  SP=0x{:04X}  A=0x{:02X}  F=0x{:02X}",
            self.cpu.pc, self.cpu.sp, self.cpu.a, self.cpu.f
        );
        log!(
            "  BC=0x{:04X}  DE=0x{:04X}  HL=0x{:04X}",
            self.cpu.bc(), self.cpu.de(), self.cpu.hl()
        );
        log!(
            "  IX=0x{:04X}  IY=0x{:04X}  I=0x{:02X}  R=0x{:02X}",
            self.cpu.ix, self.cpu.iy, self.cpu.i, self.cpu.r
        );
        log!(
            "  A'=0x{:02X}  F'=0x{:02X}  BC'=0x{:04X}  DE'=0x{:04X}  HL'=0x{:04X}",
            self.cpu.a2, self.cpu.f2,
            (self.cpu.b2 as u16) << 8 | self.cpu.c2 as u16,
            (self.cpu.d2 as u16) << 8 | self.cpu.e2 as u16,
            (self.cpu.h2 as u16) << 8 | self.cpu.l2 as u16,
        );

        // Memory dump around PC
        let pc = self.cpu.pc;
        let start = pc.saturating_sub(8);
        let mut hex = String::new();
        for addr in start..start.wrapping_add(24) {
            if addr == pc {
                hex.push_str(&format!("[{:02X}]", self.memory.read_byte(addr)));
            } else {
                hex.push_str(&format!(" {:02X} ", self.memory.read_byte(addr)));
            }
        }
        log!("  mem @PC: 0x{:04X}: {}", start, hex);

        // BASIC program area dump (first 48 bytes from PROG)
        let prog_addr = self.memory.read_word(0x5C53); // PROG sysvar
        let mut prog_hex = String::new();
        for i in 0..48u16 {
            prog_hex.push_str(&format!(
                "{:02X} ",
                self.memory.read_byte(prog_addr.wrapping_add(i))
            ));
        }
        log!("  BASIC @0x{:04X}: {}", prog_addr, prog_hex);
        log!("══════════════════════════════════════════");
    }

    /// Execute exactly one CPU instruction (respects TAP trap, ignores breakpoints).
    /// Used by the debug step API — the caller checks for new breakpoints after calling.
    pub fn step_instruction(&mut self) {
        if self.cpu.halted {
            return;
        }
        if Some(self.cpu.pc) == self.ld_bytes_addr && !self.tap_queue.is_empty() {
            self.instant_load();
        } else {
            self.cpu.step(&mut self.memory, &mut self.io);
        }
    }

    /// Short-circuit the ROM's LD_BYTES tape-loading loop.
    ///
    /// At entry to 0x0556 the Z80 calling convention has:
    ///   IX  = destination address in RAM
    ///   DE  = number of bytes to load
    ///   A   = expected flag byte (0x00 = header, 0xFF = data)
    ///
    /// We pop the front block off the queue that matches the expected flag, copy
    /// its data to memory at IX, set the carry flag (success), then return via the
    /// address already on the stack (put there by the caller's CALL instruction).
    fn instant_load(&mut self) {
        // At LD_BYTES entry (0x0556), the ROM calling convention has:
        //   A  = expected flag byte (0x00 = header, 0xFF = data)
        //   IX = destination address
        //   DE = byte count
        // The ROM's EX AF,AF' happens AFTER entry — we intercept before.
        let expected_flag = self.cpu.a;

        log!(
            "instant_load trap at PC=0x{:04X}: expected_flag=0x{:02X} IX=0x{:04X} DE=0x{:04X} queue={}",
            self.cpu.pc, expected_flag, self.cpu.ix, self.cpu.de(), self.tap_queue.len()
        );

        // Find the first queued block whose flag matches what the ROM expects.
        let block_idx = self
            .tap_queue
            .iter()
            .position(|b| b.flag == expected_flag);

        let Some(idx) = block_idx else {
            log!("  no matching block for flag 0x{:02X} — signalling failure", expected_flag);
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

        log!(
            "  loading block idx={} flag=0x{:02X} ({} bytes) → 0x{:04X}..0x{:04X}",
            idx, block.flag, copy_len, dest, dest.wrapping_add(copy_len as u16)
        );

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
        log!("  load complete, returning to 0x{:04X}", ret);
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
