pub mod cpu;
pub mod io;
pub mod keyboard;
pub mod memory;
pub mod tap;
pub mod ula;
pub mod spectrum;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, CanvasRenderingContext2d, ImageData};
use wasm_bindgen::Clamped;
use spectrum::Spectrum;

macro_rules! log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!("[zxw] {}", format!($($t)*)).into())
    };
}

const SCREEN_W: u32 = 256;
const SCREEN_H: u32 = 192;
const CELL: u32 = 8;

// --------------------------------------------------------------------------
// M3/M4 Emulator — wraps Spectrum for JS
// --------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Emulator {
    spectrum: Spectrum,
    ctx: CanvasRenderingContext2d,
}

#[wasm_bindgen]
impl Emulator {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<Emulator, JsValue> {
        let document = web_sys::window()
            .ok_or("no window")?
            .document()
            .ok_or("no document")?;

        let canvas: HtmlCanvasElement = document
            .get_element_by_id(canvas_id)
            .ok_or("canvas not found")?
            .dyn_into()?;

        canvas.set_width(SCREEN_W);
        canvas.set_height(SCREEN_H);

        let ctx = canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()?;

        Ok(Emulator {
            spectrum: Spectrum::new(),
            ctx,
        })
    }

    /// Load raw screen data (6912 bytes) into Spectrum memory at 0x4000.
    pub fn load_screen(&mut self, data: &[u8]) {
        self.spectrum.memory.load_ram(0x4000, data);
    }

    /// Load a 16K ROM binary into 0x0000–0x3FFF and mark it read-only.
    pub fn load_rom(&mut self, data: &[u8]) {
        log!("load_rom: {} bytes", data.len());
        self.spectrum.memory.load_rom(data);
        self.spectrum.detect_ld_bytes();
    }

    /// Hard reset: clear CPU + RAM, keep ROM and TAP queue. Like pressing the reset button.
    pub fn reset(&mut self) {
        log!("reset");
        self.spectrum.reset();
    }

    /// Parse a TAP file and queue its blocks for instant loading.
    ///
    /// Returns an error string on parse failure (surfaced as a JS exception).
    pub fn load_tap(&mut self, data: &[u8]) -> Result<(), JsValue> {
        log!("load_tap: {} bytes", data.len());
        let blocks = tap::parse_tap(data)
            .map_err(|e| JsValue::from_str(&e))?;
        for (i, b) in blocks.iter().enumerate() {
            log!(
                "  block {}: flag=0x{:02X} len={} {}",
                i, b.flag, b.data.len(),
                if b.flag == 0x00 { "HEADER" } else { "DATA" }
            );
        }
        log!("{} blocks queued for instant load", blocks.len());
        self.spectrum.queue_tap(blocks);
        Ok(())
    }

    /// Render current screen memory to the canvas.
    pub fn render_frame(&self) -> Result<(), JsValue> {
        let pixels = self.spectrum.render();
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&pixels),
            SCREEN_W,
            SCREEN_H,
        )?;
        self.ctx.put_image_data(&image_data, 0.0, 0.0)?;
        Ok(())
    }

    /// Advance CPU by one frame (~69 888 T-states) then render.
    /// No-ops on the CPU step if paused; still renders so display stays live.
    pub fn step_frame(&mut self) -> Result<(), JsValue> {
        self.spectrum.step_frame();
        self.render_frame()
    }

    /// Run CPU until HALT (or 10 million T-states as a safety cap), then render.
    /// Useful after loading a TAP to let the program boot before entering the RAF loop.
    pub fn run_until_halt(&mut self) -> Result<(), JsValue> {
        const MAX_TSTATES: u64 = 10_000_000;
        let mut elapsed = 0u64;
        while !self.spectrum.cpu.halted && elapsed < MAX_TSTATES {
            // Delegate entirely to step_frame which handles the tap trap.
            self.spectrum.step_frame();
            elapsed += 69_888; // approximate frame budget
            if self.spectrum.cpu.halted {
                break;
            }
        }
        self.render_frame()
    }

    // ------------------------------------------------------------------
    // M5: Keyboard input
    // ------------------------------------------------------------------

    /// Signal a Spectrum key pressed. `row` 0–7, `bit` 0–4.
    ///
    /// Row/bit layout:
    ///   0: SHIFT Z X C V      1: A S D F G      2: Q W E R T
    ///   3: 1 2 3 4 5          4: 0 9 8 7 6      5: P O I U Y
    ///   6: ENTER L K J H      7: SPACE SYM M N B
    pub fn key_down(&mut self, row: u8, bit: u8) {
        self.spectrum.io.keyboard.key_down(row as usize, bit as usize);
    }

    /// Signal a Spectrum key released. `row` 0–7, `bit` 0–4.
    pub fn key_up(&mut self, row: u8, bit: u8) {
        self.spectrum.io.keyboard.key_up(row as usize, bit as usize);
    }

    /// Signal a Kempston joystick bit pressed. bit 0=right, 1=left, 2=down, 3=up, 4=fire.
    pub fn kempston_down(&mut self, bit: u8) {
        self.spectrum.io.keyboard.kempston_down(bit as usize);
    }

    /// Signal a Kempston joystick bit released.
    pub fn kempston_up(&mut self, bit: u8) {
        self.spectrum.io.keyboard.kempston_up(bit as usize);
    }

    /// Reset the keyboard — all keys released.
    pub fn keyboard_reset(&mut self) {
        self.spectrum.io.keyboard.reset();
    }

    // ------------------------------------------------------------------
    // M7: Debug API
    // ------------------------------------------------------------------

    /// Read `len` bytes from memory at `addr`.
    pub fn read_memory(&self, addr: u16, len: u16) -> Vec<u8> {
        self.spectrum.memory.read_slice(addr, len)
    }

    /// Write `data` bytes into memory starting at `addr`.
    pub fn write_memory(&mut self, addr: u16, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            self.spectrum.memory.write_byte(addr.wrapping_add(i as u16), byte);
        }
    }

    /// Return all Z80 register values as a plain JS object.
    ///
    /// Keys: pc, sp, af, bc, de, hl, ix, iy, af2, bc2, de2, hl2,
    ///       i, r, iff1, iff2, im, halted.
    pub fn read_registers(&self) -> JsValue {
        let cpu = &self.spectrum.cpu;
        let obj = js_sys::Object::new();

        let set = |key: &str, val: u32| {
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str(key),
                &JsValue::from_f64(val as f64),
            ).unwrap_or_default();
        };

        set("pc",  cpu.pc as u32);
        set("sp",  cpu.sp as u32);
        set("af",  cpu.af() as u32);
        set("bc",  cpu.bc() as u32);
        set("de",  cpu.de() as u32);
        set("hl",  cpu.hl() as u32);
        set("ix",  cpu.ix as u32);
        set("iy",  cpu.iy as u32);
        set("af2", ((cpu.a2 as u32) << 8) | cpu.f2 as u32);
        set("bc2", ((cpu.b2 as u32) << 8) | cpu.c2 as u32);
        set("de2", ((cpu.d2 as u32) << 8) | cpu.e2 as u32);
        set("hl2", ((cpu.h2 as u32) << 8) | cpu.l2 as u32);
        set("i",   cpu.i as u32);
        set("r",   cpu.r as u32);
        set("iff1",   cpu.iff1 as u32);
        set("iff2",   cpu.iff2 as u32);
        set("im",     cpu.im as u32);
        set("halted", cpu.halted as u32);

        obj.into()
    }

    /// Pause CPU execution. The RAF loop still calls render_frame so the display stays live.
    pub fn pause(&mut self) {
        self.spectrum.paused = true;
    }

    /// Resume CPU execution.
    pub fn resume(&mut self) {
        self.spectrum.paused = false;
    }

    /// True when the emulator is paused (breakpoint hit or explicit pause).
    pub fn is_paused(&self) -> bool {
        self.spectrum.paused
    }

    /// Execute exactly one CPU instruction and re-render.
    /// Clears the paused flag first so the instruction runs even after a breakpoint.
    /// Re-pauses if the new PC lands on another breakpoint.
    pub fn step_single(&mut self) -> Result<(), JsValue> {
        self.spectrum.paused = false;
        self.spectrum.step_instruction();
        // Re-pause if the new PC is a breakpoint.
        if self.spectrum.breakpoints.contains(&self.spectrum.cpu.pc) {
            self.spectrum.paused = true;
        }
        self.render_frame()
    }

    /// Set the Z80 program counter directly.
    pub fn set_pc(&mut self, addr: u16) {
        self.spectrum.cpu.pc = addr;
        self.spectrum.paused = false;
    }

    /// Get the current Z80 program counter.
    pub fn get_pc(&self) -> u16 {
        self.spectrum.cpu.pc
    }

    /// Set a breakpoint at `addr`. Execution pauses before the instruction at that address.
    pub fn set_breakpoint(&mut self, addr: u16) {
        self.spectrum.breakpoints.insert(addr);
    }

    /// Remove a single breakpoint.
    pub fn clear_breakpoint(&mut self, addr: u16) {
        self.spectrum.breakpoints.remove(&addr);
    }

    /// Remove all breakpoints.
    pub fn clear_all_breakpoints(&mut self) {
        self.spectrum.breakpoints.clear();
    }

    // ------------------------------------------------------------------
    // M7: Watchpoint API
    // ------------------------------------------------------------------

    /// Watch a memory address for writes. Hits are recorded until drained.
    pub fn add_watchpoint(&mut self, addr: u16) {
        self.spectrum.memory.add_watchpoint(addr);
    }

    /// Stop watching an address.
    pub fn remove_watchpoint(&mut self, addr: u16) {
        self.spectrum.memory.remove_watchpoint(addr);
    }

    /// Remove all watchpoints.
    pub fn clear_watchpoints(&mut self) {
        self.spectrum.memory.clear_watchpoints();
    }

    /// Return accumulated watchpoint hits as a flat JS array: [addr, old, new, addr, old, new, ...].
    pub fn get_watchpoint_hits(&mut self) -> Vec<u32> {
        self.spectrum.memory
            .drain_watchpoint_hits()
            .iter()
            .flat_map(|&(addr, old, new)| [addr as u32, old as u32, new as u32])
            .collect()
    }
}

// --------------------------------------------------------------------------
// M1 checkerboard — kept so the M1 gate still passes
// --------------------------------------------------------------------------

#[wasm_bindgen]
pub fn init(canvas_id: &str) -> Result<(), JsValue> {
    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;

    let canvas: HtmlCanvasElement = document
        .get_element_by_id(canvas_id)
        .ok_or("canvas not found")?
        .dyn_into()?;

    canvas.set_width(SCREEN_W);
    canvas.set_height(SCREEN_H);

    let ctx = canvas
        .get_context("2d")?
        .ok_or("no 2d context")?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

    let mut pixels = vec![0u8; (SCREEN_W * SCREEN_H * 4) as usize];

    for y in 0..SCREEN_H {
        for x in 0..SCREEN_W {
            let cell_x = x / CELL;
            let cell_y = y / CELL;
            let bright = ((cell_x + cell_y) % 2 == 0) as u8;
            let v = if bright == 1 { 0xD7 } else { 0x00 };
            let i = ((y * SCREEN_W + x) * 4) as usize;
            pixels[i]     = v;
            pixels[i + 1] = v;
            pixels[i + 2] = v;
            pixels[i + 3] = 0xFF;
        }
    }

    let image_data = ImageData::new_with_u8_clamped_array_and_sh(
        Clamped(&pixels),
        SCREEN_W,
        SCREEN_H,
    )?;

    ctx.put_image_data(&image_data, 0.0, 0.0)?;

    Ok(())
}
