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
        self.spectrum.memory.load_rom(data);
    }

    /// Parse a TAP file and queue its blocks for instant loading.
    ///
    /// Returns an error string on parse failure (surfaced as a JS exception).
    pub fn load_tap(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let blocks = tap::parse_tap(data)
            .map_err(|e| JsValue::from_str(&e))?;
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
