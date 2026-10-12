pub mod cpu;
pub mod memory;
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
// M3 Emulator — wraps Spectrum for JS
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
