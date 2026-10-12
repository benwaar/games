pub mod cpu;
pub mod memory;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, ImageData};
use wasm_bindgen::Clamped;

const SCREEN_W: u32 = 256;
const SCREEN_H: u32 = 192;
const CELL: u32 = 8;

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
            pixels[i] = v;
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
