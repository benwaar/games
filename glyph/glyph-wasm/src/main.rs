use macroquad::prelude::*;

fn window_conf() -> Conf {
    Conf {
        window_title: "Glyph".to_string(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // For WASM: embed the image at compile time.
    // For desktop: accept a path via the first CLI argument, fall back to the embedded bytes.
    let texture = load_image_texture().await;

    loop {
        clear_background(BLACK);

        let (win_w, win_h) = (screen_width(), screen_height());
        let (img_w, img_h) = (texture.width(), texture.height());

        // Scale to fit window while preserving aspect ratio.
        let scale = (win_w / img_w).min(win_h / img_h);
        let draw_w = img_w * scale;
        let draw_h = img_h * scale;

        // Centre in window.
        let x = (win_w - draw_w) / 2.0;
        let y = (win_h - draw_h) / 2.0;

        draw_texture_ex(
            &texture,
            x,
            y,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(draw_w, draw_h)),
                ..Default::default()
            },
        );

        next_frame().await;
    }
}

async fn load_image_texture() -> Texture2D {
    // Embedded fallback image — used by WASM and desktop when no argument is given.
    const EMBEDDED: &[u8] = include_bytes!("../assets/embedded.jpg");

    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if let Some(path) = args.get(1) {
            match load_texture(path).await {
                Ok(t) => return t,
                Err(e) => eprintln!("failed to load '{}': {e} — using embedded image", path),
            }
        }
    }

    // Fall through: decode embedded bytes.
    let img = image::load_from_memory(EMBEDDED)
        .expect("embedded image is valid")
        .to_rgba8();
    let (w, h) = img.dimensions();
    Texture2D::from_rgba8(w as u16, h as u16, img.as_raw())
}
