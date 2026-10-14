# glyph-wasm

WASM image renderer for the glyph project. Implements the WASM side of the HAL principle: render the source image natively, full colour, no Spectrum constraints.

See `../explainers/hal-principle.md` for the full rationale.

## What it does

- Loads a PNG or JPEG image (embedded at compile time for WASM, or from a path argument on desktop)
- Opens a macroquad window
- Renders the image centred and scaled to fit, preserving aspect ratio

No game loop logic. No input handling. No colour quantisation. Just the image.

## Desktop

```bash
cargo run
# or with a custom image:
cargo run -- path/to/image.png
```

Falls back to the embedded `assets/loading.jpg` if no argument is given.

## WASM

```bash
./build.sh
# then serve the web/ directory:
python3 -m http.server 8080 --directory web
# open http://localhost:8080
```

The WASM build embeds `assets/loading.jpg` at compile time. To swap the image, replace that file and rebuild.

## Dependencies

- `macroquad 0.4` — window + rendering
- `image 0.25` — JPEG/PNG decoding for the embedded-bytes path
