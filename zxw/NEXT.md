# Next: M1 — Scaffold + WASM canvas hello world

Set up the Rust project, WASM build pipeline, and a minimal module that renders a checkerboard to an HTML canvas.

## What to do

1. Create `Cargo.toml` — lib crate with `cdylib` target, dependencies: `wasm-bindgen`, `web-sys` (Canvas features), `js-sys`
2. Write `setup.sh` — check/install rustup, add `wasm32-unknown-unknown` target, install `wasm-pack`
3. Implement `src/lib.rs` — `#[wasm_bindgen] pub fn init(canvas_id: &str)` that gets a canvas context and draws a 256×192 checkerboard
4. Add `www/index.html` — `<canvas>` element + `<script type="module">` that imports and calls `init()`
5. Add `Makefile` — `build` (wasm-pack build --target web) and `serve` (python3 -m http.server) targets

## Gate

`make build && make serve` → open localhost:8080 → checkerboard renders on the canvas, no errors in browser console.

## Reference

- Spectrum screen dimensions: 256×192 pixels
- wasm-pack target: `--target web` (ES module output)
- web-sys features needed: `HtmlCanvasElement`, `CanvasRenderingContext2d`, `ImageData`, `Document`, `Window`
