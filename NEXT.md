# Next

## WASM inference engine

- Add `--target wasm` to incant infer — WAT code generator with `i32.mul`/`i32.add` for matmul
- Demo KAOS 9 TinyNN in WASM (should be ~50µs vs ~6M T-states on Z80)
- Signal Hunt CNN in WASM — real-time mic → spectrogram → classify (under 20ms end-to-end)

## Signal Hunt on Spectrum

- 25.3 KB model fits on 48K (not 16K — 2.7× too big)
- Needs conv2d support in the Z80 inference engine
- ~30 seconds per classification — not real-time, but it works
- Would be the first CNN running on a Spectrum

## Binary protection demos

- Load self-verifying binary on zxw, tamper via debug API, show the crash
- POK file support — classic Spectrum cheat format, ties into the debug API
- Same demo on WASM — tamper data segment, show detection + bypass

## zxw UI polish

- Explainers for Spectrum modes (flashing K cursor, LOAD, etc.)
- Organise buttons, add debug panel alongside the screen
- Nice but simple UI on both zxw and incant demos