# WASM Inference Engine — plan

**Goal:** Add `--target wasm` to `incant infer`, demo KAOS 9 TinyNN in WASM, then build a real-time browser pipeline for Signal Hunt.

---

## Prerequisites

All already available:
- `wat2wasm` v1.0.42 (brew `wabt`)
- Node v20 (for running WASM in tests via `wasm-interp` or a simple JS harness)
- Python 3.12 + venv
- `tiny_nn_f32.json` — TinyNN weights (80→32→95 f32)
- `models/onnx/tiny_nn.onnx` — KAOS 9 TinyNN ONNX

New for Phase 3 only:
- `onnxruntime-web` (npm) — runs Signal Hunt ONNX in the browser via WASM

---

## Phase 1 — WAT inference engine (`incant infer --target wasm`)

Mirrors `z80_infer.py` but emits f32 WAT instead of INT8 Z80 assembly.
No LLM — deterministic code generator from weights.

### M1 — `wasm_infer.py`: f32 WAT matmul kernel ✅
- [x] Create `incant/wasm_infer.py`
- [x] `generate_wat_inference(parsed) -> str` — reads parsed ONNX layers (same dict as z80_infer),
  emits a self-contained WAT module with:
  - Linear memory (exported), data segments for weights/biases
  - `$fc` helper: generic dense layer loop (`out[j] = bias[j] + Σ in[i] * w[j*in_dim + i]`)
  - `$relu_buf`, `$argmax` helpers
  - Exported `infer` function: runs layer 1 (+ ReLU) then layer 2, returns argmax as i32
- **Gate:** ✅ `wat2wasm` assembles without error; 100/100 agreement on test_model.onnx

Memory layout is dynamic (computed from layer dims). For TinyNN (80→32→95) fits in 1 WASM page.

### M2 — accuracy test for WAT infer ✅
- [x] `tests/test_wasm_infer.py` — 100 random inputs, checks argmax agreement ≥ 95% vs onnxruntime
- [x] `tests/wasm_infer_runner.js` — Node.js shim: writes f32 bytes to WASM memory, calls infer()
- **Gate:** ✅ `pytest tests/test_wasm_infer.py` passes; 100% agreement on test_model.onnx

### M3 — `--target wasm` in CLI ✅
- [x] Add `wasm` to `choices` in `infer_parser` in `__main__.py`
- [x] Wire `cmd_infer` to call `generate_wat_inference` when `args.target == "wasm"`,
  assemble with `wat_target.assemble()`, write `.wat` + `.wasm` to output dir
- **Gate:** ✅ `python -m incant infer tests/test_model.onnx --target wasm` exits 0 and writes `output/test_model.wasm`

### M4 — update docs + tick ✅
- [x] Added `incant/explainers/wasm-inference.md` — covers pipeline, memory layout, WAT data segments, $fc/$relu_buf/$argmax, testing strategy, Z80 vs WASM comparison
- [x] Commit + push

---

## Phase 2 — KAOS 9 TinyNN WASM demo

A standalone HTML page. No framework — plain JS + the compiled WASM.

### M5 — demo page skeleton
- [ ] Create `utala/kaos9/demo/wasm_demo.html`
- [ ] Loads `tiny_nn.wasm` via `WebAssembly.instantiateStreaming`
- [ ] Writes 80 floats into WASM linear memory (input buffer at offset 0)
- [ ] Calls exported `infer`, reads result (i32 argmax)
- [ ] Displays result in the page
- **Gate:** Open in browser, page shows argmax for a hardcoded input

### M6 — timing benchmark
- [ ] Run 10,000 inferences in a loop, use `performance.now()` to measure wall time
- [ ] Display: `WASM: Xµs/inference` vs `Z80: ~6,300,000 T-states (~6s @ 3.5MHz)`
- [ ] Add a table showing params, ops, and time
- **Gate:** WASM shows < 100µs/inference (expect ~1–5µs); Z80 estimate shown for contrast

---

## Phase 3 — Signal Hunt browser pipeline

Real-time mic → Mel spectrogram → CNN → chord label. Target: < 20ms end-to-end.

Signal Hunt's CNN has Conv2d layers — too complex to hand-write in WAT.
Use `onnxruntime-web` which runs ONNX inference via WASM under the hood.

### M7 — export Signal Hunt CNN to ONNX (if not already)
- [ ] Check `signal-hunt/output/` for an existing `.onnx`
- [ ] If missing: `scripts/export_onnx.py` — load best checkpoint, export via `torch.onnx.export`
- **Gate:** `python -m onnxruntime signal-hunt/output/chord_cnn.onnx` loads without error

### M8 — browser spectrogram (JS)
- [ ] `signal-hunt/demo/browser_pipeline.html`
- [ ] Web Audio API: `getUserMedia` → `AudioContext` → `AnalyserNode` or `AudioWorklet`
- [ ] Compute 128-band Mel spectrogram in JS (use `meyda.js` or a minimal FFT implementation)
- [ ] Pad/truncate to the model's expected frame count
- **Gate:** Mel spectrogram values (printed to console) match Python `pipeline/features.py`
  output within 10% on a test WAV file

### M9 — end-to-end classify + timing
- [ ] Load `chord_cnn.onnx` via `ort.InferenceSession` (ort-web)
- [ ] Feed spectrogram tensor into model, get label
- [ ] Display top-1 prediction in the UI with confidence
- [ ] Measure pipeline time with `performance.now()`: capture → spectrogram → infer → display
- **Gate:** End-to-end < 20ms (target); shows correct chord label for a test tone

---

## Session targets

- **Phase 1 (M1–M4):** Core WAT generator — self-contained, testable, no browser needed
- **Phase 2 (M5–M6):** Demo page — 2–3 hours; browser only
- **Phase 3 (M7–M9):** Full pipeline — most complex; can be a separate session

## Follow-up (post-plan)

- Benchmark f64 WAT (better precision, same perf on 64-bit WASM engines)
- SIMD WAT (`f32x4.mul`) for batched inference — requires `--enable-simd` in wat2wasm
- Signal Hunt: adaptive re-training loop in JS (online learning in the browser)
