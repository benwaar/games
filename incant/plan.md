# WASM Inference Engine — plan

**Goal:** Real-time Signal Hunt CNN inference in the browser.

Phase 1 (M1–M4, WAT generator) and Phase 2 (M5–M6, KAOS 9 demo) are complete — see [LOG.md](LOG.md).

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

## Follow-up (post-plan)

- Benchmark f64 WAT (better precision, same perf on 64-bit WASM engines)
- SIMD WAT (`f32x4.mul`) for batched inference — requires `--enable-simd` in wat2wasm
- Signal Hunt: adaptive re-training loop in JS (online learning in the browser)
