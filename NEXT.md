# Next

Remaining work before starting Acoustic Odyssey or Artefact.

---

## 1. Incant Phase 3 — Signal Hunt browser pipeline

Real-time mic → Mel spectrogram → CNN → chord label in the browser. Target: < 20 ms end-to-end.

Full plan: [incant/plan.md](incant/plan.md)

### Steps (Sonnet — use `/research` then implement)

**M7 — Export Signal Hunt CNN to ONNX**
1. `/research` — check `signal-hunt/output/` for an existing `.onnx` export
2. If missing: write `signal-hunt/scripts/export_onnx.py` — load best checkpoint, `torch.onnx.export`
3. Gate: `python -m onnxruntime signal-hunt/output/chord_cnn.onnx` loads without error

**M8 — Browser spectrogram (JS)**
1. `/research` — how Signal Hunt computes Mel spectrograms (`signal-hunt/pipeline/features.py`), what shape/params the model expects
2. Create `signal-hunt/demo/browser_pipeline.html` — Web Audio API (`getUserMedia` → `AudioContext`)
3. Compute 128-band Mel spectrogram in JS (use `meyda.js` or minimal FFT)
4. Pad/truncate to model's expected frame count
5. Gate: Mel values (console) match Python `features.py` output within 10% on a test WAV

**M9 — End-to-end classify + timing**
1. `/research` — `onnxruntime-web` API for loading ONNX in browser
2. Load `chord_cnn.onnx` via `ort.InferenceSession` (ort-web)
3. Feed spectrogram tensor → model → top-1 label + confidence in UI
4. Measure pipeline time with `performance.now()`: capture → spectrogram → infer → display
5. Gate: end-to-end < 20 ms; correct chord label for a test tone

---

## 2. zxw — Binary protection demos

Showcase incant's self-verifying binaries running on the zxw emulator, with live tampering via the CDP debug API.

### Steps (Sonnet — use `/research` then implement)

**Demo 1 — Z80 self-verifying binary on zxw**
1. `/research` — review `incant/incant/protect.py` (XOR checksum self-verification) and `zxw/` CDP debug API
2. Load incant's self-verifying `.tap` onto zxw
3. Show it running clean, then tamper a byte via CDP `poke` command
4. Capture the crash/halt — screenshot or automated test
5. Write up in a demo script or doc

**Demo 2 — POK file support**
1. `/research` — ZX Spectrum POK file format (classic cheat/poke format)
2. Implement POK parser in zxw — read POK files and apply pokes on load
3. Tie into the CDP debug API as an alternative poke source
4. Test with a known game POK

**Demo 3 — WASM self-verifying module**
1. `/research` — review `incant/incant/protect.py` WASM data-segment mirror protection
2. Load the WASM self-verifying module in browser
3. Tamper the data segment, show detection
4. Document the bypass (patch constant, intercept load)

---

## 3. Log entries

Update LOG.md for both projects with what was completed this session:

- [ ] `incant/LOG.md` — WASM Phase 1–2 already logged; add any missing detail
- [ ] `zxw/LOG.md` — note the binary protection demo work once complete
- [ ] Delete stale `utala/kaos9/demo/screenshot.png` (old WASM screenshot, renamed)
