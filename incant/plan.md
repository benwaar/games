# Incant Phase 3 — Signal Hunt Browser Pipeline

**Goal:** Real-time mic → Mel spectrogram → CNN → note label in the browser. Target: < 20 ms end-to-end.

Uses the Signal Hunt transfer model (12-class piano note classifier) as the inference target.

---

## Prerequisites

- Signal Hunt `.venv` with `torch`, `onnx`, `onnxruntime` (already exists at `signal-hunt/.venv`)
- Node 20 for any JS tooling (`nvm use 20`)
- A browser with WebAssembly + Web Audio API support (any modern browser)
- `onnxruntime-web` loaded via CDN in the demo HTML — no build step needed

---

## Milestones

### M11 — Export Signal Hunt CNN to ONNX

- [x] Verify no `.onnx` file exists in `signal-hunt/output/`
- [x] Write `signal-hunt/scripts/export_onnx.py`:
  - Load `output/transfer/finetune/best_model.pt`
  - Reconstruct `SoundClassifier(num_classes=12)` and load state dict
  - `model.eval()` — required before export (affects BatchNorm, Dropout)
  - Create dummy input `torch.randn(1, 1, 128, 65)`
  - `torch.onnx.export(...)` → `signal-hunt/output/chord_cnn.onnx`
  - Export with `opset_version=18`, `dynamic_axes` on batch dim
- [x] Gate: input shape `[batch_size, 1, 128, 65]`, output `[batch_size, 12]` ✅
- [x] Write test `signal-hunt/tests/test_export_onnx.py` — 4 tests pass ✅

**Commit:** `feat(signal-hunt): export CNN to ONNX (M11)`

---

### M12 — Browser Mel Spectrogram (JS)

- [x] Study `signal-hunt/pipeline/features.py`: n_fft=2048, hop_length=512, n_mels=128, log-dB (power_to_db), per-spectrogram normalise, pad/truncate to 65 frames
- [x] Create `signal-hunt/demo/` directory
- [x] Write `signal-hunt/demo/browser_pipeline.js` — standalone ES module:
  - Radix-2 FFT (Float64 for precision), Hann window, Slaney Mel filterbank
  - log-dB with `top_db=80` clamping, per-spectrogram normalise
  - Pad/truncate to 65 frames → Float32Array `[128 × 65]`
- [x] Write `signal-hunt/demo/spectrogram_check.html` — generates 440 Hz sine in JS, validates against Python reference stats
- [x] Gate: 3/3 PASS — frame count, normed mean ≈ 0, normed std ≈ 1 ✅ (~13 ms pipeline time)

**Commit:** `feat(signal-hunt): JS Mel spectrogram for browser pipeline (M12)`

---

### M13 — End-to-end classify + timing

- [x] Copy `signal-hunt/output/chord_cnn.onnx` into `signal-hunt/demo/`
- [x] Write `signal-hunt/demo/index.html` — end-to-end demo:
  - Load `chord_cnn.onnx` via `ort.InferenceSession` (onnxruntime-web CDN)
  - Capture mic audio, compute Mel spectrogram, run inference
  - Display: top-1 note label + confidence percentage
  - Timing display: pipeline time in ms (`performance.now()`)
  - Threshold: show "listening..." when confidence < 0.6
- [x] Gate: end-to-end < 20 ms on a test tone ✅ (~11–13 ms warm); correct note label for A4 (440 Hz) test tone
- [x] Write explainer: `signal-hunt/explainers/browser-pipeline.md` — how it works, perf numbers, key design decisions

**Commit:** `feat(signal-hunt): end-to-end browser inference demo (M13)`

---

## Notes

- `chord_cnn.onnx` is gitignored (binary model file). The `export_onnx.py` script regenerates it from the checkpoint.
- The `.onnx` file used in `demo/` is a local copy from `output/` — not the source.
- ONNX opset 17 is the minimum for `onnxruntime-web` 1.18+.
- The Mel filterbank in JS must use the same HTK formula as librosa (fmin=0, fmax=sr/2, norm=None).
