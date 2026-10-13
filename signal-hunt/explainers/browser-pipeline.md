# Browser Inference Pipeline

Real-time piano note classification running entirely in the browser: mic → Mel spectrogram → CNN → note label.

## What it does

- Captures audio from the microphone via the Web Audio API
- Computes a Mel spectrogram in JavaScript, matching the Python training pipeline exactly
- Runs the exported CNN through `onnxruntime-web` (WASM backend)
- Displays the predicted note, confidence, and timing breakdown

**Measured performance (warm, Chrome):** ~9–10 ms Mel, ~2–3 ms inference, **~11–13 ms total** — well inside the 20 ms target.

## Architecture

```
AudioContext (22050 Hz)
  └── ScriptProcessor (512-sample chunks)
        └── ring buffer (65 frames × 512 + 2048 = 35328 samples)
              └── extractFeatures()          ← browser_pipeline.js
                    └── ort.InferenceSession  ← chord_cnn.onnx (WASM)
                          └── softmax + top-1 label → UI
```

## The Mel spectrogram pipeline

The JS implementation in `browser_pipeline.js` mirrors `pipeline/features.py` step for step. The key choices that have to match exactly between training and inference:

### 1. STFT with centre-padding

librosa's `stft()` defaults to `center=True`, which pads the signal by `n_fft/2 = 1024` zeros on each side before computing frames. Without this, a 1-second signal at 22050 Hz produces 40 frames instead of 44.

```javascript
const pad = N_FFT >> 1;  // 1024
const centered = new Float32Array(signal.length + 2 * pad);
centered.set(signal, pad);
```

### 2. Slaney Mel scale (not HTK)

librosa uses the Slaney scale by default (`htk=False`). It's piecewise:
- **Below 1000 Hz**: linear — `mel = hz / (200/3)`
- **Above 1000 Hz**: logarithmic — `mel = minLogMel + log(hz / 1000) / logStep`

The two formulas meet at 1000 Hz. HTK uses a single log formula throughout — getting this wrong shifts all the filter centres.

### 3. Slaney normalisation

Triangular filters are built in Hz space (not bin space), then multiplied by `2 / bandwidth`. librosa calls this `norm='slaney'`. Without the `2/bandwidth` factor, filter values are ~2.7× too small, which shifts the log-dB scale.

### 4. `top_db=80` clamping

`librosa.power_to_db` defaults to `top_db=80`, clamping all values to `max_db - 80`. This is easy to miss because the parameter is buried in defaults. Without it, silent bins sit at `-100 dB` rather than `-39 dB` for a typical signal, completely changing the pre-normalisation distribution.

```javascript
const dbFloor = dbMax - 80;
for (let i = 0; i < logMel.length; i++) {
  if (logMel[i] < dbFloor) logMel[i] = dbFloor;
}
```

### 5. Float64 FFT

The Radix-2 FFT uses `Float64Array` internally. Sidelobe values for a pure sine are very small (~1e-14). Float32 underflows them to zero, which shifts the Mel values slightly.

## ONNX export for the browser

`onnxruntime-web`'s WASM backend cannot access external `.onnx.data` files. PyTorch's exporter splits large tensors to external files by default. The fix is to reload and re-save the model after export:

```python
model_proto = onnx.load(str(output))
onnx.save(model_proto, str(output), save_as_external_data=False)
```

This produces a single self-contained `.onnx` file (~122 KB for this model).

## Files

| File | Purpose |
|------|---------|
| `demo/browser_pipeline.js` | Mel spectrogram pipeline (ES module) |
| `demo/index.html` | Live demo UI |
| `demo/chord_cnn.onnx` | Exported model (inlined, no external data) |
| `demo/spectrogram_check.html` | Validates JS pipeline against Python reference |
| `scripts/export_onnx.py` | Exports `best_model.pt` → `chord_cnn.onnx` |

## Running locally

```bash
cd signal-hunt/demo
python -m http.server 8312
# open http://localhost:8312/index.html
```

A browser serving `file://` URLs won't work — the WASM runtime needs HTTP.

## In practice

This pattern — export a small model once, ship it as a static file, run inference in the browser with WASM — works for any model under ~5 MB where low-latency, on-device inference matters. No server round-trips, no GPU required, no user data leaving the device. The main constraint is that the JS preprocessing must match training exactly, which requires careful replication of every librosa default.
