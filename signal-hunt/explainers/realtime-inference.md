# Real-Time Inference

How the browser demo classifies audio continuously, without ever pausing to wait for a complete recording.

## The problem with batch classification

The trained model expects a fixed-size input: `(1, 1, 128, 65)` — one spectrogram, 128 Mel bands, 65 time frames. That represents about 1.6 seconds of audio:

```
65 frames × 512 samples/frame = 33280 samples
33280 / 22050 Hz ≈ 1.51 seconds
```

In batch mode (the training setup) you load a recording, extract one spectrogram, run inference once. Real-time is different — audio never stops arriving. The solution is a **sliding window**.

## Sliding window

The demo keeps a ring buffer of exactly `BUFFER_SAMPLES = (65 - 1) × 512 + 2048 = 35328` samples — the minimum needed to produce 65 STFT frames with librosa's centre-padding.

Every 512 samples (~23ms), the `ScriptProcessor` fires and:

```
[  old audio  |  old audio  |  ...  |  new chunk  ]
              ↑ shift left 512       ↑ append here
```

1. Shift the whole buffer left by 512 (`copyWithin`)
2. Write the new 512-sample chunk at the end
3. Run inference on the full buffer

The buffer always contains exactly the **last 1.6 seconds** of audio. The window slides forward 512 samples at a time.

```
t=0ms:   [  silence  |  silence  |  note starts  ]  → "listening..."
t=23ms:  [  silence  |  note starts  |  note body  ]  → prediction appears
t=46ms:  [  note body  |  note body  |  note body  ]  → confident prediction
t=1600ms:[  note body  |  note ends  |  silence   ]  → fades out
```

## Why this feels instant

The callback fires every 512/22050 ≈ **23 ms**. Human perception of audio lag kicks in around 100ms. So predictions update ~4× faster than you'd notice a delay.

The note doesn't need to finish playing — as soon as enough of it is in the buffer to dominate the spectrogram, the model reacts.

## The pipeline per callback

```
onaudioprocess (every ~23ms)
  │
  ├── shift buffer, append new chunk
  │
  ├── RMS energy check  ──── below threshold → show "—", return
  │
  ├── extractFeatures(buffer, filterbank)     ~10ms
  │     ├── centre-pad signal (n_fft/2 each side)
  │     ├── STFT: 65 frames × 2048-point FFT
  │     ├── Mel filterbank dot product → (128, 65)
  │     ├── power_to_db + top_db=80 clamp
  │     └── per-spectrogram normalise
  │
  ├── ort.InferenceSession.run(tensor)        ~2ms
  │     └── CNN forward pass (WASM)
  │
  └── update UI: note label, bars, timing     <1ms
```

Total per-callback: **~12ms**. The callback fires every ~23ms, so there's ~11ms headroom before the next one arrives.

## What "real-time" actually means here

The model doesn't classify individual moments. It classifies **the last 1.6 seconds** on every tick. This has consequences:

- **Onset lag** — a new note takes a few ticks to dominate the buffer over old audio
- **Tail persistence** — a note continues influencing predictions for up to 1.6s after it stops
- **No silence class** — the energy gate handles silence explicitly; the model never sees a near-zero spectrogram

For a piano teacher use case this is fine. The player holds a note; the model has 1.6s of clean signal and converges confidently. Fast chord changes would need a shorter window and a model trained on shorter clips.

## In practice

This sliding-window pattern appears in almost every real-time audio ML system — keyword spotting (Siri, Alexa), wake-word detection, live music transcription. The window size is a tradeoff:

| Shorter window | Longer window |
|---------------|---------------|
| Lower latency | More context for the model |
| Less signal for features | Slower to clear after a note ends |
| Needs model trained on short clips | Can use standard training data |

The 1.6s window here was chosen to match the training data (Iowa piano recordings are 1–3s). Changing the window without retraining would break the model.

→ See also: [browser-pipeline.md](browser-pipeline.md) — implementation details, ONNX export, Mel spectrogram gotchas.
