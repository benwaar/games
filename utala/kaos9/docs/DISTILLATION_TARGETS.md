# One Brain, Three Runtimes

TinyNN — a 5,727-parameter neural network distilled from a DQN teacher — runs on three wildly different targets. Same trained weights. Same 54% win rate against hand-crafted heuristics. Three deployment pipelines that couldn't be more different.

## At a glance

| | **Flutter** (ONNX/JSON) | **WASM** (browser) | **Z80** (Spectrum) |
|---|---|---|---|
| **Binary** | 25 KB JSON | 23 KB `.wasm` | 6,435 B `.tap` |
| **Precision** | float32 | float32 | INT8 |
| **Runtime memory** | ~50 KB (Dart heap) | 64 KB (1 WASM page) | ~6.3 KB of 64 KB RAM |
| **Inference** | sub-ms | 6.5 µs | ~6 s @ 3.5 MHz |
| **Argmax agreement** | reference | 100% | 100% |

---

## The model

| Property | Value |
|----------|-------|
| Architecture | 80 → 32 (ReLU) → 95 |
| Parameters | 5,727 |
| Training | Behavioural cloning from DQN v3 (93,149 decisions, 5,000 games) |
| vs Heuristic | **54%** — first learned agent to consistently beat the baseline |

The distilled student outperforms its 39,135-parameter teacher. Averaging the teacher's noisy, oscillating policy preserves strategy and discards noise.

---

## Target 1: ONNX → Flutter (mobile)

**25 KB JSON asset. Pure Dart inference. No ML framework on device.**

The model exports to JSON (float32 weights) and ONNX. The Flutter app loads the JSON asset and runs inference with two matrix multiplications and a ReLU — implemented directly in Dart. No TensorFlow Lite, no ONNX Runtime, no native dependencies.

| File | Size | Use |
|------|------|-----|
| `export/tiny_nn_f32.json` | ~25 KB | Dart pure-inference |
| `models/onnx/tiny_nn.onnx` | ~23 KB | ONNX runtime path |

**Why it matters:** Production mobile deployment with zero ML framework overhead. The model is small enough to ship as a text asset.

---

## Target 2: WASM (browser)

**6.5 µs per inference. 10,000 inferences in 65 ms.**

The same weights compiled to a WebAssembly module. Float32 precision. Runs in any modern browser with no server round-trip.

![WASM inference demo](../demo/screenshot_wasm_inference.png)

The [demo page](../demo/wasm_demo.html) loads the `.wasm` binary, takes an 80-float state vector, and returns an action index. The timing benchmark measures pure WASM compute — no JS overhead between calls.

| Property | Value |
|----------|-------|
| Precision | float32 |
| Binary | `demo/tiny_nn.wasm` |
| Inference | **6.5 µs** |
| 10K batch | 65 ms |

**Why it matters:** Client-side inference at microsecond latency. The same model that ships to mobile also runs in the browser — different binary, identical decisions.

---

## Target 3: Z80 assembly (8-bit retro hardware)

**6,435 bytes of Z80 machine code. Runs on a 3.5 MHz CPU with 64 KB RAM.**

The ONNX model is quantised to INT8 (per-layer symmetric quantisation), the weights are packed as Z80 `db` data blocks, and a hand-generated inference engine does fixed-point matrix multiplication using shift-and-add. The result runs on a real ZX Spectrum — or in the [zxw emulator](../../zxw/) built from scratch in Rust → WASM.

![TinyNN running on ZX Spectrum emulator](../demo/screenshot_z80_emulator.png)

The Spectrum screen shows "AI:8" — action index 8 selected from 95 possible actions. The same decision the float32 model would make, computed in fixed-point arithmetic on an 8-bit CPU.

| Property | Value |
|----------|-------|
| Precision | INT8 (quantised from float32) |
| Binary | `tiny_nn.tap` (Spectrum tape format) |
| Code size | 6,435 bytes (9.8% of 64 KB) |
| Inference | ~1.8M T-states ≈ **6 seconds** @ 3.5 MHz |
| Argmax agreement | **100%** (INT8 vs float32 reference) |

**Why it matters:** If it runs on a 3.5 MHz 8-bit CPU with 64 KB RAM, it runs anywhere. This is the extreme end of edge deployment — proof that the model is small enough and robust enough to survive aggressive quantisation and still make the right decisions.

---

## The pipeline

```
PyTorch (training)
  → DQN v3 teacher (39,135 params, 56% peak)
  → behavioural cloning (93,149 decisions)
  → TinyNN student (5,727 params, 54%)
      ├── JSON float32  → Flutter app (Dart inference)
      ├── WASM float32  → browser demo (6.5 µs)
      └── INT8 quantised → Z80 assembly → Spectrum TAP (6 seconds)
```

## What this demonstrates

- **Distillation works.** A 7× smaller student matches and beats its teacher.
- **Quantisation is survivable.** INT8 agrees with float32 on 100% of argmax decisions.
- **Deployment is a spectrum.** The same model ships to a modern phone, a browser tab, and a 1982 home computer — each target has its own pipeline, its own constraints, and its own definition of "fast enough."
- **Edge has no floor.** If your model fits in 6 KB, your deployment target is "anything with a CPU."
