# cast — Edge Optimisation

Take the trained Signal Hunt `SoundClassifier` and export it to lightweight deployment
formats. Apply quantisation. Benchmark until inference runs sub-100ms on-device with
no network round-trip.

---

## The idea

The Signal Hunt model is 25K parameters — already small. But "small" in training terms
is not the same as "fast" at inference on a mobile or edge device. ONNX and TFLite are
not just different file formats — they use graph-optimised runtimes that eliminate Python
overhead, fuse operations, and run on hardware accelerators. INT8 quantisation cuts the
model size by 4× and speeds up inference on integer-optimised hardware.

This is the difference between a research model and a deployed one.

---

## Prerequisites

- Signal Hunt Phase 2 complete — `output/best_model.pt` exists
- Python venv active

---

## Milestones

### M1 — ONNX export

- [ ] Export `SoundClassifier` to ONNX format using `torch.onnx.export`
- [ ] Verify export: run inference on the same test set via ONNX Runtime, check outputs match PyTorch
- [ ] Benchmark: PyTorch vs ONNX inference time on 100 samples

**Gate:** ONNX output matches PyTorch output within floating point tolerance. ONNX faster or equal.

**Docs:**
- [ ] Explainer: what ONNX is — portable graph format, framework-agnostic, runtime-optimised
- [ ] Note: why `torch.onnx.export` requires a dummy input (tracing vs scripting)

---

### M2 — TFLite export

- [ ] Convert ONNX model to TFLite via `onnx-tf` or `ai-edge-torch`
- [ ] Run inference via TFLite interpreter
- [ ] Compare accuracy and latency: PyTorch → ONNX → TFLite

**Gate:** TFLite model produces correct classifications. Latency documented.

**Docs:**
- [ ] Explainer: ONNX vs TFLite — where each runtime is used, trade-offs

---

### M3 — INT8 quantisation

- [ ] Apply post-training quantisation (PTQ) to the ONNX or TFLite model
- [ ] Measure accuracy drop: does the quantised model still hit >95% test accuracy?
- [ ] Measure size reduction: expect ~4× smaller
- [ ] Measure latency improvement on CPU

**Gate:** Quantised model within 2% accuracy of FP32. Size reduction documented.

**Docs:**
- [ ] Explainer: quantisation — float32 → int8, why 4× size, where precision loss comes from
- [ ] Explainer: post-training quantisation vs quantisation-aware training

---

### M4 — Latency benchmarking

- [ ] Benchmark all formats: PyTorch FP32, ONNX FP32, TFLite INT8
- [ ] Measure: mean latency, p95, p99 over 1000 inference calls
- [ ] Target: sub-100ms end-to-end (audio load + ingest + inference)
- [ ] Plot: latency distribution per format

**Gate:** At least one format hits sub-100ms end-to-end. Results documented.

**Docs:**
- [ ] Explainer: latency percentiles — why p95/p99 matters more than mean for real-time systems

---

### M5 — ExecuTorch (stretch)

- [ ] Export to ExecuTorch (PyTorch's mobile/edge runtime)
- [ ] Compare against TFLite INT8
- [ ] Document: when to use ExecuTorch vs TFLite

**Gate:** ExecuTorch model runs and produces correct output.

---

### M6 — CLI + demo

- [ ] `python -m cast export --format onnx|tflite|executorch`
- [ ] `python -m cast benchmark` — runs all formats, prints comparison table
- [ ] Update `demo.sh` to use the optimised model format

**Gate:** `bash demo.sh` uses the cast model. Benchmark table reproducible from cold clone.

**Docs:**
- [ ] README: commands, what each format is for, how to choose

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Model export (ONNX / TFLite) | ⬜ → ✅ |
| Quantisation (INT8) | ⬜ → ✅ |
| Edge deployment | ⬜ → ✅ |
| Latency benchmarking | ⬜ → ✅ |

---

## Feeds into

[echo](../echo/) — the optimised model is the inference engine echo runs in real time.
