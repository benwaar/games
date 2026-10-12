# Next: M9 — Binary Protection PoC, OR M10 WASM Extension

M10 (ONNX → Z80 Inference Engine) is complete. The gate passes: 100% argmax agreement on 100 random inputs.

**What was built:**

New modules:
- `incant/onnx_parser.py` — parse ONNX Gemm/Relu graphs → {layers, weights, biases}
- `incant/quantise.py` — per-layer symmetric INT8 quantisation with correct >>8 scale propagation
- `incant/z80_infer.py` — deterministic Z80 asm generator: signed_mul (shift-add), matmul_layer, relu_buf, argmax_buf (signed comparison via overflow flag)

New CLI:

```bash
python -m incant infer tests/test_model.onnx --target z80
python -m incant infer tests/test_model.onnx --target z80 --accuracy
```

Test model fixture: `tests/test_model.onnx` (4→3→2 ReLU, deterministic seed 42).

114 tests pass.

**Two parallel paths forward:**

**M9 — Binary Protection PoC:**
Z80 self-verifying binary (XOR checksum) + WASM data-segment mirror. Uses the existing LLM pipeline (sigils). Good learning on binary analysis.

**M10 WASM extension:**
Same INT8 → WASM pipeline (WAT instead of Z80). Generates WAT `i32.add`, `i32.mul_s` for the dot product. Natural next step for M10.

**What's next: M9** — it's been queued since M8, and M10 is the only gap in the sequence.

**Steps:**
1. Z80 sigil: `self_verify.sigil.yaml` — XOR self-check over code region
2. Post-assembly checksum patching in Z80 gate
3. Test: modify one byte, verify check catches it
4. WAT sigil: `self_verify_wasm.sigil.yaml` — data-segment mirror + hash check
5. Test: modify data segment, verify check catches it
6. Explainer: worked examples in `explainers/binary-protection.md`
7. Document attacker bypass for each
