# Next

## WASM inference engine

Full plan in [incant/plan.md](incant/plan.md) — 3 phases, 9 milestones.
Branch: `feature/no-ticket-wasm-inference-engine`

### What to build next: M1 — `wasm_infer.py`

Use `/research` and start from the first unchecked item in `incant/plan.md`.

**Context for a cold start:**

`incant` is a spec-driven code generator at `games/incant/`. It already has:
- `incant/incant/z80_infer.py` — deterministic Z80 asm generator from quantised INT8 weights
- `incant/incant/onnx_parser.py` — parses ONNX, returns `{"layers": [...], "input_shape": [...]}` where each linear layer has `weight: np.ndarray (out×in, f32)` and `bias: np.ndarray (out, f32)`
- `incant/incant/targets/wat.py` — `assemble(source: str) -> (bool, Path|None, str|None)` using `wat2wasm`
- `--target z80` working in `python -m incant infer model.onnx`

**M1 task:** Create `incant/incant/wasm_infer.py` with a single public function:

```python
def generate_wat_inference(parsed: dict) -> str:
    """Emit a self-contained WAT module from an ONNX parse result.
    
    Supports: any 2-layer ReLU network (linear → relu → linear).
    No LLM. Deterministic template + data emitter, like z80_infer.py.
    Returns WAT source as a string.
    """
```

**WAT module shape to emit:**

```wat
(module
  (memory 1)
  ;; data segments: fc1_weights, fc1_bias, fc2_weights, fc2_bias at fixed offsets
  (data (i32.const 828) "...f32 bytes...")   ;; fc1 weights  80×32 = 10240 bytes
  (data (i32.const 11068) "...")             ;; fc1 bias     32×4  = 128 bytes
  (data (i32.const 11196) "...")             ;; fc2 weights  32×95 = 12160 bytes
  (data (i32.const 23356) "...")             ;; fc2 bias     95×4  = 380 bytes

  ;; buffers in low memory (caller writes input here before calling infer):
  ;;   offset   0: input  (in_dim  × 4 bytes f32)
  ;;   offset 320: hidden (hid_dim × 4 bytes f32)
  ;;   offset 448: output (out_dim × 4 bytes f32)

  (func $relu (param f32) (result f32)
    local.get 0
    f32.const 0.0
    f32.max)

  (func $fc (param $in_ptr i32) (param $out_ptr i32)
            (param $w_ptr i32) (param $b_ptr i32)
            (param $in_dim i32) (param $out_dim i32)
    ;; for j in range(out_dim):
    ;;   acc = bias[j]
    ;;   for i in range(in_dim): acc += in[i] * w[j*in_dim + i]
    ;;   out[j] = acc
    ...loop using f32.load/f32.store/f32.mul/f32.add...)

  (func (export "infer") (param $in_ptr i32) (result i32)
    ;; layer 1: fc(in_ptr, hidden_ptr, w1_ptr, b1_ptr, in_dim, hid_dim)
    ;; apply relu to hidden buffer in-place
    ;; layer 2: fc(hidden_ptr, out_ptr, w2_ptr, b2_ptr, hid_dim, out_dim)
    ;; return argmax of output buffer
    ...)
)
```

**Data segment encoding:** use `struct.pack('<' + 'f' * n, *weights_flat)` to get raw f32 LE bytes, then escape them as WAT string literals (e.g. `\xx` hex escapes per byte).

**Memory layout for TinyNN (80→32→95) — fits in 1 WASM page (64KB):**
```
offset     0: input   80 × 4 = 320 bytes
offset   320: hidden  32 × 4 = 128 bytes
offset   448: output  95 × 4 = 380 bytes
offset   828: fc1 W   80 × 32 × 4 = 10240 bytes
offset 11068: fc1 b   32 × 4 = 128 bytes
offset 11196: fc2 W   32 × 95 × 4 = 12160 bytes
offset 23356: fc2 b   95 × 4 = 380 bytes
```

**Gate:** `wat2wasm` assembles the output without error on `tiny_nn.onnx`.

**After M1:** write `tests/test_wasm_infer.py` (M2) — 100 random inputs, argmax ≥ 95% agreement vs `onnxruntime`. Then wire `--target wasm` into the CLI (M3).

## Binary protection demos

- Load self-verifying binary on zxw, tamper via debug API, show the crash
- POK file support — classic Spectrum cheat format, ties into the debug API
- Same demo on WASM — tamper data segment, show detection + bypass
