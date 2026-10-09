# WAT Types

WebAssembly has four value types.

## Value types

| Type | Size | Description |
|------|------|-------------|
| `i32` | 32-bit | Integer (signed or unsigned, interpretation depends on the instruction) |
| `i64` | 64-bit | Integer |
| `f32` | 32-bit | IEEE 754 float |
| `f64` | 64-bit | IEEE 754 double |

There are no smaller types (no i8, i16). Byte-level operations use i32 with masking.

## Function signatures

```wat
(func $add (param $a i32) (param $b i32) (result i32)
  local.get $a
  local.get $b
  i32.add
)
```

- `(param $name type)` — named parameter
- `(result type)` — return type (at most one in MVP WASM)
- `(local $name type)` — local variable (initialised to 0)

## Type coercion

No implicit conversion. Use explicit instructions:
- `i32.wrap_i64` — i64 → i32 (truncate)
- `i64.extend_i32_s` — i32 → i64 (sign-extend)
- `i64.extend_i32_u` — i32 → i64 (zero-extend)
- `f32.convert_i32_s` — signed i32 → f32
- `i32.trunc_f32_s` — f32 → signed i32 (truncate toward zero)
