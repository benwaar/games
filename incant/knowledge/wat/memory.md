# WAT Memory

WASM memory is a contiguous, resizable array of bytes.

## Declaring memory

```wat
(module
  (memory 1)           ;; 1 page = 64KB
  (export "memory" (memory 0))
)
```

Each page is 65536 bytes (64KB). `(memory 1)` means 1 initial page. `(memory 1 4)` means 1 initial, 4 max.

## Load and store

All memory operations take a byte offset from the stack.

### i32
- `i32.load offset=0` — load 4 bytes as i32
- `i32.store offset=0` — store i32 as 4 bytes
- `i32.load8_s` / `i32.load8_u` — load 1 byte, sign/zero extend to i32
- `i32.load16_s` / `i32.load16_u` — load 2 bytes
- `i32.store8` — store low byte of i32
- `i32.store16` — store low 2 bytes

### Alignment
The `offset` and `align` are static (compile-time). The dynamic address comes from the stack.

```wat
;; store value 42 at address 0
i32.const 0        ;; address
i32.const 42       ;; value
i32.store          ;; mem[0..3] = 42
```

```wat
;; load from address 0
i32.const 0        ;; address
i32.load           ;; push mem[0..3]
```

## Memory is little-endian

Bytes are stored least-significant first (same as x86, same as Z80).

## Common pattern: swap two values in memory

```wat
(func $swap (param $a i32) (param $b i32) (result i32)
  ;; store $a at offset 0
  i32.const 0
  local.get $a
  i32.store

  ;; store $b at offset 4
  i32.const 4
  local.get $b
  i32.store

  ;; swap: load offset 4, store at 0; load offset 0, store at 4
  ;; ... (use a local as temp)

  ;; return value at offset 4 (originally $a)
  i32.const 4
  i32.load
)
```

## Memory size

- `memory.size` — current size in pages
- `memory.grow` — grow by N pages, returns previous size (-1 on failure)
