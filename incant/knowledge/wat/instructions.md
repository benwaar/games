# WAT Instructions

## Constants

- `i32.const 42` — push i32 value 42
- `i64.const 100` — push i64 value
- `f32.const 3.14` — push f32 value
- `f64.const 2.718` — push f64 value

## Arithmetic (i32)

- `i32.add` — a + b
- `i32.sub` — a - b
- `i32.mul` — a * b
- `i32.div_s` — signed division
- `i32.div_u` — unsigned division
- `i32.rem_s` — signed remainder
- `i32.rem_u` — unsigned remainder

## Bitwise (i32)

- `i32.and` — bitwise AND
- `i32.or` — bitwise OR
- `i32.xor` — bitwise XOR
- `i32.shl` — shift left
- `i32.shr_s` — shift right (signed / arithmetic)
- `i32.shr_u` — shift right (unsigned / logical)
- `i32.rotl` — rotate left
- `i32.rotr` — rotate right
- `i32.clz` — count leading zeros
- `i32.ctz` — count trailing zeros
- `i32.popcnt` — count set bits

## Comparison (i32, push 1 or 0)

- `i32.eq` — equal
- `i32.ne` — not equal
- `i32.lt_s` / `i32.lt_u` — less than (signed/unsigned)
- `i32.gt_s` / `i32.gt_u` — greater than
- `i32.le_s` / `i32.le_u` — less or equal
- `i32.ge_s` / `i32.ge_u` — greater or equal
- `i32.eqz` — equal to zero (unary)

## Variables

- `local.get $name` — push local onto stack
- `local.set $name` — pop stack into local
- `local.tee $name` — set local AND keep value on stack
- `global.get $name` — push global
- `global.set $name` — pop into global (must be mutable)

## Control flow

### Block and branch
```wat
(block $label
  ;; code
  br $label        ;; break out of block
)
```

### Loop
```wat
(loop $label
  ;; code
  br $label        ;; jump back to top of loop
)
```

### If/else

Folded (S-expression) form:
```wat
(if (i32.eqz (local.get $n))
  (then
    i32.const 1
  )
  (else
    ;; recursive case
  )
)
```

Flat (stack) form uses `if`/`else`/`end` without `then`:
```wat
local.get $n
i32.eqz
if (result i32)
  i32.const 1
else
  ;; recursive case
end
```

### Branch table
```wat
br_table $case0 $case1 $case2 $default
```

### Return
```wat
return           ;; early return (value must be on stack if function has result)
```

## Function calls

- `call $func_name` — call function, args taken from stack
- `call_indirect` — indirect call via table (for function pointers)
