# WAT Common Patterns

## Exported function (the basic pattern)

Every incant-generated WAT module follows this structure:

```wat
(module
  (func $name (param $a i32) (param $b i32) (result i32)
    ;; generated body
  )
  (export "name" (func $name))
)
```

The export name must match the sigil's `export` field.

## Add two numbers

```wat
(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  (export "add" (func $add))
)
```

## Factorial (iterative)

```wat
(module
  (func $factorial (param $n i32) (result i32)
    (local $result i32)
    i32.const 1
    local.set $result

    (block $break
      (loop $loop
        ;; if n <= 1, break
        local.get $n
        i32.const 1
        i32.le_s
        br_if $break

        ;; result = result * n
        local.get $result
        local.get $n
        i32.mul
        local.set $result

        ;; n = n - 1
        local.get $n
        i32.const 1
        i32.sub
        local.set $n

        br $loop
      )
    )

    local.get $result
  )
  (export "factorial" (func $factorial))
)
```

## Memory read/write

```wat
(module
  (memory 1)
  (export "memory" (memory 0))

  (func $store_and_load (param $val i32) (result i32)
    ;; store at offset 0
    i32.const 0
    local.get $val
    i32.store

    ;; load back
    i32.const 0
    i32.load
  )
  (export "store_and_load" (func $store_and_load))
)
```

If a sigil has `memory: 1`, include `(memory 1)` and export it.

## Testing with wasm-interp

After `wat2wasm` produces a .wasm file:

```bash
# Run all exports and print results
wasm-interp module.wasm --run-all-exports

# Run a specific export with arguments
wasm-interp module.wasm -r add -a i32:2 -a i32:3
```

`wasm-interp` prints the return value of each exported function. The gate parses this output.

## Program structure for incant

Every generated WAT module must:

1. Start with `(module`
2. Declare the function with typed params and result
3. Export the function with the name from the sigil's `export` field
4. If the sigil has `memory: N`, include `(memory N)` and export it
5. End with `)`
