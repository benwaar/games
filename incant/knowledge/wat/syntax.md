# WAT Syntax

WAT (WebAssembly Text) uses S-expressions — parenthesised prefix notation.

## Module structure

Every WAT file is a module:

```wat
(module
  ;; functions, memory, exports go here
)
```

## Functions

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

- `$add` is the internal name (optional, for readability)
- `(export "add" (func $add))` makes it callable from outside

## Stack machine

WASM is a stack machine. Instructions push/pop values:

```wat
local.get $a    ;; push $a onto stack
local.get $b    ;; push $b onto stack
i32.add         ;; pop two, push sum
```

The last value on the stack is the return value.

## Locals

```wat
(func $example (result i32)
  (local $x i32)
  (local $y i32)
  i32.const 10
  local.set $x
  local.get $x
)
```

Locals are initialised to 0 (i32/i64) or 0.0 (f32/f64).

## Comments

```wat
;; line comment
(; block comment ;)
```

## Folded (nested) S-expression syntax

Instructions can be written flat (stack-style) or folded (nested):

```wat
;; flat (stack) style
local.get $a
local.get $b
i32.add

;; folded style (equivalent)
(i32.add (local.get $a) (local.get $b))
```

Both produce the same bytecode. Flat is clearer for complex control flow.
