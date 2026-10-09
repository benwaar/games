# Z80 Common Patterns

## Loop N times (DJNZ)

```asm
    ld b, 10       ; loop counter
loop:
    ; ... body ...
    djnz loop      ; decrement B, jump if not zero
```

`DJNZ` is the idiomatic Z80 loop. B is the counter (max 256 iterations, 0 means 256).

## Loop with counter in A

When you need A as the counter (DJNZ uses B):

```asm
    ld a, 0        ; result accumulator
    ld b, 5        ; loop count
loop:
    inc a          ; body: increment A
    djnz loop
    halt
```

## Memory store and load

```asm
    ld a, 42
    ld (0x8000), a   ; store A at address 0x8000
    ld a, 0          ; clear A
    ld a, (0x8000)   ; load back from 0x8000
    halt             ; A should be 42
```

## Add two registers

```asm
    ; A and B are pre-loaded by test harness
    add a, b       ; A = A + B
    halt
```

## Conditional branch

```asm
    cp 10          ; compare A to 10
    jr z, is_ten   ; jump if A == 10
    jr c, less     ; jump if A < 10
    ; A > 10
    jr done
is_ten:
    ; A == 10
    jr done
less:
    ; A < 10
done:
    halt
```

## Subroutine call

```asm
    ld sp, 0xFFFE  ; init stack
    call my_func
    halt

my_func:
    ; ... body ...
    ret
```

## 16-bit addition

```asm
    ld hl, 1000
    ld de, 2000
    add hl, de     ; HL = 3000
```

## Program structure for incant

Every generated Z80 program should follow this structure:

```asm
    ; registers pre-loaded by test harness
    ; ... generated code ...
    halt           ; must end with HALT for test harness
```

The test harness:
1. Sets register values from sigil test inputs
2. Loads assembled code at address 0x0000
3. Runs until HALT
4. Reads register values and compares to sigil expected outputs
