# Z80 Core Instructions

## Load and store

- `LD dst, src` — copy src to dst. Works with registers, immediates, and memory.
  - `LD A, 42` — load immediate value 42 into A
  - `LD A, B` — copy B into A
  - `LD A, (HL)` — load byte from memory address in HL into A
  - `LD (HL), A` — store A into memory at address in HL
  - `LD (addr), A` — store A into absolute memory address
  - `LD A, (addr)` — load from absolute memory address into A
  - `LD HL, 0x8000` — load 16-bit immediate into HL

## Arithmetic (8-bit, operates on A)

- `ADD A, r` — A = A + r. Sets carry flag if overflow.
- `ADC A, r` — A = A + r + carry.
- `SUB r` — A = A - r.
- `SBC A, r` — A = A - r - carry.
- `INC r` — r = r + 1 (does not affect carry flag).
- `DEC r` — r = r - 1 (does not affect carry flag).

Where `r` is any 8-bit register (A, B, C, D, E, H, L) or `(HL)` or immediate `n`.

## Arithmetic (16-bit)

- `ADD HL, rr` — HL = HL + rr (BC, DE, HL, SP).
- `INC rr` — rr = rr + 1.
- `DEC rr` — rr = rr - 1.

## Logic (operates on A)

- `AND r` — A = A & r
- `OR r` — A = A | r
- `XOR r` — A = A ^ r
- `CP r` — compare A with r (subtract without storing, just sets flags)

## Rotate and shift

- `RLCA` — rotate A left circular (bit 7 → carry and bit 0)
- `RRCA` — rotate A right circular
- `RLA` — rotate A left through carry
- `RRA` — rotate A right through carry
- `SLA r` — shift left arithmetic (bit 7 → carry, 0 → bit 0)
- `SRA r` — shift right arithmetic (bit 7 preserved, bit 0 → carry)
- `SRL r` — shift right logical (0 → bit 7, bit 0 → carry)

## Jumps

- `JP addr` — unconditional jump to addr
- `JP cc, addr` — conditional: Z (zero), NZ (not zero), C (carry), NC (no carry)
- `JR offset` — relative jump (signed 8-bit offset, -128 to +127)
- `JR cc, offset` — conditional relative jump (Z, NZ, C, NC only)
- `DJNZ offset` — decrement B, jump if B != 0. The standard loop instruction.

## Call and return

- `CALL addr` — push PC onto stack, jump to addr
- `CALL cc, addr` — conditional call
- `RET` — pop PC from stack, return
- `RET cc` — conditional return

## Stack

- `PUSH rr` — push 16-bit register pair onto stack (AF, BC, DE, HL, IX, IY)
- `POP rr` — pop from stack into register pair (AF, BC, DE, HL, IX, IY)

## Control

- `NOP` — do nothing (1 byte, 4 T-states)
- `HALT` — stop CPU until interrupt. Used to end programs in testing.
- `DI` — disable interrupts
- `EI` — enable interrupts
