# Z80 Registers

The Z80 has 8-bit and 16-bit registers.

## 8-bit registers

- **A** — accumulator. Most arithmetic and logic operations use A as one operand and store the result in A.
- **B, C, D, E, H, L** — general-purpose. Often used in pairs (BC, DE, HL) as 16-bit values.
- **F** — flags register. Not directly addressable. Set by arithmetic/logic ops.

## 16-bit register pairs

- **BC** — B (high byte) + C (low byte). Often used as a counter.
- **DE** — D (high) + E (low). Often used as a destination pointer.
- **HL** — H (high) + L (low). The primary memory pointer. `(HL)` means "the byte at the address in HL".
- **SP** — stack pointer. Points to the top of the stack. Grows downward.
- **PC** — program counter. Address of the next instruction.

## Shadow registers

A', F', B', C', D', E', H', L' — a second set, swapped in with `EX AF,AF'` or `EXX`. Used for fast context switches (interrupt handlers).

## Index registers

- **IX, IY** — 16-bit. Used for indexed addressing: `(IX+d)` accesses memory at IX plus a signed offset d (-128 to +127).

## Special registers

- **I** — interrupt vector (upper 8 bits of interrupt handler address in mode 2).
- **R** — refresh counter. Auto-incremented. Sometimes used as a pseudo-random source.
