# Z80 Flags

The F register holds flags set by arithmetic, logic, and comparison instructions.

## Flag bits

| Bit | Name | Symbol | Set when |
|-----|------|--------|----------|
| 7 | Sign | S | Result is negative (bit 7 is 1) |
| 6 | Zero | Z | Result is zero |
| 5 | — | — | Undocumented (copy of bit 5 of result) |
| 4 | Half carry | H | Carry from bit 3 to bit 4 (BCD ops) |
| 3 | — | — | Undocumented (copy of bit 3 of result) |
| 2 | Parity/Overflow | P/V | Even parity (logic ops) or overflow (arithmetic ops) |
| 1 | Subtract | N | Last op was subtract (BCD ops) |
| 0 | Carry | C | Unsigned overflow: result doesn't fit in 8 bits |

## Condition codes for jumps and calls

- `Z` — zero flag is set (result was 0)
- `NZ` — zero flag is clear (result was not 0)
- `C` — carry flag is set (unsigned overflow)
- `NC` — carry flag is clear (no unsigned overflow)
- `PE` — parity even / overflow set
- `PO` — parity odd / no overflow
- `M` — sign flag set (negative)
- `P` — sign flag clear (positive)

## Common patterns

### Test if A is zero
```
OR A        ; sets Z flag if A == 0 (also clears carry)
JR Z, label ; jump if A was zero
```

### Compare A to a value
```
CP 10       ; compare A to 10 (subtract without storing)
JR Z, equal ; jump if A == 10
JR C, less  ; jump if A < 10 (unsigned)
```

### Check carry after addition
```
ADD A, B
JR C, overflow  ; jump if result > 255
```
