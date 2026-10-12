# Next: M2 — Z80 CPU core (foundational opcodes)

Build the Z80 CPU struct and a fetch-decode-execute loop covering the opcodes needed for basic programs.

## What to do

1. `src/cpu.rs` — `Cpu` struct: AF/BC/DE/HL/IX/IY/SP/PC + alternates + I/R/IFF1/IFF2/IM/halted. Flag register helpers (S/Z/H/P/N/C).
2. `src/memory.rs` — `Memory` struct: 64K flat array, read/write. `load_rom(data)` writes to 0x0000–0x3FFF with read-only guard. `load_ram(addr, data)` writes anywhere in RAM.
3. `src/cpu.rs` — `step(&mut self, mem: &mut Memory) -> u32` (returns T-states):
   - Fetch opcode at PC, decode, execute, advance PC
   - Unprefixed: LD r,r / LD r,n / LD r,(HL) / LD (HL),r / LD A,(nn) / LD (nn),A / ADD/ADC/SUB/SBC/AND/OR/XOR/CP A,r / INC/DEC r / JP/JR/CALL/RET / DJNZ / PUSH/POP / EX AF,AF' / EXX / LDIR / HALT / NOP / EI/DI / IN/OUT stubs
   - CB prefix: BIT/SET/RES b,r / RLC/RRC/RL/RR/SLA/SRA/SRL r
   - ED prefix subset: LD A,I / LD A,R / IM 0/1/2 / LDIR/LDDR/CPIR/CPDR / NEG / RETI / RETN
4. `src/lib.rs` — expose `Cpu` and `Memory` via wasm-bindgen (or keep internal for now; just needs to compile)
5. `tests/cpu_tests.rs` — unit tests: ADD A,n accumulator; DJNZ countdown loop; LDIR block copy; CALL + RET stack round-trip; BIT 3,(HL) flag check

## Gate

`cargo test` — all CPU unit tests pass. A synthetic test program (load regs, arithmetic, halt) produces the expected register state.

## Reference

- Z80 flag register: bit 7=S, 6=Z, 5=X, 4=H, 3=X, 2=P/V, 1=N, 0=C
- T-states per opcode: NOP=4, LD r,r=4, LD r,n=7, LDIR=21(match)/16(no match), DJNZ=13(branch)/8(no branch)
- CB prefix: fetch opcode at PC+1, all 256 encodings follow the r/b encoding table
- ED prefix: not all 256 slots defined — undefined ED xx = NOP (4 T-states)
