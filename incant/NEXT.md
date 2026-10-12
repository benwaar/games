# Next: M9 — Binary Protection PoC: Self-Verifying Code

M8 complete — TAP writer, reader, disassembler, and RAG knowledge pipeline in place. 90 tests pass.

**What was built:**

`incant/tap.py` exports:

```python
from incant.tap import write_tap, read_tap, extract_code_blocks, disassemble

# Write a TAP file from assembled Z80 bytes
write_tap(binary_data, "myprog", start_addr=0x8000, output_path="output/myprog.tap")

# Parse a TAP file into blocks
blocks = read_tap("game.tap")  # [{flag, payload, valid, type, name, param1, ...}]

# Extract typed Code blocks
code_blocks = extract_code_blocks("game.tap")  # [{name, start_addr, data}]

# Disassemble Z80 bytes to annotated assembly (uses z80dis)
asm = disassemble(code_bytes, start_addr=0x8000)
```

New CLI subcommands:

```bash
# Cast Z80 sigil and also produce a .tap
python -m incant cast sigils/examples/z80_add.sigil.yaml --tap

# Disassemble all Code blocks from a TAP
python -m incant disasm game.tap

# Disassemble and embed into RAG z80 collection
python -m incant disasm game.tap --embed
```

**What's next:**

M9 builds self-verifying binaries — code that checks its own integrity at runtime and refuses to run if tampered with. Two targets: Z80 (XOR checksum over own code region) and WASM (data-segment mirror + hash check).

**Steps:**
1. Z80 sigil: `self_verify.sigil.yaml` — XOR self-check over code region
2. Post-assembly checksum patching in the Z80 gate
3. Test: modify one byte, verify check catches it
4. WAT sigil: `self_verify_wasm.sigil.yaml` — data-segment mirror + hash
5. Test: modify data segment, verify check catches it
6. Explainer update: add worked examples to `explainers/binary-protection.md`
7. Document attacker bypass for each
