#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ASM="output/greet.asm"
BIN="output/greet.bin"

if [ ! -f "$ASM" ]; then
    echo "No greet.asm found. Generating..."
    source .venv/bin/activate
    python -m incant cast sigils/examples/z80_greet.sigil.yaml -v
fi

NAME="${1:-Ben}"

echo "=== Z80 greet ==="
echo "Input:  \"$NAME\""

# Run in the Z80 emulator via Python
source .venv/bin/activate
OUTPUT=$(python3 -c "
import z80
machine = z80.Z80Machine()
with open('$BIN', 'rb') as f:
    data = f.read()
machine.set_memory_block(0, data)
machine.ticks_to_stop = 100000
# Load input string
name = '$NAME'.encode('ascii') + b'\x00'
machine.set_memory_block(0x8000, name)
machine.run()
# Read output
out = []
for i in range(256):
    b = machine.memory[0x9000 + i]
    if b == 0:
        break
    out.append(b)
print(bytes(out).decode('ascii'))
")

echo "Output: \"$OUTPUT\""
echo ""
echo "Assembly ($ASM):"
cat "$ASM"
echo ""
echo "Binary: $BIN ($(wc -c < "$BIN" | tr -d ' ') bytes)"
