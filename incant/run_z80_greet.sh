#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ASM="output/greet.asm"
BIN="output/greet.bin"

if [ ! -f "$BIN" ]; then
    echo "No greet.bin found. Generating..."
    source .venv/bin/activate
    python -m incant cast sigils/examples/z80_greet.sigil.yaml -v
fi

if [ -n "${1:-}" ]; then
    NAME="$1"
else
    echo "Type a name, press Enter:"
    read -r NAME
fi

source .venv/bin/activate
python3 -c "
import z80, sys
machine = z80.Z80Machine()
with open('output/greet.bin', 'rb') as f:
    data = f.read()
machine.set_memory_block(0, data)
machine.ticks_to_stop = 100000
name = sys.argv[1].encode('ascii') + b'\x00'
machine.set_memory_block(0x8000, name)
machine.run()
out = []
for i in range(256):
    b = machine.memory[0x9000 + i]
    if b == 0:
        break
    out.append(b)
print(bytes(out).decode('ascii'))
" "$NAME"
