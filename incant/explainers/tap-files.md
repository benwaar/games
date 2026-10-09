# TAP Files — Spectrum Tape Format, Audio Loading, and Disassembly

How ZX Spectrum programs were stored on cassette tape, how TAP files represent that data digitally, and how to go from incant's Z80 binary output all the way to a physical tape loadable on real hardware.

## How the Spectrum loads from tape

The Spectrum has no disk drive. Programs live on audio cassettes. The hardware is brutally simple:

1. Cassette player connects to the **EAR port** (3.5mm jack)
2. The Spectrum reads **voltage edges** — high/low transitions in the audio signal
3. A **pilot tone** (steady alternating pulses) says "data is coming"
4. Then **data bits** — short pulse = 0, long pulse = 1
5. The ROM loader reads bytes, verifies a checksum, and copies into RAM

That screeching sound IS the program. Every screech is bytes encoded as audio pulses.

## What BASIC does

```basic
LOAD "" CODE        ' load next code block from tape into RAM at the address in the header
RANDOMIZE USR 32768 ' jump to that address and run the machine code
```

`LOAD "" CODE` tells the ROM to:
1. Read the next **header block** from tape (filename, type, start address, length)
2. Read the next **data block** (the actual bytes)
3. Copy the data into RAM at the start address from the header

## TAP file format

TAP is the digital shortcut — it stores the raw bytes without any audio encoding. No pilot tones, no pulse timings, just the data the Spectrum would have read.

A TAP file is a sequence of **blocks**:

```
[2 bytes: block length (little-endian)]
[1 byte:  flag — 0x00 = header, 0xFF = data]
[N bytes: payload]
[1 byte:  checksum — XOR of flag + all payload bytes]
```

A standard program save produces two blocks: header then data.

### Header block (17 bytes payload)

```
Offset  Len  What
0       1    Type: 0=Program, 1=NumArray, 2=CharArray, 3=Code
1       10   Filename (space-padded)
11      2    Data block length (little-endian)
13      2    Param 1 — for Code: start address (e.g. 0x8000 = 32768)
15      2    Param 2 — for Code: 32768 (conventional)
```

### Data block

Just the raw bytes, wrapped with a 0xFF flag and a checksum. For machine code, this is the assembled binary verbatim.

### Example: wrapping a 2-byte program

The program `add a,b; halt` assembles to two bytes: `0x80 0x76`.

```
Header block:
  Length:    19 00          (19 bytes follow)
  Flag:     00             (header)
  Type:     03             (Code)
  Filename: 41 44 44 20 20 20 20 20 20 20  ("ADD       ")
  DataLen:  02 00          (2 bytes of code)
  Start:    00 80          (32768 = 0x8000)
  Param2:   00 80          (32768)
  Checksum: XOR of all 18 bytes above

Data block:
  Length:   04 00          (4 bytes follow)
  Flag:     FF             (data)
  Payload:  80 76          (add a,b; halt)
  Checksum: XOR of FF 80 76
```

TAP files can be concatenated — `cat a.tap b.tap > combined.tap` — because each block is self-describing.

## The tape format family

| Format | What it stores | Use case |
|--------|---------------|----------|
| **TAP** | Raw bytes only — no audio, no timing | Simple, every emulator reads it |
| **TZX** | Pulse timings — faithfully represents the tape signal | Copy protection, custom loaders |
| **WAV** | Literal audio recording | Physical tape transfer |
| **CSW** | Compressed audio waveform | Archival |

TAP is the simplest and most portable. TZX matters for games with custom loaders or copy protection schemes that TAP can't represent.

## From TAP to real hardware

The full chain to load AI-generated code on a real Spectrum:

```
sigil.yaml → LLM → Z80 asm → assemble → .bin → wrap → .tap → audio → tape/phone → Spectrum
```

### Step 1: TAP to WAV

```bash
brew install fuse-utils
tape2wav game.tap game.wav
```

This generates audio with correct pilot tones, pulse timings, and gaps — exactly what the Spectrum ROM expects to hear.

### Step 2: WAV to Spectrum

**Option A — Phone as tape player (easiest):**
1. Transfer the WAV to your phone
2. 3.5mm cable from phone headphone jack → Spectrum EAR port
3. On the Spectrum: `LOAD ""`
4. Press play on the phone

**Option B — Record to cassette:**
1. 3.5mm cable from Mac/phone → cassette recorder line-in
2. Record the WAV onto a blank tape
3. Play the tape into the Spectrum's EAR port

**Option C — Direct from laptop:**
1. Play the WAV through a 3.5mm cable straight into EAR
2. Works fine, just need the right volume level

### Tips

- **Volume matters.** Too quiet = the Spectrum can't detect edges. Too loud = clipping. Start at ~75%.
- The Spectrum +2/+3 (with built-in datacorders) are more forgiving than the original rubber-key models.
- Android apps like **PlayZX** play TZX/TAP files as audio directly — no WAV conversion needed.
- At retro meetups, people routinely load games from phones. The Spectrum doesn't care where the signal comes from.

## Disassembly — TAP back to assembly

Going the other direction: extract machine code from existing TAP files and disassemble it.

### Tools

| Tool | Type | What it does | Install |
|------|------|-------------|---------|
| **skoolkit** | Python (pip) | Full disassembly suite — TAP → snapshot → annotated ASM with labels | `pip install skoolkit` |
| **z80dis** | Python (pip) | Pure Python Z80 disassembler — works on raw bytes | `pip install z80dis` |
| **z80dasm** | CLI (brew) | Disassembler for raw binaries | `brew install z80dasm` |
| **pasmo** | CLI | Z80 assembler that can output TAP directly | `pasmo --tap source.asm output.tap` |

### skoolkit pipeline (best for full games)

```bash
pip install skoolkit
tap2sna.py game.tap game.z80          # TAP to Z80 snapshot
sna2skool.py game.z80 > game.skool    # snapshot to annotated disassembly
skool2asm.py game.skool > game.asm    # to standard Z80 assembly
```

### z80dis (quick, pure Python)

```python
from z80dis import z80

data = open("code.bin", "rb").read()
pc = 0x8000
while pc < 0x8000 + len(data):
    inst, ln = z80.decode(data, pc - 0x8000)
    print(f"{pc:04X}  {inst}")
    pc += ln
```

### Extraction from TAP (Python)

```python
def extract_code_blocks(tap_path):
    """Parse TAP file, yield (start_address, bytes) for each code block."""
    data = open(tap_path, "rb").read()
    pos = 0
    pending_header = None

    while pos < len(data):
        length = int.from_bytes(data[pos:pos+2], "little")
        pos += 2
        block = data[pos:pos+length]
        pos += length

        flag = block[0]
        payload = block[1:-1]

        if flag == 0x00 and len(payload) == 17 and payload[0] == 3:
            start_addr = int.from_bytes(payload[13:15], "little")
            pending_header = start_addr
        elif flag == 0xFF and pending_header is not None:
            yield pending_header, payload
            pending_header = None
```

## Relevance to incant

TAP files close the loop in two directions:

**Output:** incant's Z80 gate already produces `.bin` files. Wrapping to TAP is ~30 lines of Python — write a header block + data block with checksums. This makes incant output directly loadable in emulators or on real hardware.

**Input (knowledge):** Disassembling real Spectrum games produces Z80 assembly patterns — actual idiomatic code written by human programmers under extreme constraints. Feeding this into RAG gives the LLM better grounding than instruction set docs alone.
