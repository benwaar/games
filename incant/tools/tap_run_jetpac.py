#!/usr/bin/env python3
"""Run the Jetpac loader in a Z80 emulator.

Loads all blocks from the original TAP at their correct addresses,
simulates the multi-stage loader, and traces execution to find
why the game doesn't start.

Three modes:
  1. original  — load all blocks including sysvar patches, run stub
  2. relocated — load pre-relocated code (as tap_rebuild_jetpac does), jump to 0x61E5
  3. probe     — try multiple entry points and report which one gets furthest
"""

import sys
from pathlib import Path

import z80 as z80lib


def extract_blocks(tap_path: str) -> list[dict]:
    """Parse TAP into list of {flag, payload, header} dicts."""
    data = Path(tap_path).read_bytes()
    blocks = []
    pos = 0
    pending_header = None
    while pos < len(data):
        length = int.from_bytes(data[pos:pos+2], "little")
        pos += 2
        block_data = data[pos:pos+length]
        pos += length
        flag = block_data[0]
        payload = block_data[1:-1]

        entry = {"flag": flag, "payload": payload}

        if flag == 0x00 and len(payload) == 17:
            entry["header"] = {
                "type": payload[0],
                "name": payload[1:11].decode("ascii", errors="replace").rstrip(),
                "data_len": int.from_bytes(payload[11:13], "little"),
                "load_addr": int.from_bytes(payload[13:15], "little"),
            }
            pending_header = entry["header"]
        elif flag == 0xFF and pending_header:
            entry["header"] = pending_header
            pending_header = None

        blocks.append(entry)
    return blocks


def dump_region(machine: z80lib.Z80Machine, addr: int, length: int, label: str) -> None:
    """Print a hex dump of a memory region."""
    data = bytes(machine.memory[addr + i] for i in range(length))
    print(f"\n  {label} (0x{addr:04X}–0x{addr + length - 1:04X}):")
    for row in range(0, length, 16):
        hex_part = " ".join(f"{data[row+i]:02X}" for i in range(min(16, length - row)))
        ascii_part = "".join(
            chr(data[row+i]) if 32 <= data[row+i] < 127 else "."
            for i in range(min(16, length - row))
        )
        print(f"    0x{addr + row:04X}: {hex_part:<48s} {ascii_part}")


def run_original(tap_path: str) -> None:
    """Load all blocks at original addresses, run the relocation stub."""
    blocks = extract_blocks(tap_path)
    machine = z80lib.Z80Machine()
    machine.ticks_to_stop = 1_000_000

    print("=== Original loader simulation ===\n")

    # Load each data block at its declared address
    # blocks[1] = BASIC (skip), blocks[3] = screen, blocks[5] = game,
    # blocks[7] = stub, blocks[9] = JP(HL), blocks[11] = ERR_SP
    data_blocks = [(i, b) for i, b in enumerate(blocks) if b["flag"] == 0xFF and "header" in b]

    for idx, block in data_blocks:
        hdr = block["header"]
        addr = hdr["load_addr"]
        data = block["payload"]
        name = hdr.get("name", "?")
        print(f"  Block {idx}: \"{name}\" → 0x{addr:04X} ({len(data)} bytes)")
        machine.set_memory_block(addr, data)

    # Show what's at key addresses
    print("\n--- Memory state after loading ---")
    dump_region(machine, 0x5B80, 15, "Relocation stub (Jetpac4)")
    dump_region(machine, 0x5C78, 2, "ERR_SP patch (Jetpac6)")
    dump_region(machine, 0x5CB0, 1, "JP (HL) byte (Jetpac5)")
    dump_region(machine, 0x6000, 32, "Game code first 32 bytes (before relocation)")

    # Run the relocation stub at 0x5B80
    print("\n--- Running relocation stub at 0x5B80 ---")
    machine.pc = 0x5B80
    machine.sp = 0x5FFF

    # Set a breakpoint at the JP target to catch the jump
    machine.set_breakpoint(0x61E5)
    machine.run()

    if machine.pc == 0x61E5:
        print(f"  ✓ Reached 0x61E5 (stub completed relocation and jumped)")
    else:
        print(f"  PC stopped at 0x{machine.pc:04X} (halted={machine.halted})")

    dump_region(machine, 0x6000, 32, "Game code first 32 bytes (after relocation)")
    dump_region(machine, 0x61E5, 32, "Entry point at 0x61E5")

    # Now let it run from the entry point
    print("\n--- Running from 0x61E5 ---")
    machine.clear_breakpoint(0x61E5)
    machine.ticks_to_stop = 5_000_000
    machine.run()

    print(f"  PC stopped at 0x{machine.pc:04X}")
    print(f"  Halted: {machine.halted}")
    print(f"  Registers: A=0x{machine.a:02X} BC=0x{machine.bc:04X} DE=0x{machine.de:04X} HL=0x{machine.hl:04X} SP=0x{machine.sp:04X}")

    dump_region(machine, machine.pc, 16, f"Code at PC (0x{machine.pc:04X})")


def run_relocated(tap_path: str) -> None:
    """Load pre-relocated code (as the rebuild does) and jump to entry."""
    blocks = extract_blocks(tap_path)
    machine = z80lib.Z80Machine()
    machine.ticks_to_stop = 1_000_000

    print("=== Relocated (rebuild) simulation ===\n")

    # Get game code and apply the LDIR relocation in Python
    game_code = bytearray(blocks[5]["payload"])  # 8192 bytes at 0x6000

    relocated = bytearray(len(game_code))
    for i in range(len(game_code)):
        src_offset = i + 4
        if src_offset < len(game_code):
            relocated[i] = game_code[src_offset]
        else:
            relocated[i] = relocated[i - 4]

    # Load screen + relocated code
    screen_data = blocks[3]["payload"]
    machine.set_memory_block(0x4000, screen_data)
    machine.set_memory_block(0x6000, bytes(relocated))
    print(f"  Loaded screen at 0x4000 ({len(screen_data)} bytes)")
    print(f"  Loaded relocated code at 0x6000 ({len(relocated)} bytes)")

    dump_region(machine, 0x61E5, 32, "Entry point at 0x61E5")

    # Try without sysvar patches first
    print("\n--- Running from 0x61E5 (NO sysvar patches) ---")
    machine.pc = 0x61E5
    machine.sp = 0x5FFF
    machine.ticks_to_stop = 5_000_000
    machine.run()

    print(f"  PC stopped at 0x{machine.pc:04X}")
    print(f"  Halted: {machine.halted}")
    print(f"  Registers: A=0x{machine.a:02X} BC=0x{machine.bc:04X} DE=0x{machine.de:04X} HL=0x{machine.hl:04X} SP=0x{machine.sp:04X}")
    dump_region(machine, machine.pc, 16, f"Code at PC (0x{machine.pc:04X})")

    # Now try WITH sysvar patches
    print("\n--- Running from 0x61E5 (WITH sysvar patches) ---")
    machine2 = z80lib.Z80Machine()
    machine2.ticks_to_stop = 5_000_000
    machine2.set_memory_block(0x4000, screen_data)
    machine2.set_memory_block(0x6000, bytes(relocated))
    # Apply the sysvar patches
    machine2.set_memory_block(0x5C78, bytes([0x83, 0x5A]))  # ERR_SP = 0x5A83
    machine2.set_memory_block(0x5CB0, bytes([0xE9]))  # JP (HL) = 0xE9
    machine2.pc = 0x61E5
    machine2.sp = 0x5FFF
    machine2.run()

    print(f"  PC stopped at 0x{machine2.pc:04X}")
    print(f"  Halted: {machine2.halted}")
    print(f"  Registers: A=0x{machine2.a:02X} BC=0x{machine2.bc:04X} DE=0x{machine2.de:04X} HL=0x{machine2.hl:04X} SP=0x{machine2.sp:04X}")
    dump_region(machine2, machine2.pc, 16, f"Code at PC (0x{machine2.pc:04X})")


def run_probe(tap_path: str) -> None:
    """Try multiple entry points and report which gets furthest."""
    blocks = extract_blocks(tap_path)

    # Apply relocation
    game_code = bytearray(blocks[5]["payload"])
    relocated = bytearray(len(game_code))
    for i in range(len(game_code)):
        src_offset = i + 4
        if src_offset < len(game_code):
            relocated[i] = game_code[src_offset]
        else:
            relocated[i] = relocated[i - 4]

    screen_data = blocks[3]["payload"]

    entry_points = [
        (0x61E5, "JP target from stub"),
        (0x6000, "Start of game code"),
        (0x6004, "Original code start (pre-relocation offset)"),
    ]

    print("=== Entry point probe ===\n")

    for addr, label in entry_points:
        for with_patches in [False, True]:
            machine = z80lib.Z80Machine()
            machine.ticks_to_stop = 5_000_000
            machine.set_memory_block(0x4000, screen_data)
            machine.set_memory_block(0x6000, bytes(relocated))
            if with_patches:
                machine.set_memory_block(0x5C78, bytes([0x83, 0x5A]))
                machine.set_memory_block(0x5CB0, bytes([0xE9]))
            machine.pc = addr
            machine.sp = 0x5FFF
            machine.run()

            patches_label = "with patches" if with_patches else "no patches"
            status = "HALT" if machine.halted else f"stuck at 0x{machine.pc:04X}"
            print(f"  0x{addr:04X} ({label}, {patches_label}): {status}  A=0x{machine.a:02X} HL=0x{machine.hl:04X}")


if __name__ == "__main__":
    tap_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/tap/jetpac.tap"
    mode = sys.argv[1] if len(sys.argv) > 1 else "original"

    if not Path(tap_path).exists():
        print(f"TAP file not found: {tap_path}")
        sys.exit(1)

    if mode == "original":
        run_original(tap_path)
    elif mode == "relocated":
        run_relocated(tap_path)
    elif mode == "probe":
        run_probe(tap_path)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: tap_run_jetpac.py [original|relocated|probe] [path/to/jetpac.tap]")
        sys.exit(1)
