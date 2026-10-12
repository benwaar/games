use zxw::cpu::{Cpu, FLAG_C, FLAG_Z, FLAG_PV};
use zxw::io::NullIo;
use zxw::memory::Memory;

fn run_program(prog: &[u8]) -> (Cpu, Memory) {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    mem.load_ram(0x0000, prog);
    for _ in 0..10_000 {
        if cpu.halted { break; }
        cpu.step(&mut mem, &mut NullIo);
    }
    (cpu, mem)
}

/// Run up to max_steps or until halted
fn run_n(prog: &[u8], max_steps: usize) -> (Cpu, Memory) {
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    mem.load_ram(0x0000, prog);
    for _ in 0..max_steps {
        if cpu.halted { break; }
        cpu.step(&mut mem, &mut NullIo);
    }
    (cpu, mem)
}

// ── ADD A,n accumulator test ───────────────────────────────────────────────

#[test]
fn test_add_a_n() {
    // LD A, 0x10 ; ADD A, 0x20 ; HALT
    let prog = &[0x3E, 0x10, 0xC6, 0x20, 0x76];
    let (cpu, _) = run_program(prog);
    assert_eq!(cpu.a, 0x30, "A should be 0x30");
    assert!(!cpu.flag(FLAG_Z), "Z flag should be clear");
    assert!(!cpu.flag(FLAG_C), "C flag should be clear");
}

#[test]
fn test_add_overflow_sets_carry() {
    // LD A, 0xFF ; ADD A, 0x01 ; HALT
    let prog = &[0x3E, 0xFF, 0xC6, 0x01, 0x76];
    let (cpu, _) = run_program(prog);
    assert_eq!(cpu.a, 0x00, "A should wrap to 0x00");
    assert!(cpu.flag(FLAG_Z), "Z flag should be set");
    assert!(cpu.flag(FLAG_C), "C flag should be set");
}

// ── LD r,r register-to-register transfers ─────────────────────────────────

#[test]
fn test_ld_rr() {
    // LD A, 0x42 ; LD B, A ; LD C, B ; HALT
    // LD A,n = 3E 42  LD B,A = 47  LD C,B = 48  HALT = 76
    let prog = &[0x3E, 0x42, 0x47, 0x48, 0x76];
    let (cpu, _) = run_program(prog);
    assert_eq!(cpu.a, 0x42);
    assert_eq!(cpu.b, 0x42);
    assert_eq!(cpu.c, 0x42);
}

// ── DJNZ countdown loop ────────────────────────────────────────────────────

#[test]
fn test_djnz_loop() {
    // LD B, 3
    // loop: DJNZ loop  (offset = -2 = 0xFE)
    // HALT
    let prog = &[
        0x06, 0x03, // LD B, 3
        0x10, 0xFE, // DJNZ -2  (back to self)
        0x76,       // HALT
    ];
    let (cpu, _) = run_program(prog);
    assert_eq!(cpu.b, 0, "B should be 0 after loop");
}

// ── LDIR block copy ────────────────────────────────────────────────────────

#[test]
fn test_ldir_block_copy() {
    // Copy 4 bytes from 0x1000 to 0x2000
    // LD HL, 0x1000  ; LD DE, 0x2000  ; LD BC, 4  ; ED B0 (LDIR)  ; HALT
    let prog = &[
        0x21, 0x00, 0x10, // LD HL, 0x1000
        0x11, 0x00, 0x20, // LD DE, 0x2000
        0x01, 0x04, 0x00, // LD BC, 4
        0xED, 0xB0,       // LDIR
        0x76,             // HALT
    ];
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    mem.load_ram(0x0000, prog);
    // Put source data at 0x1000
    mem.load_ram(0x1000, &[0xAA, 0xBB, 0xCC, 0xDD]);

    for _ in 0..100 {
        if cpu.halted { break; }
        cpu.step(&mut mem, &mut NullIo);
    }

    assert_eq!(mem.read_byte(0x2000), 0xAA);
    assert_eq!(mem.read_byte(0x2001), 0xBB);
    assert_eq!(mem.read_byte(0x2002), 0xCC);
    assert_eq!(mem.read_byte(0x2003), 0xDD);
    assert_eq!(cpu.bc(), 0, "BC should be 0 after LDIR");
    assert!(!cpu.flag(FLAG_PV), "PV should be clear after LDIR completes");
}

// ── CALL + RET stack round-trip ────────────────────────────────────────────

#[test]
fn test_call_ret() {
    // At 0x0000: CALL 0x0010 ; LD B, 0x01 ; HALT
    // At 0x0010: LD A, 0x42  ; RET
    let mut prog = [0x76u8; 0x20];
    // Main code at 0x0000
    prog[0x00] = 0xCD; prog[0x01] = 0x10; prog[0x02] = 0x00; // CALL 0x0010
    prog[0x03] = 0x06; prog[0x04] = 0x01;                     // LD B, 1
    prog[0x05] = 0x76;                                         // HALT
    // Subroutine at 0x0010
    prog[0x10] = 0x3E; prog[0x11] = 0x42; // LD A, 0x42
    prog[0x12] = 0xC9;                     // RET

    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    cpu.sp = 0xFFFF;
    mem.load_ram(0x0000, &prog);

    for _ in 0..50 {
        if cpu.halted { break; }
        cpu.step(&mut mem, &mut NullIo);
    }

    assert_eq!(cpu.a, 0x42, "A should be 0x42 from subroutine");
    assert_eq!(cpu.b, 0x01, "B should be 0x01 from code after CALL");
    assert!(cpu.halted);
}

// ── BIT 3,(HL) flag check ──────────────────────────────────────────────────

#[test]
fn test_bit3_hl() {
    // Write 0x08 (bit 3 set) to 0x2000
    // LD HL, 0x2000 ; BIT 3,(HL) ; HALT
    let prog = &[
        0x21, 0x00, 0x20, // LD HL, 0x2000
        0xCB, 0x5E,       // BIT 3,(HL)  — opcode: CB (3<<3)|6 = CB 5E
        0x76,             // HALT
    ];
    let mut cpu = Cpu::new();
    let mut mem = Memory::new();
    mem.load_ram(0x0000, prog);
    mem.load_ram(0x2000, &[0x08]);

    for _ in 0..50 {
        if cpu.halted { break; }
        cpu.step(&mut mem, &mut NullIo);
    }

    // Bit 3 of 0x08 is set → Z flag should be CLEAR (bit found)
    assert!(!cpu.flag(FLAG_Z), "Z should be clear: bit 3 is set in memory");

    // Now test with 0x00 — bit 3 not set, Z should be set
    let mut cpu2 = Cpu::new();
    let mut mem2 = Memory::new();
    mem2.load_ram(0x0000, prog);
    mem2.load_ram(0x2000, &[0x00]);
    for _ in 0..50 {
        if cpu2.halted { break; }
        cpu2.step(&mut mem2, &mut NullIo);
    }
    assert!(cpu2.flag(FLAG_Z), "Z should be set: bit 3 is clear in memory");
}

// ── JP / JR conditional branching ─────────────────────────────────────────

#[test]
fn test_jp_conditional() {
    // LD A, 5 ; CP 5 ; JP Z, 0x0010 ; LD B, 0xFF ; HALT
    // At 0x0010: LD B, 0x01 ; HALT
    let mut prog = [0x76u8; 0x20];
    prog[0x00] = 0x3E; prog[0x01] = 0x05; // LD A, 5
    prog[0x02] = 0xFE; prog[0x03] = 0x05; // CP 5   → Z set
    prog[0x04] = 0xCA; prog[0x05] = 0x10; prog[0x06] = 0x00; // JP Z, 0x0010
    prog[0x07] = 0x06; prog[0x08] = 0xFF; // LD B, 0xFF  (skipped)
    prog[0x09] = 0x76;                     // HALT (skipped)
    prog[0x10] = 0x06; prog[0x11] = 0x01; // LD B, 1
    prog[0x12] = 0x76;                     // HALT

    let (cpu, _) = run_n(&prog, 100);
    assert_eq!(cpu.b, 0x01, "JP Z should have jumped to 0x0010");
}

#[test]
fn test_jr_loop() {
    // LD A, 0 ; loop: INC A ; CP 5 ; JR NZ, loop ; HALT
    // 3E 00 | 3C | FE 05 | 20 FB | 76
    // JR NZ offset: loop is at 0x02, after JR NZ we're at 0x07, offset = 0x02 - 0x07 = -5 = 0xFB
    let prog = &[
        0x3E, 0x00, // LD A, 0     — 0x00
        0x3C,       // INC A       — 0x02
        0xFE, 0x05, // CP 5        — 0x03
        0x20, 0xFB, // JR NZ, -5  — 0x05 (back to 0x02)
        0x76,       // HALT        — 0x07
    ];
    let (cpu, _) = run_program(prog);
    assert_eq!(cpu.a, 5, "A should be 5 when loop exits");
}

// ── Memory read-only ROM guard ─────────────────────────────────────────────

#[test]
fn test_rom_read_only() {
    let mut mem = Memory::new();
    mem.load_rom(&[0xAA; 0x4000]);
    mem.write_byte(0x0000, 0xFF); // attempt to write to ROM
    assert_eq!(mem.read_byte(0x0000), 0xAA, "ROM write should be silently ignored");
}

#[test]
fn test_ram_writable() {
    let mut mem = Memory::new();
    mem.load_rom(&[0xAA; 0x4000]);
    mem.write_byte(0x4000, 0x42);
    assert_eq!(mem.read_byte(0x4000), 0x42, "RAM at 0x4000 should be writable");
}
