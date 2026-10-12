/// M7 debug API tests — breakpoints, memory read/write, watchpoints.
use zxw::memory::Memory;
use zxw::spectrum::Spectrum;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Load a program at 0x0000, run one full frame, return the Spectrum.
fn spectrum_with_prog(prog: &[u8]) -> Spectrum {
    let mut s = Spectrum::new();
    s.memory.load_ram(0x0000, prog);
    s
}

// ── Memory read / write round-trip ────────────────────────────────────────────

#[test]
fn memory_read_write_roundtrip() {
    let mut mem = Memory::new();
    mem.load_ram(0x4000, &[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(mem.read_byte(0x4000), 0xDE);
    assert_eq!(mem.read_byte(0x4003), 0xEF);

    mem.write_byte(0x4001, 0xFF);
    assert_eq!(mem.read_byte(0x4001), 0xFF);

    let slice = mem.read_slice(0x4000, 4);
    assert_eq!(slice, vec![0xDE, 0xFF, 0xBE, 0xEF]);
}

#[test]
fn memory_read_slice_wraps_at_64k() {
    let mut mem = Memory::new();
    // Write to 0xFFFE and 0xFFFF; read_slice from 0xFFFE with len 4 should wrap.
    mem.write_byte(0xFFFE, 0x11);
    mem.write_byte(0xFFFF, 0x22);
    mem.write_byte(0x0000, 0x33);
    mem.write_byte(0x0001, 0x44);
    let slice = mem.read_slice(0xFFFE, 4);
    assert_eq!(slice, vec![0x11, 0x22, 0x33, 0x44]);
}

// ── Watchpoints ───────────────────────────────────────────────────────────────

#[test]
fn watchpoint_fires_on_change() {
    let mut mem = Memory::new();
    mem.load_ram(0x5000, &[0x00]);
    mem.add_watchpoint(0x5000);

    // No hit yet — same value
    mem.write_byte(0x5000, 0x00);
    assert!(mem.drain_watchpoint_hits().is_empty());

    // Hit: value changed
    mem.write_byte(0x5000, 0x42);
    let hits = mem.drain_watchpoint_hits();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0], (0x5000, 0x00, 0x42));

    // Drain again — cleared
    assert!(mem.drain_watchpoint_hits().is_empty());
}

#[test]
fn watchpoint_tracks_multiple_writes() {
    let mut mem = Memory::new();
    mem.load_ram(0x6000, &[0x00]);
    mem.add_watchpoint(0x6000);

    mem.write_byte(0x6000, 0x01);
    mem.write_byte(0x6000, 0x02);
    mem.write_byte(0x6000, 0x02); // no change — no hit
    mem.write_byte(0x6000, 0x03);

    let hits = mem.drain_watchpoint_hits();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].2, 0x01);
    assert_eq!(hits[1].2, 0x02);
    assert_eq!(hits[2].2, 0x03);
}

#[test]
fn watchpoint_removed_no_more_hits() {
    let mut mem = Memory::new();
    mem.load_ram(0x7000, &[0x00]);
    mem.add_watchpoint(0x7000);
    mem.write_byte(0x7000, 0x99); // hit
    mem.drain_watchpoint_hits();

    mem.remove_watchpoint(0x7000);
    mem.write_byte(0x7000, 0xAA); // removed — no hit
    assert!(mem.drain_watchpoint_hits().is_empty());
}

// ── Breakpoints ───────────────────────────────────────────────────────────────

/// A minimal Z80 program: two NOPs then a HALT.
/// NOP = 0x00, HALT = 0x76.
///   0x0000: NOP
///   0x0001: NOP
///   0x0002: HALT
const NOP_NOP_HALT: &[u8] = &[0x00, 0x00, 0x76];

#[test]
fn breakpoint_pauses_before_instruction() {
    let mut s = spectrum_with_prog(NOP_NOP_HALT);
    // Set breakpoint at second NOP (0x0001).
    s.breakpoints.insert(0x0001);

    s.step_frame();

    // Should have paused: PC is at 0x0001 (breakpoint), not yet halted.
    assert!(s.paused, "expected paused == true");
    assert_eq!(s.cpu.pc, 0x0001, "expected PC == 0x0001");
    assert!(!s.cpu.halted, "expected not halted yet");
}

#[test]
fn step_instruction_advances_past_breakpoint() {
    let mut s = spectrum_with_prog(NOP_NOP_HALT);
    s.breakpoints.insert(0x0001);
    s.step_frame(); // pauses at 0x0001

    // step_instruction should execute the NOP at 0x0001, moving PC to 0x0002.
    s.paused = false;
    s.step_instruction();
    assert_eq!(s.cpu.pc, 0x0002);
}

#[test]
fn clear_breakpoint_lets_execution_continue() {
    let mut s = spectrum_with_prog(NOP_NOP_HALT);
    s.breakpoints.insert(0x0001);
    s.step_frame(); // pauses at 0x0001

    s.paused = false;
    s.breakpoints.remove(&0x0001);
    s.step_frame(); // should run through to HALT

    assert!(s.cpu.halted, "expected HALT after clearing breakpoint");
}

#[test]
fn no_breakpoints_runs_to_halt() {
    let mut s = spectrum_with_prog(NOP_NOP_HALT);
    s.step_frame();
    assert!(s.cpu.halted);
    assert!(!s.paused);
}

// ── Pause / resume ────────────────────────────────────────────────────────────

#[test]
fn pause_prevents_cpu_advance() {
    let mut s = spectrum_with_prog(NOP_NOP_HALT);
    s.paused = true;
    let pc_before = s.cpu.pc;
    s.step_frame();
    assert_eq!(s.cpu.pc, pc_before, "PC must not advance when paused");
}

#[test]
fn resume_allows_cpu_advance() {
    let mut s = spectrum_with_prog(NOP_NOP_HALT);
    s.paused = true;
    s.step_frame(); // no-op
    s.paused = false;
    s.step_frame(); // runs
    assert!(s.cpu.halted);
}
