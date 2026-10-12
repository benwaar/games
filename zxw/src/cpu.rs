use crate::memory::Memory;

// Flag bit positions in F register
pub const FLAG_C: u8 = 0x01; // bit 0 — Carry
pub const FLAG_N: u8 = 0x02; // bit 1 — Add/Subtract
pub const FLAG_PV: u8 = 0x04; // bit 2 — Parity/Overflow
pub const FLAG_H: u8 = 0x10; // bit 4 — Half-carry
pub const FLAG_Z: u8 = 0x40; // bit 6 — Zero
pub const FLAG_S: u8 = 0x80; // bit 7 — Sign

/// Full Z80 register file.
#[derive(Debug, Default, Clone)]
pub struct Cpu {
    // Main registers
    pub a: u8,
    pub f: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub e: u8,
    pub h: u8,
    pub l: u8,
    // Alternate registers
    pub a2: u8,
    pub f2: u8,
    pub b2: u8,
    pub c2: u8,
    pub d2: u8,
    pub e2: u8,
    pub h2: u8,
    pub l2: u8,
    // Index / special
    pub ix: u16,
    pub iy: u16,
    pub sp: u16,
    pub pc: u16,
    pub i: u8,
    pub r: u8,
    pub iff1: bool,
    pub iff2: bool,
    pub im: u8,
    pub halted: bool,
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            sp: 0xFFFF,
            ..Default::default()
        }
    }

    // --- Register pair helpers ---

    pub fn af(&self) -> u16 { ((self.a as u16) << 8) | self.f as u16 }
    pub fn bc(&self) -> u16 { ((self.b as u16) << 8) | self.c as u16 }
    pub fn de(&self) -> u16 { ((self.d as u16) << 8) | self.e as u16 }
    pub fn hl(&self) -> u16 { ((self.h as u16) << 8) | self.l as u16 }

    pub fn set_af(&mut self, v: u16) { self.a = (v >> 8) as u8; self.f = (v & 0xFF) as u8; }
    pub fn set_bc(&mut self, v: u16) { self.b = (v >> 8) as u8; self.c = (v & 0xFF) as u8; }
    pub fn set_de(&mut self, v: u16) { self.d = (v >> 8) as u8; self.e = (v & 0xFF) as u8; }
    pub fn set_hl(&mut self, v: u16) { self.h = (v >> 8) as u8; self.l = (v & 0xFF) as u8; }

    // --- Flag helpers ---

    pub fn flag(&self, mask: u8) -> bool { self.f & mask != 0 }
    pub fn set_flag(&mut self, mask: u8, v: bool) {
        if v { self.f |= mask; } else { self.f &= !mask; }
    }

    fn set_szp(&mut self, result: u8) {
        self.set_flag(FLAG_S, result & 0x80 != 0);
        self.set_flag(FLAG_Z, result == 0);
        self.set_flag(FLAG_PV, parity(result));
    }

    // --- ADD A, val ---
    fn add_a(&mut self, val: u8, carry: bool) {
        let cy = if carry && self.flag(FLAG_C) { 1u16 } else { 0 };
        let result = self.a as u16 + val as u16 + cy;
        let half = (self.a & 0x0F) + (val & 0x0F) + cy as u8;
        self.set_flag(FLAG_C, result > 0xFF);
        self.set_flag(FLAG_H, half > 0x0F);
        self.set_flag(FLAG_N, false);
        let r8 = result as u8;
        // Overflow: two positives → negative, or two negatives → positive
        let overflow = (!(self.a ^ val) & (self.a ^ r8)) & 0x80 != 0;
        self.set_flag(FLAG_PV, overflow);
        self.set_flag(FLAG_S, r8 & 0x80 != 0);
        self.set_flag(FLAG_Z, r8 == 0);
        self.a = r8;
    }

    fn sub_a(&mut self, val: u8, borrow: bool) {
        let cy = if borrow && self.flag(FLAG_C) { 1u16 } else { 0 };
        let result = (self.a as u16).wrapping_sub(val as u16).wrapping_sub(cy);
        let half = (self.a & 0x0F) as i16 - (val & 0x0F) as i16 - cy as i16;
        self.set_flag(FLAG_C, result > 0xFF);
        self.set_flag(FLAG_H, half < 0);
        self.set_flag(FLAG_N, true);
        let r8 = result as u8;
        let overflow = ((self.a ^ val) & (self.a ^ r8)) & 0x80 != 0;
        self.set_flag(FLAG_PV, overflow);
        self.set_flag(FLAG_S, r8 & 0x80 != 0);
        self.set_flag(FLAG_Z, r8 == 0);
        self.a = r8;
    }

    fn and_a(&mut self, val: u8) {
        self.a &= val;
        self.set_szp(self.a);
        self.set_flag(FLAG_H, true);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_C, false);
    }

    fn or_a(&mut self, val: u8) {
        self.a |= val;
        self.set_szp(self.a);
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_C, false);
    }

    fn xor_a(&mut self, val: u8) {
        self.a ^= val;
        self.set_szp(self.a);
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_C, false);
    }

    fn cp_a(&mut self, val: u8) {
        let old_a = self.a;
        self.sub_a(val, false);
        self.a = old_a; // CP doesn't store result
    }

    fn inc8(&mut self, val: u8) -> u8 {
        let result = val.wrapping_add(1);
        self.set_flag(FLAG_H, (val & 0x0F) == 0x0F);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_PV, val == 0x7F);
        self.set_flag(FLAG_S, result & 0x80 != 0);
        self.set_flag(FLAG_Z, result == 0);
        result
    }

    fn dec8(&mut self, val: u8) -> u8 {
        let result = val.wrapping_sub(1);
        self.set_flag(FLAG_H, (val & 0x0F) == 0);
        self.set_flag(FLAG_N, true);
        self.set_flag(FLAG_PV, val == 0x80);
        self.set_flag(FLAG_S, result & 0x80 != 0);
        self.set_flag(FLAG_Z, result == 0);
        result
    }

    // --- Stack helpers ---

    fn push16(&mut self, mem: &mut Memory, val: u16) {
        self.sp = self.sp.wrapping_sub(2);
        mem.write_word(self.sp, val);
    }

    fn pop16(&mut self, mem: &mut Memory) -> u16 {
        let v = mem.read_word(self.sp);
        self.sp = self.sp.wrapping_add(2);
        v
    }

    // --- Fetch helpers ---

    fn fetch_byte(&mut self, mem: &Memory) -> u8 {
        let b = mem.read_byte(self.pc);
        self.pc = self.pc.wrapping_add(1);
        b
    }

    fn fetch_word(&mut self, mem: &Memory) -> u16 {
        let lo = self.fetch_byte(mem) as u16;
        let hi = self.fetch_byte(mem) as u16;
        lo | (hi << 8)
    }

    // --- Register number → value / setter (encoding 0..7 = B,C,D,E,H,L,(HL),A) ---

    fn reg_read(&self, r: u8, mem: &Memory) -> u8 {
        match r {
            0 => self.b,
            1 => self.c,
            2 => self.d,
            3 => self.e,
            4 => self.h,
            5 => self.l,
            6 => mem.read_byte(self.hl()),
            7 => self.a,
            _ => unreachable!(),
        }
    }

    fn reg_write(&mut self, r: u8, val: u8, mem: &mut Memory) {
        match r {
            0 => self.b = val,
            1 => self.c = val,
            2 => self.d = val,
            3 => self.e = val,
            4 => self.h = val,
            5 => self.l = val,
            6 => mem.write_byte(self.hl(), val),
            7 => self.a = val,
            _ => unreachable!(),
        }
    }

    // --- Main fetch-decode-execute ---

    pub fn step(&mut self, mem: &mut Memory) -> u32 {
        if self.halted {
            return 4;
        }

        let op = self.fetch_byte(mem);
        self.r = self.r.wrapping_add(1);

        match op {
            // NOP
            0x00 => 4,

            // LD BC/DE/HL/SP, nn
            0x01 => { let nn = self.fetch_word(mem); self.set_bc(nn); 10 }
            0x11 => { let nn = self.fetch_word(mem); self.set_de(nn); 10 }
            0x21 => { let nn = self.fetch_word(mem); self.set_hl(nn); 10 }
            0x31 => { let nn = self.fetch_word(mem); self.sp = nn; 10 }

            // INC rr
            0x03 => { self.set_bc(self.bc().wrapping_add(1)); 6 }
            0x13 => { self.set_de(self.de().wrapping_add(1)); 6 }
            0x23 => { self.set_hl(self.hl().wrapping_add(1)); 6 }
            0x33 => { self.sp = self.sp.wrapping_add(1); 6 }

            // DEC rr
            0x0B => { self.set_bc(self.bc().wrapping_sub(1)); 6 }
            0x1B => { self.set_de(self.de().wrapping_sub(1)); 6 }
            0x2B => { self.set_hl(self.hl().wrapping_sub(1)); 6 }
            0x3B => { self.sp = self.sp.wrapping_sub(1); 6 }

            // INC r (B,C,D,E,H,L,(HL),A) = 0x04,0x0C,0x14,0x1C,0x24,0x2C,0x34,0x3C
            0x04 => { self.b = self.inc8(self.b); 4 }
            0x0C => { self.c = self.inc8(self.c); 4 }
            0x14 => { self.d = self.inc8(self.d); 4 }
            0x1C => { self.e = self.inc8(self.e); 4 }
            0x24 => { self.h = self.inc8(self.h); 4 }
            0x2C => { self.l = self.inc8(self.l); 4 }
            0x34 => { let v = mem.read_byte(self.hl()); let r = self.inc8(v); mem.write_byte(self.hl(), r); 11 }
            0x3C => { self.a = self.inc8(self.a); 4 }

            // DEC r
            0x05 => { self.b = self.dec8(self.b); 4 }
            0x0D => { self.c = self.dec8(self.c); 4 }
            0x15 => { self.d = self.dec8(self.d); 4 }
            0x1D => { self.e = self.dec8(self.e); 4 }
            0x25 => { self.h = self.dec8(self.h); 4 }
            0x2D => { self.l = self.dec8(self.l); 4 }
            0x35 => { let v = mem.read_byte(self.hl()); let r = self.dec8(v); mem.write_byte(self.hl(), r); 11 }
            0x3D => { self.a = self.dec8(self.a); 4 }

            // LD r, n (immediate)
            0x06 => { self.b = self.fetch_byte(mem); 7 }
            0x0E => { self.c = self.fetch_byte(mem); 7 }
            0x16 => { self.d = self.fetch_byte(mem); 7 }
            0x1E => { self.e = self.fetch_byte(mem); 7 }
            0x26 => { self.h = self.fetch_byte(mem); 7 }
            0x2E => { self.l = self.fetch_byte(mem); 7 }
            0x36 => { let n = self.fetch_byte(mem); mem.write_byte(self.hl(), n); 10 }
            0x3E => { self.a = self.fetch_byte(mem); 7 }

            // RLCA
            0x07 => {
                let c = (self.a >> 7) & 1;
                self.a = (self.a << 1) | c;
                self.set_flag(FLAG_C, c != 0);
                self.set_flag(FLAG_H, false);
                self.set_flag(FLAG_N, false);
                4
            }
            // RRCA
            0x0F => {
                let c = self.a & 1;
                self.a = (self.a >> 1) | (c << 7);
                self.set_flag(FLAG_C, c != 0);
                self.set_flag(FLAG_H, false);
                self.set_flag(FLAG_N, false);
                4
            }
            // RLA
            0x17 => {
                let old_c = if self.flag(FLAG_C) { 1u8 } else { 0 };
                let new_c = (self.a >> 7) & 1;
                self.a = (self.a << 1) | old_c;
                self.set_flag(FLAG_C, new_c != 0);
                self.set_flag(FLAG_H, false);
                self.set_flag(FLAG_N, false);
                4
            }
            // RRA
            0x1F => {
                let old_c = if self.flag(FLAG_C) { 0x80u8 } else { 0 };
                let new_c = self.a & 1;
                self.a = (self.a >> 1) | old_c;
                self.set_flag(FLAG_C, new_c != 0);
                self.set_flag(FLAG_H, false);
                self.set_flag(FLAG_N, false);
                4
            }

            // EX AF, AF'
            0x08 => {
                core::mem::swap(&mut self.a, &mut self.a2);
                core::mem::swap(&mut self.f, &mut self.f2);
                4
            }

            // DJNZ e
            0x10 => {
                let offset = self.fetch_byte(mem) as i8;
                self.b = self.b.wrapping_sub(1);
                if self.b != 0 {
                    self.pc = (self.pc as i32 + offset as i32) as u16;
                    13
                } else {
                    8
                }
            }

            // JR e
            0x18 => {
                let offset = self.fetch_byte(mem) as i8;
                self.pc = (self.pc as i32 + offset as i32) as u16;
                12
            }
            // JR NZ, e
            0x20 => {
                let offset = self.fetch_byte(mem) as i8;
                if !self.flag(FLAG_Z) { self.pc = (self.pc as i32 + offset as i32) as u16; 12 } else { 7 }
            }
            // JR Z, e
            0x28 => {
                let offset = self.fetch_byte(mem) as i8;
                if self.flag(FLAG_Z) { self.pc = (self.pc as i32 + offset as i32) as u16; 12 } else { 7 }
            }
            // JR NC, e
            0x30 => {
                let offset = self.fetch_byte(mem) as i8;
                if !self.flag(FLAG_C) { self.pc = (self.pc as i32 + offset as i32) as u16; 12 } else { 7 }
            }
            // JR C, e
            0x38 => {
                let offset = self.fetch_byte(mem) as i8;
                if self.flag(FLAG_C) { self.pc = (self.pc as i32 + offset as i32) as u16; 12 } else { 7 }
            }

            // LD (BC),A / (DE),A
            0x02 => { mem.write_byte(self.bc(), self.a); 7 }
            0x12 => { mem.write_byte(self.de(), self.a); 7 }

            // LD A,(BC) / (DE)
            0x0A => { self.a = mem.read_byte(self.bc()); 7 }
            0x1A => { self.a = mem.read_byte(self.de()); 7 }

            // LD (nn), HL
            0x22 => { let nn = self.fetch_word(mem); mem.write_word(nn, self.hl()); 16 }
            // LD HL, (nn)
            0x2A => { let nn = self.fetch_word(mem); let v = mem.read_word(nn); self.set_hl(v); 16 }
            // LD (nn), A
            0x32 => { let nn = self.fetch_word(mem); mem.write_byte(nn, self.a); 13 }
            // LD A, (nn)
            0x3A => { let nn = self.fetch_word(mem); self.a = mem.read_byte(nn); 13 }

            // ADD HL, rr
            0x09 => { let v = self.bc(); self.add_hl(v); 11 }
            0x19 => { let v = self.de(); self.add_hl(v); 11 }
            0x29 => { let v = self.hl(); self.add_hl(v); 11 }
            0x39 => { let v = self.sp; self.add_hl(v); 11 }

            // DAA
            0x27 => { self.daa(); 4 }
            // CPL
            0x2F => { self.a = !self.a; self.set_flag(FLAG_H, true); self.set_flag(FLAG_N, true); 4 }
            // SCF
            0x37 => { self.set_flag(FLAG_C, true); self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false); 4 }
            // CCF
            0x3F => {
                let c = self.flag(FLAG_C);
                self.set_flag(FLAG_H, c);
                self.set_flag(FLAG_C, !c);
                self.set_flag(FLAG_N, false);
                4
            }

            // LD r, r' — 0x40..0x7F (excluding 0x76 = HALT)
            0x40..=0x7F => {
                if op == 0x76 {
                    self.halted = true;
                    return 4;
                }
                let dst = (op >> 3) & 0x07;
                let src = op & 0x07;
                let val = self.reg_read(src, mem);
                let cycles = if src == 6 || dst == 6 { 7 } else { 4 };
                self.reg_write(dst, val, mem);
                cycles
            }

            // Arithmetic/logic on A: 0x80..0xBF
            0x80..=0xBF => {
                let r = op & 0x07;
                let val = self.reg_read(r, mem);
                let cycles = if r == 6 { 7 } else { 4 };
                match (op >> 3) & 0x07 {
                    0 => self.add_a(val, false),
                    1 => self.add_a(val, true),
                    2 => self.sub_a(val, false),
                    3 => self.sub_a(val, true),
                    4 => self.and_a(val),
                    5 => self.xor_a(val),
                    6 => self.or_a(val),
                    7 => self.cp_a(val),
                    _ => unreachable!(),
                }
                cycles
            }

            // ADD A,n / ADC A,n / SUB n / SBC A,n / AND n / XOR n / OR n / CP n
            0xC6 => { let n = self.fetch_byte(mem); self.add_a(n, false); 7 }
            0xCE => { let n = self.fetch_byte(mem); self.add_a(n, true); 7 }
            0xD6 => { let n = self.fetch_byte(mem); self.sub_a(n, false); 7 }
            0xDE => { let n = self.fetch_byte(mem); self.sub_a(n, true); 7 }
            0xE6 => { let n = self.fetch_byte(mem); self.and_a(n); 7 }
            0xEE => { let n = self.fetch_byte(mem); self.xor_a(n); 7 }
            0xF6 => { let n = self.fetch_byte(mem); self.or_a(n); 7 }
            0xFE => { let n = self.fetch_byte(mem); self.cp_a(n); 7 }

            // RET
            0xC9 => { self.pc = self.pop16(mem); 10 }

            // RET cc
            0xC0 => { if !self.flag(FLAG_Z) { self.pc = self.pop16(mem); 11 } else { 5 } }
            0xC8 => { if self.flag(FLAG_Z)  { self.pc = self.pop16(mem); 11 } else { 5 } }
            0xD0 => { if !self.flag(FLAG_C) { self.pc = self.pop16(mem); 11 } else { 5 } }
            0xD8 => { if self.flag(FLAG_C)  { self.pc = self.pop16(mem); 11 } else { 5 } }
            0xE0 => { if !self.flag(FLAG_PV){ self.pc = self.pop16(mem); 11 } else { 5 } }
            0xE8 => { if self.flag(FLAG_PV) { self.pc = self.pop16(mem); 11 } else { 5 } }
            0xF0 => { if !self.flag(FLAG_S) { self.pc = self.pop16(mem); 11 } else { 5 } }
            0xF8 => { if self.flag(FLAG_S)  { self.pc = self.pop16(mem); 11 } else { 5 } }

            // JP nn
            0xC3 => { let nn = self.fetch_word(mem); self.pc = nn; 10 }

            // JP cc, nn
            0xC2 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_Z) { self.pc = nn; } 10 }
            0xCA => { let nn = self.fetch_word(mem); if self.flag(FLAG_Z)  { self.pc = nn; } 10 }
            0xD2 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_C) { self.pc = nn; } 10 }
            0xDA => { let nn = self.fetch_word(mem); if self.flag(FLAG_C)  { self.pc = nn; } 10 }
            0xE2 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_PV){ self.pc = nn; } 10 }
            0xEA => { let nn = self.fetch_word(mem); if self.flag(FLAG_PV) { self.pc = nn; } 10 }
            0xF2 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_S) { self.pc = nn; } 10 }
            0xFA => { let nn = self.fetch_word(mem); if self.flag(FLAG_S)  { self.pc = nn; } 10 }

            // CALL nn
            0xCD => {
                let nn = self.fetch_word(mem);
                let ret = self.pc;
                self.push16(mem, ret);
                self.pc = nn;
                17
            }

            // CALL cc, nn
            0xC4 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_Z) { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xCC => { let nn = self.fetch_word(mem); if self.flag(FLAG_Z)  { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xD4 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_C) { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xDC => { let nn = self.fetch_word(mem); if self.flag(FLAG_C)  { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xE4 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_PV){ let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xEC => { let nn = self.fetch_word(mem); if self.flag(FLAG_PV) { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xF4 => { let nn = self.fetch_word(mem); if !self.flag(FLAG_S) { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }
            0xFC => { let nn = self.fetch_word(mem); if self.flag(FLAG_S)  { let r = self.pc; self.push16(mem, r); self.pc = nn; 17 } else { 10 } }

            // PUSH / POP rr
            0xC5 => { let v = self.bc(); self.push16(mem, v); 11 }
            0xD5 => { let v = self.de(); self.push16(mem, v); 11 }
            0xE5 => { let v = self.hl(); self.push16(mem, v); 11 }
            0xF5 => { let v = self.af(); self.push16(mem, v); 11 }

            0xC1 => { let v = self.pop16(mem); self.set_bc(v); 10 }
            0xD1 => { let v = self.pop16(mem); self.set_de(v); 10 }
            0xE1 => { let v = self.pop16(mem); self.set_hl(v); 10 }
            0xF1 => { let v = self.pop16(mem); self.set_af(v); 10 }

            // EXX
            0xD9 => {
                core::mem::swap(&mut self.b, &mut self.b2);
                core::mem::swap(&mut self.c, &mut self.c2);
                core::mem::swap(&mut self.d, &mut self.d2);
                core::mem::swap(&mut self.e, &mut self.e2);
                core::mem::swap(&mut self.h, &mut self.h2);
                core::mem::swap(&mut self.l, &mut self.l2);
                4
            }
            // EX DE, HL
            0xEB => {
                core::mem::swap(&mut self.d, &mut self.h);
                core::mem::swap(&mut self.e, &mut self.l);
                4
            }
            // EX (SP), HL
            0xE3 => {
                let top = mem.read_word(self.sp);
                let hl = self.hl();
                mem.write_word(self.sp, hl);
                self.set_hl(top);
                19
            }
            // JP (HL)
            0xE9 => { self.pc = self.hl(); 4 }
            // LD SP,HL
            0xF9 => { self.sp = self.hl(); 6 }

            // DI / EI
            0xF3 => { self.iff1 = false; self.iff2 = false; 4 }
            0xFB => { self.iff1 = true;  self.iff2 = true;  4 }

            // RST n
            0xC7 | 0xCF | 0xD7 | 0xDF | 0xE7 | 0xEF | 0xF7 | 0xFF => {
                let target = (op & 0x38) as u16;
                let ret = self.pc;
                self.push16(mem, ret);
                self.pc = target;
                11
            }

            // IN A,(n) — stub, returns 0xFF
            0xDB => { self.fetch_byte(mem); self.a = 0xFF; 11 }
            // OUT (n),A — stub, ignore
            0xD3 => { self.fetch_byte(mem); 11 }

            // CB prefix
            0xCB => self.step_cb(mem),

            // ED prefix
            0xED => self.step_ed(mem),

            // DD prefix (IX) — not implemented yet; panic with context
            0xDD => {
                let next = mem.read_byte(self.pc);
                panic!("DD prefix (IX) not implemented: PC={:#06x} next={:#04x}", self.pc - 1, next);
            }
            // FD prefix (IY) — not implemented yet; panic with context
            0xFD => {
                let next = mem.read_byte(self.pc);
                panic!("FD prefix (IY) not implemented: PC={:#06x} next={:#04x}", self.pc - 1, next);
            }
        }
    }

    // --- CB prefix (bit operations + shifts/rotates) ---

    fn step_cb(&mut self, mem: &mut Memory) -> u32 {
        let op = self.fetch_byte(mem);
        self.r = self.r.wrapping_add(1);
        let r = op & 0x07;
        let b = (op >> 3) & 0x07;
        let val = self.reg_read(r, mem);
        let mem_ref = r == 6;

        match op >> 6 {
            // Rotates and shifts (0x00..0x3F)
            0 => {
                let result = match b {
                    0 => self.rlc(val),
                    1 => self.rrc(val),
                    2 => self.rl(val),
                    3 => self.rr(val),
                    4 => self.sla(val),
                    5 => self.sra(val),
                    6 => self.sll(val), // undocumented SLL
                    7 => self.srl(val),
                    _ => unreachable!(),
                };
                self.reg_write(r, result, mem);
                if mem_ref { 15 } else { 8 }
            }
            // BIT b, r (0x40..0x7F)
            1 => {
                let mask = 1u8 << b;
                self.set_flag(FLAG_Z, val & mask == 0);
                self.set_flag(FLAG_H, true);
                self.set_flag(FLAG_N, false);
                if mem_ref { 12 } else { 8 }
            }
            // RES b, r (0x80..0xBF)
            2 => {
                let result = val & !(1u8 << b);
                self.reg_write(r, result, mem);
                if mem_ref { 15 } else { 8 }
            }
            // SET b, r (0xC0..0xFF)
            3 => {
                let result = val | (1u8 << b);
                self.reg_write(r, result, mem);
                if mem_ref { 15 } else { 8 }
            }
            _ => unreachable!(),
        }
    }

    fn rlc(&mut self, val: u8) -> u8 {
        let c = val >> 7;
        let r = (val << 1) | c;
        self.set_flag(FLAG_C, c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn rrc(&mut self, val: u8) -> u8 {
        let c = val & 1;
        let r = (val >> 1) | (c << 7);
        self.set_flag(FLAG_C, c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn rl(&mut self, val: u8) -> u8 {
        let old_c = if self.flag(FLAG_C) { 1u8 } else { 0 };
        let new_c = val >> 7;
        let r = (val << 1) | old_c;
        self.set_flag(FLAG_C, new_c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn rr(&mut self, val: u8) -> u8 {
        let old_c = if self.flag(FLAG_C) { 0x80u8 } else { 0 };
        let new_c = val & 1;
        let r = (val >> 1) | old_c;
        self.set_flag(FLAG_C, new_c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn sla(&mut self, val: u8) -> u8 {
        let c = val >> 7;
        let r = val << 1;
        self.set_flag(FLAG_C, c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn sra(&mut self, val: u8) -> u8 {
        let c = val & 1;
        let r = ((val as i8) >> 1) as u8;
        self.set_flag(FLAG_C, c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn sll(&mut self, val: u8) -> u8 {
        let c = val >> 7;
        let r = (val << 1) | 1;
        self.set_flag(FLAG_C, c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }
    fn srl(&mut self, val: u8) -> u8 {
        let c = val & 1;
        let r = val >> 1;
        self.set_flag(FLAG_C, c != 0);
        self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_PV, parity(r));
        r
    }

    // --- ED prefix ---

    fn step_ed(&mut self, mem: &mut Memory) -> u32 {
        let op = self.fetch_byte(mem);
        self.r = self.r.wrapping_add(1);

        match op {
            // IN r,(C)
            0x40 => { self.b = 0xFF; 12 }
            0x48 => { self.c = 0xFF; 12 }
            0x50 => { self.d = 0xFF; 12 }
            0x58 => { self.e = 0xFF; 12 }
            0x60 => { self.h = 0xFF; 12 }
            0x68 => { self.l = 0xFF; 12 }
            0x70 => { /* IN F,(C) — flags only, discard */ 12 }
            0x78 => { self.a = 0xFF; 12 }

            // OUT (C),r — stubs
            0x41 | 0x49 | 0x51 | 0x59 | 0x61 | 0x69 | 0x71 | 0x79 => { 12 }

            // SBC HL, rr
            0x42 => { let v = self.bc(); self.sbc_hl(v); 15 }
            0x52 => { let v = self.de(); self.sbc_hl(v); 15 }
            0x62 => { let v = self.hl(); self.sbc_hl(v); 15 }
            0x72 => { let v = self.sp; self.sbc_hl(v); 15 }

            // ADC HL, rr
            0x4A => { let v = self.bc(); self.adc_hl(v); 15 }
            0x5A => { let v = self.de(); self.adc_hl(v); 15 }
            0x6A => { let v = self.hl(); self.adc_hl(v); 15 }
            0x7A => { let v = self.sp; self.adc_hl(v); 15 }

            // LD (nn), rr / LD rr, (nn) — ED extended
            0x43 => { let nn = self.fetch_word(mem); mem.write_word(nn, self.bc()); 20 }
            0x53 => { let nn = self.fetch_word(mem); mem.write_word(nn, self.de()); 20 }
            0x63 => { let nn = self.fetch_word(mem); mem.write_word(nn, self.hl()); 20 }
            0x73 => { let nn = self.fetch_word(mem); mem.write_word(nn, self.sp); 20 }
            0x4B => { let nn = self.fetch_word(mem); let v = mem.read_word(nn); self.set_bc(v); 20 }
            0x5B => { let nn = self.fetch_word(mem); let v = mem.read_word(nn); self.set_de(v); 20 }
            0x6B => { let nn = self.fetch_word(mem); let v = mem.read_word(nn); self.set_hl(v); 20 }
            0x7B => { let nn = self.fetch_word(mem); self.sp = mem.read_word(nn); 20 }

            // NEG
            0x44 | 0x4C | 0x54 | 0x5C | 0x64 | 0x6C | 0x74 | 0x7C => {
                let old = self.a;
                self.a = 0;
                self.sub_a(old, false);
                8
            }

            // RETN / RETI
            0x45 | 0x4D | 0x55 | 0x5D | 0x65 | 0x6D | 0x75 | 0x7D => {
                self.iff1 = self.iff2;
                self.pc = self.pop16(mem);
                14
            }

            // IM 0/1/2
            0x46 | 0x4E | 0x66 | 0x6E => { self.im = 0; 8 }
            0x56 | 0x76 => { self.im = 1; 8 }
            0x5E | 0x7E => { self.im = 2; 8 }

            // LD A,I / LD A,R
            0x57 => {
                self.a = self.i;
                let a = self.a;
                self.set_flag(FLAG_S, a & 0x80 != 0);
                self.set_flag(FLAG_Z, a == 0);
                self.set_flag(FLAG_H, false);
                self.set_flag(FLAG_PV, self.iff2);
                self.set_flag(FLAG_N, false);
                9
            }
            0x5F => {
                self.a = self.r;
                let a = self.a;
                self.set_flag(FLAG_S, a & 0x80 != 0);
                self.set_flag(FLAG_Z, a == 0);
                self.set_flag(FLAG_H, false);
                self.set_flag(FLAG_PV, self.iff2);
                self.set_flag(FLAG_N, false);
                9
            }
            // LD I,A / LD R,A
            0x47 => { self.i = self.a; 9 }
            0x4F => { self.r = self.a; 9 }

            // RLD / RRD
            0x6F => {
                let hl_val = mem.read_byte(self.hl());
                let new_hl = (hl_val << 4) | (self.a & 0x0F);
                self.a = (self.a & 0xF0) | (hl_val >> 4);
                mem.write_byte(self.hl(), new_hl);
                let a = self.a;
                self.set_szp(a);
                self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
                18
            }
            0x67 => {
                let hl_val = mem.read_byte(self.hl());
                let new_hl = (self.a << 4) | (hl_val >> 4);
                self.a = (self.a & 0xF0) | (hl_val & 0x0F);
                mem.write_byte(self.hl(), new_hl);
                let a = self.a;
                self.set_szp(a);
                self.set_flag(FLAG_H, false); self.set_flag(FLAG_N, false);
                18
            }

            // LDIR
            0xB0 => self.ldir(mem),
            // LDDR
            0xB8 => self.lddr(mem),
            // CPIR
            0xB1 => self.cpir(mem),
            // CPDR
            0xB9 => self.cpdr(mem),
            // LDI
            0xA0 => self.ldi(mem),
            // LDD
            0xA8 => self.ldd(mem),
            // CPI
            0xA1 => self.cpi(mem),
            // CPD
            0xA9 => self.cpd(mem),

            // INIR / OTIR stubs
            0xB2 | 0xB3 | 0xBA | 0xBB => 16,

            // Undefined ED xx — treat as NOP (4 cycles)
            _ => 4,
        }
    }

    fn ldi(&mut self, mem: &mut Memory) -> u32 {
        let val = mem.read_byte(self.de());
        mem.write_byte(self.de(), val);
        let de = self.de().wrapping_add(1); self.set_de(de);
        let hl = self.hl().wrapping_add(1); self.set_hl(hl);
        let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_PV, bc != 0);
        16
    }

    fn ldd(&mut self, mem: &mut Memory) -> u32 {
        let val = mem.read_byte(self.hl());
        mem.write_byte(self.de(), val);
        let de = self.de().wrapping_sub(1); self.set_de(de);
        let hl = self.hl().wrapping_sub(1); self.set_hl(hl);
        let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_PV, bc != 0);
        16
    }

    fn ldir(&mut self, mem: &mut Memory) -> u32 {
        let mut cycles = 0u32;
        loop {
            let val = mem.read_byte(self.hl());
            mem.write_byte(self.de(), val);
            let de = self.de().wrapping_add(1); self.set_de(de);
            let hl = self.hl().wrapping_add(1); self.set_hl(hl);
            let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
            cycles += if bc != 0 { 21 } else { 16 };
            if bc == 0 { break; }
        }
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_PV, false);
        cycles
    }

    fn lddr(&mut self, mem: &mut Memory) -> u32 {
        let mut cycles = 0u32;
        loop {
            let val = mem.read_byte(self.hl());
            mem.write_byte(self.de(), val);
            let de = self.de().wrapping_sub(1); self.set_de(de);
            let hl = self.hl().wrapping_sub(1); self.set_hl(hl);
            let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
            cycles += if bc != 0 { 21 } else { 16 };
            if bc == 0 { break; }
        }
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_PV, false);
        cycles
    }

    fn cpi(&mut self, mem: &mut Memory) -> u32 {
        let val = mem.read_byte(self.hl());
        let result = self.a.wrapping_sub(val);
        let hl = self.hl().wrapping_add(1); self.set_hl(hl);
        let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
        let h = (self.a & 0x0F) < (val & 0x0F);
        self.set_flag(FLAG_S, result & 0x80 != 0);
        self.set_flag(FLAG_Z, result == 0);
        self.set_flag(FLAG_H, h);
        self.set_flag(FLAG_PV, bc != 0);
        self.set_flag(FLAG_N, true);
        16
    }

    fn cpd(&mut self, mem: &mut Memory) -> u32 {
        let val = mem.read_byte(self.hl());
        let result = self.a.wrapping_sub(val);
        let hl = self.hl().wrapping_sub(1); self.set_hl(hl);
        let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
        let h = (self.a & 0x0F) < (val & 0x0F);
        self.set_flag(FLAG_S, result & 0x80 != 0);
        self.set_flag(FLAG_Z, result == 0);
        self.set_flag(FLAG_H, h);
        self.set_flag(FLAG_PV, bc != 0);
        self.set_flag(FLAG_N, true);
        16
    }

    fn cpir(&mut self, mem: &mut Memory) -> u32 {
        let mut cycles = 0u32;
        loop {
            let val = mem.read_byte(self.hl());
            let result = self.a.wrapping_sub(val);
            let hl = self.hl().wrapping_add(1); self.set_hl(hl);
            let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
            let h = (self.a & 0x0F) < (val & 0x0F);
            self.set_flag(FLAG_S, result & 0x80 != 0);
            self.set_flag(FLAG_Z, result == 0);
            self.set_flag(FLAG_H, h);
            self.set_flag(FLAG_PV, bc != 0);
            self.set_flag(FLAG_N, true);
            let found = result == 0;
            cycles += if bc != 0 && !found { 21 } else { 16 };
            if bc == 0 || found { break; }
        }
        cycles
    }

    fn cpdr(&mut self, mem: &mut Memory) -> u32 {
        let mut cycles = 0u32;
        loop {
            let val = mem.read_byte(self.hl());
            let result = self.a.wrapping_sub(val);
            let hl = self.hl().wrapping_sub(1); self.set_hl(hl);
            let bc = self.bc().wrapping_sub(1); self.set_bc(bc);
            let h = (self.a & 0x0F) < (val & 0x0F);
            self.set_flag(FLAG_S, result & 0x80 != 0);
            self.set_flag(FLAG_Z, result == 0);
            self.set_flag(FLAG_H, h);
            self.set_flag(FLAG_PV, bc != 0);
            self.set_flag(FLAG_N, true);
            let found = result == 0;
            cycles += if bc != 0 && !found { 21 } else { 16 };
            if bc == 0 || found { break; }
        }
        cycles
    }

    fn add_hl(&mut self, val: u16) {
        let hl = self.hl();
        let result = hl as u32 + val as u32;
        self.set_flag(FLAG_C, result > 0xFFFF);
        self.set_flag(FLAG_H, (hl & 0x0FFF) + (val & 0x0FFF) > 0x0FFF);
        self.set_flag(FLAG_N, false);
        self.set_hl(result as u16);
    }

    fn adc_hl(&mut self, val: u16) {
        let hl = self.hl();
        let cy = if self.flag(FLAG_C) { 1u32 } else { 0 };
        let result = hl as u32 + val as u32 + cy;
        let overflow = (!(hl ^ val) & (hl ^ result as u16)) & 0x8000 != 0;
        self.set_flag(FLAG_C, result > 0xFFFF);
        self.set_flag(FLAG_H, (hl & 0x0FFF) + (val & 0x0FFF) + cy as u16 > 0x0FFF);
        self.set_flag(FLAG_N, false);
        self.set_flag(FLAG_PV, overflow);
        let r16 = result as u16;
        self.set_flag(FLAG_S, r16 & 0x8000 != 0);
        self.set_flag(FLAG_Z, r16 == 0);
        self.set_hl(r16);
    }

    fn sbc_hl(&mut self, val: u16) {
        let hl = self.hl();
        let cy = if self.flag(FLAG_C) { 1u32 } else { 0 };
        let result = (hl as u32).wrapping_sub(val as u32).wrapping_sub(cy);
        let overflow = ((hl ^ val) & (hl ^ result as u16)) & 0x8000 != 0;
        self.set_flag(FLAG_C, result > 0xFFFF);
        self.set_flag(FLAG_H, (hl & 0x0FFF) < (val & 0x0FFF) + cy as u16);
        self.set_flag(FLAG_N, true);
        self.set_flag(FLAG_PV, overflow);
        let r16 = result as u16;
        self.set_flag(FLAG_S, r16 & 0x8000 != 0);
        self.set_flag(FLAG_Z, r16 == 0);
        self.set_hl(r16);
    }

    fn daa(&mut self) {
        let mut correction = 0u8;
        let mut set_c = false;
        let a = self.a;
        let n = self.flag(FLAG_N);
        let h = self.flag(FLAG_H);
        let c = self.flag(FLAG_C);

        if !n {
            if h || (a & 0x0F) > 9 { correction |= 0x06; }
            if c || a > 0x99 { correction |= 0x60; set_c = true; }
            self.a = a.wrapping_add(correction);
        } else {
            if h || (a & 0x0F) > 9 { correction |= 0x06; }
            if c || a > 0x99 { correction |= 0x60; set_c = true; }
            self.a = a.wrapping_sub(correction);
        }
        let r = self.a;
        self.set_flag(FLAG_S, r & 0x80 != 0);
        self.set_flag(FLAG_Z, r == 0);
        self.set_flag(FLAG_H, false);
        self.set_flag(FLAG_PV, parity(r));
        self.set_flag(FLAG_C, set_c);
    }
}

fn parity(v: u8) -> bool {
    v.count_ones() % 2 == 0
}
