use std::collections::HashMap;

/// 64K flat address space: 0x0000–0x3FFF ROM (read-only), 0x4000–0xFFFF RAM.
pub struct Memory {
    data: [u8; 0x10000],
    rom_loaded: bool,
    /// Address → last recorded value. Changed when add_watchpoint is called.
    watchpoints: HashMap<u16, u8>,
    /// Accumulated hits since the last drain_watchpoint_hits call.
    watchpoint_hits: Vec<(u16, u8, u8)>,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            data: [0u8; 0x10000],
            rom_loaded: false,
            watchpoints: HashMap::new(),
            watchpoint_hits: Vec::new(),
        }
    }

    pub fn read_byte(&self, addr: u16) -> u8 {
        self.data[addr as usize]
    }

    pub fn write_byte(&mut self, addr: u16, val: u8) {
        if addr < 0x4000 && self.rom_loaded {
            // ROM is read-only — silently ignore writes
            return;
        }
        // Watchpoint: record if a watched address changes value.
        if let Some(old) = self.watchpoints.get_mut(&addr) {
            if val != *old {
                self.watchpoint_hits.push((addr, *old, val));
                *old = val;
            }
        }
        self.data[addr as usize] = val;
    }

    pub fn read_word(&self, addr: u16) -> u16 {
        let lo = self.read_byte(addr) as u16;
        let hi = self.read_byte(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    pub fn write_word(&mut self, addr: u16, val: u16) {
        self.write_byte(addr, (val & 0xFF) as u8);
        self.write_byte(addr.wrapping_add(1), (val >> 8) as u8);
    }

    /// Load up to 16K ROM into 0x0000–0x3FFF. Marks ROM region read-only.
    pub fn load_rom(&mut self, data: &[u8]) {
        let len = data.len().min(0x4000);
        self.data[..len].copy_from_slice(&data[..len]);
        self.rom_loaded = true;
    }

    /// Load arbitrary data into RAM (0x4000+).
    pub fn load_ram(&mut self, addr: u16, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            let target = addr.wrapping_add(i as u16);
            self.data[target as usize] = b;
        }
    }

    // ------------------------------------------------------------------
    // Watchpoint API
    // ------------------------------------------------------------------

    /// Start watching an address. Records its current value as the baseline.
    pub fn add_watchpoint(&mut self, addr: u16) {
        let current = self.data[addr as usize];
        self.watchpoints.insert(addr, current);
    }

    /// Stop watching an address.
    pub fn remove_watchpoint(&mut self, addr: u16) {
        self.watchpoints.remove(&addr);
    }

    /// Remove all watchpoints.
    pub fn clear_watchpoints(&mut self) {
        self.watchpoints.clear();
    }

    /// Drain and return accumulated (addr, old_value, new_value) hits.
    pub fn drain_watchpoint_hits(&mut self) -> Vec<(u16, u8, u8)> {
        std::mem::take(&mut self.watchpoint_hits)
    }

    // ------------------------------------------------------------------
    // Debug read (used by WASM API — avoids any side-effects)
    // ------------------------------------------------------------------

    /// Read `len` bytes starting at `addr`, wrapping at 0xFFFF.
    pub fn read_slice(&self, addr: u16, len: u16) -> Vec<u8> {
        (0..len)
            .map(|i| self.data[addr.wrapping_add(i) as usize])
            .collect()
    }
}

impl Default for Memory {
    fn default() -> Self {
        Memory::new()
    }
}
