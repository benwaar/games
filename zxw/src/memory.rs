/// 64K flat address space: 0x0000–0x3FFF ROM (read-only), 0x4000–0xFFFF RAM.
pub struct Memory {
    data: [u8; 0x10000],
    rom_loaded: bool,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            data: [0u8; 0x10000],
            rom_loaded: false,
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
}

impl Default for Memory {
    fn default() -> Self {
        Memory::new()
    }
}
