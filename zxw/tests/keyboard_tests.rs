use zxw::keyboard::Keyboard;

// ── No key pressed → read returns 0xFF (bits 5–7 set, bits 0–4 set) ───────

#[test]
fn no_key_pressed_returns_0xff_for_any_row() {
    let kb = Keyboard::new();
    // Select each individual row
    for row in 0..8u8 {
        let select = !(1u8 << row); // all bits 1 except bit `row`
        let result = kb.read_port(select);
        assert_eq!(
            result, 0xFF,
            "row {row}: expected 0xFF with no keys pressed, got {result:#04x}"
        );
    }
    // Select all rows at once (high byte = 0x00)
    assert_eq!(kb.read_port(0x00), 0xFF, "all rows selected, no keys → 0xFF");
}

// ── Single key press clears the correct bit ────────────────────────────────

#[test]
fn single_key_press_clears_correct_bit() {
    let mut kb = Keyboard::new();
    // Press row 0 bit 1 (Z key)
    kb.key_down(0, 1);

    // Row 0 selected: bit 1 should be 0
    let result = kb.read_port(0xFE); // A8 low → selects row 0
    assert_eq!(result & 0x1F, 0x1D, "bit 1 of row 0 should be clear (0x1D = 0b11101)");
    assert_eq!(result & 0xE0, 0xE0, "bits 5–7 must be 1");

    // Other rows unaffected
    let other = kb.read_port(0xFD); // A9 low → row 1
    assert_eq!(other, 0xFF, "row 1 should be unaffected");
}

#[test]
fn key_release_restores_bit() {
    let mut kb = Keyboard::new();
    kb.key_down(3, 0); // press 1 (row 3 bit 0)
    kb.key_up(3, 0);   // release it
    let result = kb.read_port(0xF7); // A11 low → row 3
    assert_eq!(result, 0xFF, "key release should restore bit to 1");
}

// ── Multiple keys in the same row → both bits clear ────────────────────────

#[test]
fn multiple_keys_same_row() {
    let mut kb = Keyboard::new();
    kb.key_down(1, 0); // A
    kb.key_down(1, 2); // D
    let result = kb.read_port(0xFD); // A9 low → row 1
    // bits 0 and 2 should be clear; result & 0x1F = 0b11010 = 0x1A
    assert_eq!(result & 0x1F, 0x1A, "bits 0 and 2 should both be clear");
}

// ── Multiple address lines active → rows ANDed ─────────────────────────────

#[test]
fn multiple_rows_selected_are_anded() {
    let mut kb = Keyboard::new();
    // Press SHIFT (row 0 bit 0) and SPACE (row 7 bit 0)
    kb.key_down(0, 0);
    kb.key_down(7, 0);

    // Select rows 0 AND 7 simultaneously (high byte 0x7E = all bits set except 0 and 7)
    let select = 0x7Eu8; // bit 0 = 0, bit 7 = 0 → rows 0 and 7 selected
    let result = kb.read_port(select);
    // Both rows have bit 0 clear → result bit 0 = 0
    assert_eq!(result & 0x01, 0, "bit 0 should be clear (pressed in both rows)");
    // Bits 1–4 not pressed in either → should be 1
    assert_eq!(result & 0x1E, 0x1E, "bits 1–4 should be 1");
}

#[test]
fn multiple_rows_only_common_bits_cleared() {
    let mut kb = Keyboard::new();
    // Press bit 1 only in row 0, bit 2 only in row 2
    kb.key_down(0, 1); // Z
    kb.key_down(2, 2); // E

    // Select rows 0 and 2 (high byte 0xFA = bits 0,2 clear)
    let select = 0xFAu8; // ~0x05 & 0xFF
    let result = kb.read_port(select);
    // Row 0 contributes: bit 1 clear
    // Row 2 contributes: bit 2 clear
    // AND: bits 1 and 2 both clear
    assert_eq!(result & 0x02, 0, "bit 1 should be clear (from row 0)");
    assert_eq!(result & 0x04, 0, "bit 2 should be clear (from row 2)");
    assert_eq!(result & 0x19, 0x19, "bits 0,3,4 should be 1");
}

// ── Kempston joystick ──────────────────────────────────────────────────────

#[test]
fn kempston_default_is_zero() {
    let kb = Keyboard::new();
    assert_eq!(kb.read_kempston(), 0x00, "Kempston should be 0 with no inputs");
}

#[test]
fn kempston_bits_set_independently() {
    let mut kb = Keyboard::new();
    kb.kempston_down(0); // right
    assert_eq!(kb.read_kempston(), 0x01, "bit 0 = right");

    kb.kempston_down(3); // up
    assert_eq!(kb.read_kempston(), 0x09, "bits 0 and 3 = right + up");

    kb.kempston_up(0); // release right
    assert_eq!(kb.read_kempston(), 0x08, "only up remains");
}

#[test]
fn kempston_fire_bit() {
    let mut kb = Keyboard::new();
    kb.kempston_down(4); // fire
    assert_eq!(kb.read_kempston() & 0x10, 0x10, "bit 4 = fire");
    kb.kempston_up(4);
    assert_eq!(kb.read_kempston() & 0x10, 0x00, "fire released");
}

// ── Reset clears all state ─────────────────────────────────────────────────

#[test]
fn reset_clears_all_keys_and_kempston() {
    let mut kb = Keyboard::new();
    kb.key_down(0, 0);
    kb.key_down(3, 2);
    kb.kempston_down(4);
    kb.reset();
    assert_eq!(kb.read_port(0x00), 0xFF, "all keys should be released after reset");
    assert_eq!(kb.read_kempston(), 0x00, "Kempston should be 0 after reset");
}

// ── Bits 5–7 always 1 ─────────────────────────────────────────────────────

#[test]
fn high_bits_always_one() {
    let mut kb = Keyboard::new();
    // Press all keys in all rows
    for row in 0..8 {
        for bit in 0..5 {
            kb.key_down(row, bit);
        }
    }
    // With all keys pressed, read any row — bits 5–7 must still be 1
    let result = kb.read_port(0x00);
    assert_eq!(result & 0xE0, 0xE0, "bits 5–7 must always be 1 even when all keys pressed");
}
