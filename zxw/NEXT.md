# Next: M6 — Jetpac bring-up (incremental opcode completion)

Wire up Jetpac and implement enough opcodes to get it running.

## What to do

1. **Load Jetpac** — `load_rom(48.rom)` + `load_tap(jetpac_fixed.tap)` → screen at 0x4000,
   code at 0x6000, entry at 0x61E5.

2. **Unimplemented-opcode trap** — when the CPU hits an unknown DD/FD prefix or unknown opcode,
   log PC, opcode, and register context to the console, then halt. This gives a tight feedback
   loop for filling in opcodes incrementally.

3. **IX/IY prefix opcodes (DD/FD)** — indexed addressing `(IX+d)` / `(IY+d)`:
   - LD r,(IX+d) / LD (IX+d),r
   - ADD/SUB/AND/XOR/OR/CP (IX+d)
   - INC/DEC (IX+d)
   - LD IX,nn / LD (nn),IX / LD IX,(nn)
   - ADD IX,rr / PUSH/POP IX / JP (IX)
   - Same set for IY.

4. **Remaining ED-prefix gaps** as Jetpac hits them.

5. **IM 1 interrupt handling** — RST 0x38 every ~69 888 T-states (once per frame).
   The Spectrum ROM's IM 1 handler at 0x0038 updates the frame counter and runs the keyboard scan;
   Jetpac needs it to advance its title animation and respond to input.

6. **Title screen → gameplay** — verify:
   - Title screen renders (ULA).
   - Keypress starts game (keyboard via `key_down`).
   - Player moves left/right and fires.
   - Frame runs at ≈50fps with no visible slowdown.

## Gate

Jetpac loads → title screen renders → key press → game starts → player moves and collects rockets.
~50fps, no visible slowdown, no unhandled opcodes.

## Reference

- Jetpac entry point: 0x61E5 (after instant-load of 3-block TAP).
- IM 1 vector: 0x0038 (Spectrum ROM interrupt routine).
- DD/FD prefix docs: Z80 CPU User Manual, chapter 4 (indexed instructions).
