; scr_loader.asm — load SCR data into ZX Spectrum screen memory
;
; SCR data (6912 bytes) is appended immediately after the assembled code.
; The loader copies it into video RAM and halts.
;
; Memory layout:
;   0x4000–0x57FF  pixel bitmap  (6144 bytes, 192 rows x 32 columns)
;   0x5800–0x5AFF  colour attributes (768 bytes, 24 rows x 32 columns)
;
; Usage:
;   1. Assemble this file with ORG 0x8000
;   2. Append 6912 bytes of SCR data immediately after the binary
;   3. Load the combined binary and execute from 0x8000

    org 0x8000

    ld hl, scr_data   ; source: SCR data appended after code
    ld de, 0x4000     ; dest: screen memory start
    ld bc, 6912       ; length: full screen (bitmap + attributes)
    ldir              ; block copy HL->DE, BC bytes

    halt              ; stop — screen now shows the image

scr_data:
; 6912 bytes of SCR data appended here by build_scr.py
