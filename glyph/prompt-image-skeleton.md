# Glyph Image Prompt Skeleton

Template for generating art via image LLMs (Midjourney, DALL-E, Flux, etc.)
that renders well on both WASM (full colour) and Z80 (Spectrum constraints).

---

## Base prompt structure

```
[subject description]

Style: [art direction]

Technical constraints:
- High contrast between foreground and background
- Bold, distinct shapes — avoid fine gradients or subtle textures
- Key text rendered large, high contrast, on solid background
- Important elements use saturated primary colours (red, green, cyan, yellow, blue, magenta, white on black)
- Background areas are solid dark/black where possible
- No important detail smaller than 8x8 pixels at final resolution
- QR codes or data elements: large modules, black on white, positioned in corners

Composition:
- [aspect ratio — 4:3 for direct Spectrum fit, or portrait with letterbox intent]
- [text placement — bottom strip on solid black, or top-left badge]
- [focal point — centre, bold silhouette]
```

---

## Colour guidance for prompts

These map to the ZX Spectrum's 15-colour palette. Art using these hues
converts cleanly. Other colours get snapped to the nearest match.

| Prompt term | Spectrum colour | Hex (bright) |
|-------------|----------------|--------------|
| "black background" | Black | #000000 |
| "bright red", "crimson" | Bright Red | #FF0000 |
| "bright blue", "electric blue" | Bright Blue | #0000FF |
| "bright green", "neon green" | Bright Green | #00FF00 |
| "bright cyan", "teal glow" | Bright Cyan | #00FFFF |
| "bright magenta", "hot pink" | Bright Magenta | #FF00FF |
| "bright yellow", "gold" | Bright Yellow | #FFFF00 |
| "bright white", "pure white" | Bright White | #FFFFFF |

Normal-intensity variants exist too (0xCD instead of 0xFF) but image LLMs
don't distinguish — the quantiser maps to the nearest palette entry.

**Colours that cause problems:**
- Orange → snaps to red or yellow (no orange in Spectrum palette)
- Brown → snaps to red or black
- Purple → snaps to magenta or blue
- Skin tones → snap to red, yellow, or white depending on shade
- Pastel anything → muddy result, avoid

---

## Example prompts

### Game loading screen (portrait, letterbox)

```
A steampunk warrior in powered armour, dynamic action pose, wielding
a glowing cyan energy weapon. Mechanical wings spread wide. Industrial
cityscape background.

Style: detailed digital painting, bold colours, high contrast

Technical constraints:
- Black background in sky and edges
- Armour uses white/silver and cyan highlights
- Energy weapon is bright cyan (#00FFFF) on black
- Text "AERYTHEN" on chest plate in white on dark armour
- QR code bottom-right corner: large modules, black on white, 
  minimum 4px per module
- URL text bottom strip: white on solid black, large sans-serif

Composition: portrait 2:3, key figure centred, text/QR in lower quarter
on solid black strip
```

### Game title screen (landscape, direct 4:3)

```
A retro space battle scene. Two ships facing off against a starfield.
Title "VOID DUEL" in large block letters across the top.

Style: pixel-art inspired, bold flat colours, no gradients

Technical constraints:
- Starfield: pure black with scattered bright white dots
- Ships: bright cyan and bright red, solid filled, no shading
- Title text: bright yellow on black, block font, each letter at
  least 16px tall
- Explosions: bright yellow and bright red only
- No fine detail under 8px — shapes are chunky and readable

Composition: landscape 4:3 (256x192 native), title top third,
ships middle, starfield fills background
```

### Card art (portrait, designed for both)

```
[character description]

Style: bold graphic illustration, limited palette, strong silhouettes

Technical constraints:
- Maximum 6 dominant colours from: black, white, red, cyan, yellow,
  green, blue, magenta
- Background is solid black or single dark colour
- Character outlined with bright white or cyan for Z80 readability
- Any text is minimum 16px height, white on black strip
- Card border is single bright colour on black (1 cell wide = 8px)
- Face/hands use bright white + one accent colour (not skin-tone 
  gradients)

Composition: portrait, full bleed to edges, character fills 70% of frame
```

---

## Post-generation checklist

After generating the image, before running through glyph:

1. **Check text readability** — zoom to 256×192. Can you read it? If not, make it bigger or higher contrast.
2. **Check colour count** — squint at the image. More than 6-7 dominant colours will fight on Z80.
3. **Check 8×8 cell boundaries** — mentally overlay an 8px grid. Do important edges land on grid lines? They'll be sharper.
4. **Check QR/data elements** — are they big enough? Each module needs to survive 8×8 quantisation.
5. **Run the converter** — `python -m glyph.scr convert image.png --output test.scr` then `python -m glyph.scr compare image.png test.scr --output check.png`
