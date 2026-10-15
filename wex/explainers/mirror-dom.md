# Mirror DOM — making canvas games accessible

## The problem

`<canvas>` is a black box to screen readers. It renders pixels — no text, no structure, no semantics. A blind user tabbing through a page skips the canvas entirely. Everything happening inside the game is invisible to assistive tech.

## The solution: a hidden DOM overlay

Create a parallel HTML structure that sits on top of the canvas. Each game object gets a `<div>` with:

- `role` — what it is (`img`, `region`, `button`)
- `aria-label` — what to announce ("Castle towers, upper portion")
- Position — matches the object's pixel coordinates (as percentages, so it scales)

The divs use the sr-only pattern (visually hidden via clip-rect, still accessible to screen readers). Sighted users see the canvas. Screen reader users traverse the DOM tree.

## How it works in our pipeline

```
scene.json → pipeline → WAT data segment → debug.wasm
                                                 ↓
                                    player reads a11y segment
                                                 ↓
                                    buildMirrorDom(canvas, nodes)
                                                 ↓
                                    hidden positioned divs over canvas
```

1. **Author** writes `scene.json` — array of `{role, label, x, y, w, h}` nodes
2. **Pipeline** encodes it as UTF-8 JSON in a WAT data segment, sets bit 3 in `type()`
3. **Player** detects bit 3, reads bytes via `a11y_offset()`/`a11y_size()`, decodes JSON
4. **`buildMirrorDom()`** creates the overlay divs, positioned as percentages of canvas size

## The sr-only pattern

```css
.sr-only {
  position: absolute;
  width: 1px; height: 1px;
  padding: 0; margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
```

Visually invisible. Screen readers still find it. This is the standard WCAG pattern — used by Bootstrap, Tailwind, and every major framework.

## Live regions

For dynamic content (score changes, state transitions), we use `aria-live="polite"`:

```html
<div aria-live="polite" class="sr-only" id="announcer"></div>
```

Setting its `textContent` causes screen readers to announce the change. The "polite" setting means it waits for the reader to finish its current announcement.

## Coordinates as percentages

Game objects have pixel coordinates (`x: 200, y: 100, w: 624, h: 500` on a 1024×1536 canvas). The overlay divs use percentage positioning:

```
left:   200/1024 * 100 = 19.5%
top:    100/1536 * 100 =  6.5%
width:  624/1024 * 100 = 60.9%
height: 500/1536 * 100 = 32.6%
```

This means the overlay scales with the responsive canvas — no resize handlers needed.

## Preference imports

The WASM module can also import browser preferences:

```wat
(import "env" "prefers_dark" (func (result i32)))
(import "env" "prefers_reduced_motion" (func (result i32)))
(import "env" "prefers_high_contrast" (func (result i32)))
```

The player provides these via `matchMedia`. Games that care can query them to adapt rendering. A settings panel can override the browser defaults.

## Connection to toolkit/flare

This is a slimmed-down version of the mirror DOM in `toolkit/flare`:

| Concept | flare | WASM player |
|---------|-------|-------------|
| Semantic model | `SemNode` struct (Rust) | JSON array in WASM memory |
| Contract | `Describable` trait | `a11y_offset()`/`a11y_size()` exports |
| DOM builder | `buildMirrorDom()` (JS) | `buildMirrorDom()` (JS, simpler) |
| Announcements | `announce()` + live region | Same pattern |
| Reduced motion | `prefers_reduced_motion()` in Rust | WASM import from host |

The flare version has chart-specific features (data tables, bar rect extraction, drill-down). Ours is game-focused — regions and labels, no tabular data.

## In practice

This pattern maps directly to real accessibility work:

- **Data dashboards** — same problem. Canvas charts need mirror DOMs. The toolkit's flare does this for bar charts; the pattern extends to any visualisation.
- **Sonification** — translating visual data to audio. Our beeper engine already does this. In business contexts: audio sparklines, trend alerts, data exploration for blind analysts.
- **Preference bridging** — browser settings flowing into application rendering. Every web app needs this; games make it concrete because the rendering is entirely custom.
