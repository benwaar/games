# SCALING.md — WASM player responsive scaling

Research notes for making `wex/index.html` work across screen sizes, mobile, and HiDPI.

## Core principle

The canvas backing store stays at **content resolution** — whatever the WASM exports via `width()` and `height()`. We don't re-render at screen resolution. CSS handles display scaling.

This is the opposite of how you'd handle a vector-drawing canvas (where you'd multiply by `devicePixelRatio` and `ctx.scale()`). Our content is a fixed pixel grid — scaling it in CSS with `image-rendering: pixelated` gives nearest-neighbour upscale, which is correct for pixel art and game content.

## `image-rendering: pixelated`

- Tells the browser to use nearest-neighbour interpolation when scaling the canvas
- Keeps hard pixel edges — no bilinear smoothing
- Minor caveat: non-integer `devicePixelRatio` (e.g. 1.5× on some Android) can produce uneven pixel sizes. Acceptable for our use case — the alternative (rendering at screen resolution) would blur the pixel grid
- Supported in all modern browsers

Source: [MDN — Crisp pixel art look](https://developer.mozilla.org/en-US/docs/Games/Techniques/Crisp_pixel_art_look)

## Responsive CSS pattern

```css
canvas {
  max-width: calc(100vw - 24px);   /* viewport width minus padding */
  max-height: 80vh;                 /* leave room for status + controls */
  image-rendering: pixelated;
  border: 2px solid #333;
}
```

JS sets `canvas.style.aspectRatio` after WASM loads (e.g. `1024 / 1536`). The browser then scales the canvas to the largest size that fits both constraints while preserving aspect ratio.

Body uses `display: flex; align-items: center; justify-content: center` — this letterboxes automatically.

## HiDPI / Retina

For our use case: **do nothing special.**

- Canvas `.width` / `.height` = content pixel dimensions (set by WASM)
- CSS scales the canvas to display size
- `image-rendering: pixelated` ensures nearest-neighbour
- No `ctx.scale(dpr, dpr)` — that's for vector/drawing canvases where you want crisp lines at retina resolution

If we later add a vector overlay (UI, HUD), that would need a separate canvas with DPR scaling.

Source: [Responsive Canvas Rendering (ates.dev, Dec 2024)](https://ates.dev/posts/2024-12-08-responsive-canvas/)

## Mobile / orientation

- `<meta name="viewport" content="width=device-width, initial-scale=1.0">` already present
- Portrait: canvas scales to fit width, letterboxed vertically
- Landscape: canvas scales to fit height, letterboxed horizontally
- No JS resize handlers needed — CSS max-width/max-height + aspect-ratio handles both
- Touch: click-to-start overlay for audio already works as touch target

## Fullscreen (deferred)

`canvas.requestFullscreen()` on double-click or dedicated button. Browser handles scaling in fullscreen. Not needed for the POC — noted as a future option.

## Testing checklist

Verify with these viewports (Chrome DevTools device emulation):

| Device | Viewport | Notes |
|--------|----------|-------|
| Desktop | 1440×900 | baseline |
| Desktop HD | 2560×1440 | large screen |
| iPhone SE | 375×667 | small phone, portrait |
| iPhone SE | 667×375 | small phone, landscape |
| iPhone 14 Pro | 393×852 | modern phone, portrait |
| iPhone 14 Pro | 852×393 | modern phone, landscape |
| iPad | 768×1024 | tablet, portrait |
| iPad | 1024×768 | tablet, landscape |
| Galaxy S21 | 360×800 | Android, portrait |

Test with: tall portrait image (1024×1536), square (256×256), audio-only.
