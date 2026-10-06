# Artefact

Digital archaeology for early computing.

**dig** finds hidden art buried in binary memory dumps.
**bloom** takes that art and scales it to HD.

Together they form a pipeline: scan an old game file, extract the sprites nobody documented, make them usable.

---

## The problem

Old game binaries are dense. Sprites, fonts, loading screens, tile sets — all packed into 48K of RAM with no labels, no headers, no index. BinViewer and tools like it let you find them manually: scroll through rendered byte windows until something looks like art. It works. It takes hours.

**dig** automates that visual judgement using a trained classifier.
**bloom** takes whatever dig finds and scales it from 8×8 pixels to something you can actually use.

---

## Projects

| Project | What it does | Key techniques |
|---------|-------------|----------------|
| [dig](dig/) | Scan a binary, score every byte window — art or noise | CNN classifier, ROC/AUC, sliding window, autoencoder |
| [bloom](bloom/) | Scale pixel art sprites to HD without blurring | Super-resolution CNN, perceptual loss, GAN |

---

## Pipeline

```
game.tap / game.z80
     ↓
  dig scan
     ↓
ranked windows → sprite_001.png, sprite_002.png ...
     ↓
  bloom scale
     ↓
sprite_001_hd.png, sprite_002_hd.png ...
```

---

## Status

- dig: planned
- bloom: planned

See each project's PLAN.md for milestones and approach.
