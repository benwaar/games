# Phase 3 — Piano Note Classification

## Why piano (not voice)

Originally planned to record 240+ voice clips (12 notes × 2 sound types × 10 each). In practice this was too much manual data collection — humans can't reliably hit the same pitch twice, so labelling becomes the bottleneck, not the model.

**Switched to piano:**
- **NSynth** (Google/Magenta) — free dataset of 300K+ instrument notes including clean acoustic grand piano at every pitch with exact MIDI labels. No recording needed, no labelling uncertainty.
- An electric piano is available for recording real-world test clips to close the domain gap at test time.
- The learning objectives (transfer learning, fine-grained classification, class imbalance) are identical.

## Longer-term goal

Phase 3 is the foundation for a piano teacher application:

```
Phase 3: "what single note is this?"
Phase 4+: "what chord is this?" (multi-label)
End goal: player plays a chord → detect which notes were played
          → compare against expected → "you hit Ab, should be A"
```

## Start here: M13 — Dataset preparation

1. Download NSynth train split (~300K notes, ~22GB) or use the small subset via TensorFlow Datasets
2. Filter to: `instrument_family = keyboard`, `instrument_source = acoustic`, MIDI notes 60–71 (C4–B4)
3. Organise into `data/raw/notes/{note}/` (e.g. `C4/`, `Cs4/`, `D4/` ...)
4. Run `python -m pipeline.batch data/raw/notes data/processed/notes`
5. Spot-check: render spectrograms — can you see the pitch differences visually?

**One thing to check before downloading:** NSynth notes are 4 seconds. The pipeline uses 1.5-second clips. Either truncate to 1.5s (captures attack + sustain, which is where pitch is clearest), or extend the pipeline.

## NSynth quick start

```python
import tensorflow_datasets as tfds
ds = tfds.load('nsynth/full-pitches', split='train')
# filter: instrument_family_str == 'keyboard', pitch in range(60, 72)
```

Or direct download: https://magenta.tensorflow.org/datasets/nsynth

## Gate

12 note classes (C4–B4), 10+ clips each, tensors generated, labels verified. Spectrogram spot-check shows visible pitch differences between notes.
