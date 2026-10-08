# Chord Progressions as Sequence Modelling

A chord progression is not a bag of chords — it's an ordered sequence. Cmaj → Fmaj → Gmaj → Cmaj (I-IV-V-I) is a different thing from Gmaj → Fmaj → Cmaj → Cmaj (V-IV-I-I). The order carries musical meaning. This document explains how progressions map onto sequence classification problems, and why synthetic data makes training tractable.

---

## Why order matters

In a bag-of-words model, "the dog bit the man" and "the man bit the dog" are identical — same words, same counts. Chord progressions have the opposite property: the order is the signal.

A I-IV-V-I progression (Cmaj → Fmaj → Gmaj → Cmaj) is the most common cadence in Western music — it sounds resolved and complete. The same chords in a different order don't carry the same meaning:

| Progression | Roman numerals | Character |
|-------------|----------------|-----------|
| Cmaj → Fmaj → Gmaj → Cmaj | I-IV-V-I | Resolved cadence |
| Amin → Fmaj → Cmaj → Gmaj | vi-IV-I-V | Pop verse |
| Cmaj → Gmaj → Amin → Fmaj | I-V-vi-IV | "Axis" progression |
| Dmin → Gmaj → Cmaj → Cmaj | ii-V-I-I | Jazz cadence |

A model that classifies progressions must learn temporal order — which is exactly what an RNN is designed for.

---

## How synthetic data gives free labels

Recording real piano progressions and labelling them is slow and expensive. Synthesised progressions from the Iowa chord clips give exact labels for free.

The process:
1. Take the first 1.5s of each chord WAV (attack + early sustain — the most information-rich part)
2. Add 0.25s silence gaps between chords (simulates the natural pause between chord changes)
3. Concatenate four chord clips → one 6.75s progression clip
4. Label = the sequence of chord names: `"I-IV-V-I"`

Because we control the synthesis, the label is always exact. No labelling uncertainty, no misaligned boundaries.

**The limitation:** synthetic progressions sound different from real played progressions. A pianist playing I-IV-V-I lets notes ring through transitions, uses pedal sustain, and varies timing. The synthesised version has hard cuts at the 1.5s marks and uniform silence gaps. The model trained on synthetic data would need fine-tuning on real recordings before deployment — the same domain gap problem as Phase 3 (Iowa piano vs real piano).

---

## The dataset

Four progressions, four dynamic combinations each:

| Label | Chords | Clips |
|-------|--------|-------|
| I-IV-V-I | Cmaj → Fmaj → Gmaj → Cmaj | 4 |
| vi-IV-I-V | Amin → Fmaj → Cmaj → Gmaj | 4 |
| I-V-vi-IV | Cmaj → Gmaj → Amin → Fmaj | 4 |
| ii-V-I-I | Dmin → Gmaj → Cmaj → Cmaj | 4 |

16 source clips total. After the progression-specific pipeline (no augmentation — the clips are already long enough): 16 tensors of shape `(1, 128, 291)`.

**Why 16 clips is a ceiling, not a floor:** The CNN-RNN architecture is proven at 66.7% accuracy on 4 classes. With more chord types, more progressions, and augmentation, accuracy would improve significantly. The data is the constraint.

---

## What the model learns

The CNN extracts per-frame chord features — it already knows what Cmaj and Fmaj look like from Phase 4. The GRU learns the *temporal pattern* of those features: which chord follows which, and in what order.

From the spectrogram's perspective, the progression is a sequence of ~9 time steps per chord (6.75s ÷ 4 chords × 36 GRU steps ÷ 6.75s ≈ 9 steps per chord). The GRU hidden state carries information from earlier chords as it processes later ones — by the end of the sequence, it has a compressed summary of the whole progression.

---

## What per-step decoding would add

The current model classifies the whole clip as one label. Per-step decoding would instead produce a label at each time step:

```
time step:  1  2  3  4  5  6  7  8  9  10 11 12 ...
label:      C  C  C  F  F  F  G  G  G  C  C  C
```

This is useful for two things:
- Detecting *when* a chord change happens (chord boundary detection)
- Handling progressions of variable length (not all progressions are 4 chords)

For Phase 4 the whole-clip classification was the right starting point — it teaches the sequence modelling concepts without the additional complexity of CTC loss or sequence-to-sequence labelling.

---

## The piano teacher connection

The full tutor pipeline now handles three levels:

| Question | Model | Command |
|----------|-------|---------|
| What note is this? | Phase 3 note classifier | `bash demo_note.sh` |
| What chord is this? | Phase 4 two-pass pipeline | `bash demo_chord.sh` |
| What progression did I play? | Phase 4b CNN-RNN | `python -m model.progression_train` |

The progression classifier closes the loop: a student can play a full I-IV-V-I and the model classifies it — compare against the expected progression, flag which bar was wrong.

> **In practice:** Progression classification is sequence classification at the musical level — exactly the same problem as intent detection over a multi-turn conversation, or fraud detection over a sequence of transactions. The chord is the event; the progression is the journey.

See: [scripts/synthesise_progressions.py](../scripts/synthesise_progressions.py) — how progression clips are built  
See: [model/progression.py](../model/progression.py) — the CNN-RNN architecture  
See: [explainers/rnn-sequence-modelling.md](rnn-sequence-modelling.md) — how the GRU works
