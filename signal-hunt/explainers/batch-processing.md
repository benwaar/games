# Batch Processing — From Folder to Training Set

Takes a folder of `.wav` files and produces a complete, labelled, augmented tensor dataset ready for a DataLoader. One command, deterministic output.

---

## What it does

```
data/raw/                          data/processed/
  hum_440hz.wav          →           hum_440hz_clean.pt
  whistle_chirp.wav                  hum_440hz_noise_low.pt
  clap_burst.wav                     hum_440hz_noise_high.pt
                                     hum_440hz_pitch_up.pt
                                     hum_440hz_pitch_down.pt
                                     hum_440hz_slow.pt
                                     hum_440hz_fast.pt
                                     whistle_chirp_clean.pt
                                     ... (7 per source file)
                                     manifest.json
```

Each source file produces **7 variants** — one clean pass plus six augmentations. 3 files become 21 training samples. More source files scale linearly.

---

## The augmentation strategy

| Variant | What it does | Why |
|---------|-------------|-----|
| `clean` | No augmentation — original signal | Baseline the model must get right |
| `noise_low` | White noise at SNR 20 dB | Mild mic hiss — common in laptop recordings |
| `noise_high` | White noise at SNR 10 dB | Noisy environment — phone in a busy room |
| `pitch_up` | Shift pitch up 2 semitones | Different voices, instruments |
| `pitch_down` | Shift pitch down 2 semitones | Same, other direction |
| `slow` | Time-stretch to 0.85× speed | Tempo variation |
| `fast` | Time-stretch to 1.15× speed | Tempo variation |

The augmentations are applied *before* feature extraction, so the Mel-spectrogram captures the augmented signal. Time-stretched outputs are padded/truncated back to the standard length to keep tensor shapes uniform.

> **Coming from C/JS/TS:** This is like generating test fixtures from a template — you start with a handful of canonical inputs and systematically produce variants that cover edge cases your system needs to handle.

---

## The manifest

`manifest.json` records what was produced and where it came from:

```json
[
  {
    "file": "hum_440hz_clean.pt",
    "source": "hum_440hz.wav",
    "label": "hum_440hz",
    "augmentation": "clean",
    "shape": [1, 128, 65]
  },
  ...
]
```

Labels come from the filename stem. This is fine for Phase 1 — in Phase 2 we'll likely switch to folder-based labels (`data/raw/hum/`, `data/raw/clap/`) or an explicit label map when the classification scheme is defined.

> **In practice:** The manifest is data provenance — the ML equivalent of an audit trail. In production pipelines (financial data, medical records, regulatory reporting), you always track which raw inputs produced which processed outputs, with what parameters. Without it, debugging a bad model prediction means guessing what data it trained on.

---

## Using the output with PyTorch

The tensors are uniform `(1, 128, 65)` and load directly:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load all tensors
manifest = json.loads(Path("data/processed/manifest.json").read_text())
tensors = [torch.load(f"data/processed/{r['file']}", weights_only=True) for r in manifest]

# Stack into a single tensor and create labels
X = torch.cat(tensors, dim=0)  # (21, 128, 65)
y = torch.tensor([label_map[r["label"]] for r in manifest])

# DataLoader for training
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

Phase 2 will build a proper `Dataset` class on top of this, but the raw tensors + manifest are already enough to start training.

---

## Determinism

Same seed → same output. The random number generator for noise augmentations is seeded explicitly (`--seed 42` by default). This means:

- Runs are reproducible across machines
- You can regenerate the exact same dataset from the same raw files
- Comparing two runs with different augmentation parameters is meaningful

> **In practice:** Reproducibility is a first-class concern in ML. If you can't regenerate your training data, you can't debug why a model behaves differently after retraining. Seeded pipelines, pinned dependency versions, and committed configs are how production ML teams avoid "works on my machine" disasters.

---

## Key parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `input_dir` | — | Folder of `.wav` source files |
| `output_dir` | — | Folder for `.pt` tensors + manifest |
| `--seed` | 42 | RNG seed for noise augmentations |

Run: `python -m pipeline.batch data/raw data/processed`
