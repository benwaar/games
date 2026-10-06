# Acoustic Odyssey

Production phases for Signal Hunt — taking a trained model into the real world.

Signal Hunt builds and trains the model. Acoustic Odyssey deploys it and makes it adaptive.

---

## Projects

| Project | What it does | Key techniques |
|---------|-------------|----------------|
| [cast](cast/) | Export the trained Signal Hunt model to lightweight deployment formats, apply quantisation, benchmark for on-device inference | ONNX, TFLite, ExecuTorch, INT8 quantisation, latency benchmarking |
| [echo](echo/) | Add an RL agent on top of the classifier that adapts task difficulty based on user performance over time | MDP, reinforcement learning, user modelling, exploration vs exploitation |

---

## How it fits

```
Signal Hunt (Phase 1–4)
  → trained SoundClassifier checkpoint
       ↓
     cast
  → lightweight model running on-device, sub-100ms, no cloud
       ↓
     echo
  → model + RL agent, difficulty adapts to the user in real time
```

---

## Why this order

**cast first:** the model must run efficiently on-device before you can adapt it in real time.
A cloud-latency model can't respond fast enough for an interactive adaptive system.

**echo second:** once cast proves on-device inference is fast enough, echo layers the RL agent
on top — treating user accuracy as the reward signal that drives adaptation.

---

## Status

- cast: planned
- echo: planned
