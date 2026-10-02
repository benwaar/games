# SNR — Signal-to-Noise Ratio

SNR measures how much louder the signal is compared to the noise, in decibels.

| SNR (dB) | What it sounds like |
|----------|---------------------|
| 40+ | Near-silent background — studio quality |
| 20 | Noticeable hiss but signal is clear |
| 10 | Noisy — like a phone call in a busy room |
| 5 | Very noisy — signal barely audible |
| 0 | Signal and noise equally loud |

## How we use it

`add_noise(signal, snr_db=20)` computes the signal's power, then generates white noise scaled so the ratio matches the target SNR:

```
noise_power = signal_power / 10^(snr_db / 10)
```

Higher `snr_db` → less noise added. For training augmentation, we typically sample `snr_db` from a range (e.g. 10–30) to expose the model to varied conditions.

## Why it matters

A model trained only on SNR=40 (clean) audio will fail on SNR=10 (noisy) real-world input. By augmenting across a range of SNRs, the model learns to extract signal regardless of noise level.
