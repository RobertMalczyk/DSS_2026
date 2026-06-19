# Feature stacks used in the music-genre classifier

Two feature stacks have been used in this project — the **baseline (v1, 46 dims)** in `train_compare.py:79`, and the **v2 stack (173 dims)** in `train_cross_v2_noisy_to_7th.py:112`. Both are computed per 3 s window (50 % overlap, 22050 Hz, `n_fft=2048`, `hop=512` → ~129 STFT frames per window) and then aggregated to one fixed-length vector per window.

## Baseline v1 — 46 dims (`features_for_window`)

Per-frame quantities, then **mean + std** over frames:

| group | per-frame dim | aggregated |
|---|---|---|
| MFCC (`n_mfcc=20`) | 20 | 20 mean + 20 std = 40 |
| spectral centroid | 1 | 2 |
| spectral rolloff | 1 | 2 |
| spectral bandwidth | 1 | 2 |
| zero-crossing rate | 1 | 2 |

**Total: 46.** Cheap, fully classical, what every textbook MFCC genre classifier uses.

## v2 stack — 173 dims (`features_v2`)

Three extensions over v1: more cepstral information, more harmonic/timbral descriptors, and a robust aggregator.

**Per-frame matrices (each → median + 1.4826·MAD over frames):**

| group | per-frame dim | aggregated |
|---|---|---|
| static MFCC | 20 | 40 |
| Δ-MFCC (1st diff) | 20 | 40 |
| ΔΔ-MFCC (2nd diff) | 20 | 40 |
| spectral centroid, rolloff, bandwidth, ZCR | 4×1 | 8 |
| chroma_stft | 12 | 24 |
| spectral contrast (default 6 bands → 7 outputs) | 7 | 14 |
| spectral flatness | 1 | 2 |
| RMS (manual framing) | 1 | 2 |
| crest factor (peak/RMS, same framing) | 1 | 2 |

**Plus one whole-window scalar:** `onset_rate` = number of detected onsets / window length → 1 dim.

**Total: 120 + 8 + 24 + 14 + 2 + 2 + 2 + 1 = 173.**

### Why each addition

- **Δ / ΔΔ-MFCC** — captures *temporal* timbre evolution (attack shape, modulation), which static MFCC misses. Helps separate sustained jazz horns from percussive metal hits.
- **chroma_stft (12)** — pitch-class energy; harmonic vs atonal content. Strong jazz/pop discriminator.
- **spectral contrast (7)** — peak-to-valley ratio per sub-band; correlates with how "tonal vs noisy" each band is. Good metal detector.
- **spectral flatness** — geometric/arithmetic mean ratio; pure tone (0) ↔ white noise (1).
- **RMS + crest factor** — loudness and dynamic-range compression signature (pop is squashed, jazz is dynamic).
- **onset rate** — events per second; rhythmic density.
- **median + 1.4826·MAD instead of mean + std** — robust to impulsive bursts and 8-bit quantisation spikes from the noise model. The MAD scaling makes it a consistent estimator of σ under normality, so v2 numbers stay roughly comparable to v1 in scale.

The v2 stack is what produced **H_v2 = 0.796** (clean-trained → tested on Stage-7 ComponentMatched output), recovering 70 % of the C-cliff drop versus the baseline H = 0.560.
