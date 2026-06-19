# v2 feature stack: bf4000 + median/MAD + Δ/ΔΔ + musical features

Per-window vector is 173 dims:
- 60 MFCC (static + Δ + ΔΔ), each {median, MAD} -> 120
-  4 scalar spectral (centroid, rolloff, bandwidth, ZCR), each {med,MAD} -> 8
- 12-dim chroma_stft, each {med, MAD} -> 24
-  7-output spectral_contrast, each {med, MAD} -> 14
- 1-dim spectral_flatness {med, MAD} -> 2
- 1-dim RMS {med, MAD} -> 2
- 1-dim crest factor {med, MAD} -> 2
- onset rate (events / sec) -> 1

Prefilter (nominal frequencies only, no noisy-side measurements): HP 25 Hz, mains notches [50.0, 100.0, 150.0, 200.0] Hz, LP 4000 Hz.

Same SVC(rbf, C=10), same StratifiedGroupKFold (n=5, seed=42), same per-track soft-vote aggregation as the baseline.

## Results (per-track accuracy, mean +/- std across 5 folds)

| experiment           | accuracy            | wrong/250 | jazz | metal | pop |
|----------------------|---------------------|-----------|------|-------|-----|
| A_v2 clean -> clean  | **0.964 +/- 0.015** | 9        | 0.99 | 0.96 | 0.94 |
| B_v2 noisy -> noisy  | **0.916 +/- 0.015** | 21        | 0.92 | 0.94 | 0.89 |
| H_v2 clean -> 7th    | **0.796 +/- 0.054** | 51        | 0.85 | 0.78 | 0.76 |
| **M_v2 noisy -> 7th**| **0.332 +/- 0.010** | 167        | 0.80 | 0.00 | 0.19 |

## Placement against frozen baseline

- A clean -> clean              : 0.988  (upper bound, plain features)
- B noisy -> noisy              : 0.912  (noise-aware, plain features)
- C clean -> noisy              : 0.344  (untreated floor)
- H clean -> 7th (plain feats)  : 0.560
- H_bf4000 clean -> 7th (band-focused features) : 0.660
- **H_v2 clean -> 7th  : 0.796**  (rec 70.2%, delta vs H_bf4000 = +0.136)
- **M_v2 noisy -> 7th  : 0.332**  (rec -1.9%, delta vs B = -0.580)

## Leakage status

Same StratifiedGroupKFold splits as A/B/C/D/E/F/G/H/H_pf/H_bf4000. Train and test tracks are disjoint per fold for every experiment. Audit: `verify_no_leakage_v2.py`.
