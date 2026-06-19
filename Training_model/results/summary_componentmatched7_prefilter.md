# Cross-condition: train CLEAN_pf -> test COMPONENTMATCHED_7TH_pf

Same windowing, classifier, folds, and aggregation as the rest of the
baseline. Only the **feature extractor** changes: each waveform is run
through a fixed linear prefilter that mirrors the linear stages of the
Stage-7 denoiser, then the standard 48-dim MFCC + spectral features are
extracted from the prefiltered waveform.

Prefilter (nominal design frequencies, no noisy-side detection):
- Zero-phase Butterworth HP, cutoff 25 Hz, order 2
- IIR notches at the mains comb [50.0, 100.0, 150.0, 200.0] Hz, Q=35.0
- IIR notch at the switching tone 4410 Hz, Q=220.0

(Stage 7's adaptive Wiener is not replayed; on already-clean audio its
gain is near unity in high-SNR bins.)

## Results (per-track accuracy, mean +/- std across 5 folds)

| experiment                                                | accuracy            | wrong / 250 | jazz | metal | pop |
|-----------------------------------------------------------|---------------------|-----------------------|------|-------|-----|
| A_pf : train clean_pf, test clean_pf (sanity)             | **0.992 +/- 0.016** | 2                   | 0.99 | 0.99 | 1.00 |
| H_pf : train clean_pf, test componentmatched_7th_pf       | **0.572 +/- 0.043** | 107                   | 0.99 | 0.00 | 0.72 |

Window-level accuracy (H_pf): 0.493 +/- 0.031
Confusion matrix: `confusion_clean_to_componentmatched7_prefilter.png`.

## Placement against frozen baseline

- A clean -> clean                       : 0.988  (upper bound)
- C clean -> noisy                       : 0.344  (untreated floor)
- G clean -> adaptivems_6th              : 0.492
- H clean -> componentmatched_7th        : 0.560  (no feature change)
- **H_pf clean_pf -> componentmatched_7th_pf : 0.572**  (Stage-7-matched features)

Recovery fraction of the A-C gap closed by Stage-7 + matched features:
`(H_pf - C) / (A - C) = (0.572 - 0.344) / (0.988 - 0.344) = 35.40%`

Delta vs H (same Stage-7 audio, plain features): `H_pf - H = +0.012`.

## Leakage status

For each fold, train_tracks and test_tracks are disjoint (asserted in-loop).
Train windows come from clean/ (prefiltered), test windows come from the
7th-stage filter output (prefiltered), but the underlying track IDs never
overlap. Full audit: `verify_no_leakage_componentmatched7_prefilter.py`.
