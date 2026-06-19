# Cross-condition under 3 kHz feature-bandlimit

Same pipeline as the frozen baseline, but every waveform (train and
test) is low-pass filtered at 3000 Hz (Butterworth order
8, zero-phase) **before** windowing and feature extraction.
This guarantees that no feature depends on spectral content above
3 kHz on either side of the train/test boundary.

## Results (per-track accuracy, mean +/- std across 5 folds)

| experiment                                             | accuracy            | wrong / 250 | jazz | metal | pop |
|--------------------------------------------------------|---------------------|-----------------------|------|-------|-----|
| E' : clean_3kHz -> filtered_1st_stage_3kHz             | **0.404 +/- 0.032** | 149                   | 0.11 | 0.92 | 0.19 |
| F' : clean_3kHz -> filtered_2nd_stage_3kHz             | **0.456 +/- 0.032** | 136                   | 0.25 | 0.28 | 0.84 |

Window-level accuracy:
- E' : 0.443 +/- 0.029
- F' : 0.507 +/- 0.018

## Comparison to full-bandwidth baseline

| pair                    | full-bw | 3 kHz  | delta  |
|-------------------------|--------:|-------:|-------:|
| E  vs  E'               |  0.340  | 0.404 | +0.064 |
| F  vs  F'               |  0.436  | 0.456 | +0.020 |

Recovery fraction of the full-bandwidth A-C gap closed under bandlimit:
- E' : (E' - C_fb) / (A_fb - C_fb) = 9.32%
- F' : (F' - C_fb) / (A_fb - C_fb) = 17.39%

(Anchors A_fb = 0.988, C_fb = 0.344 are the clean and domain-shift
points from the full-bandwidth frozen baseline, not re-run here.)

## Leakage status

Folds are byte-identical to the baseline StratifiedGroupKFold
(random_state=42, n_splits=5), so E' and F' are directly
comparable to the full-bandwidth experiments. Full audit:
`verify_no_leakage_fcutoff3khz.py`.

## Confusion matrices

- `confusion_fcutoff3khz_clean_to_filtered_1st.png`
- `confusion_fcutoff3khz_clean_to_filtered_2nd.png`
