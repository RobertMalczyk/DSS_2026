# Cross-condition (Stage-7 audio) with band-pass-focused features

Same SVM, folds, and aggregation as the rest of the baseline.
Both train (clean) and test (Stage-7 output) are run through the
same matched band-pass prefilter (HP 25 Hz, mains notches [50.0, 100.0, 150.0, 200.0] Hz, optional switching notch at 4410 Hz, LP at the band edge) BEFORE feature extraction.

## Results

| variant | LP (Hz) | A_bf (clean->clean) | H_bf (clean->7th) | wrong/250 | jazz | metal | pop |
|---------|---------|---------------------|-------------------|-----------|------|-------|-----|
| compmatch_bf4000 | 4000 | 0.948 +/- 0.016 | **0.660 +/- 0.078** | 85 | 0.92 | 0.53 | 0.53 |
| compmatch_bf5500 | 5500 | 0.952 +/- 0.016 | **0.632 +/- 0.118** | 92 | 0.98 | 0.54 | 0.37 |

## Placement against frozen baseline

- A clean -> clean                          : 0.988  (upper bound)
- C clean -> noisy                          : 0.344  (untreated floor)
- G clean -> adaptivems_6th                 : 0.492
- H clean -> componentmatched_7th           : 0.560  (no feature change)
- H_pf clean_pf -> componentmatched_7th_pf  : 0.572  (linear prefilter only)
- **H_bf4000 clean -> 7th (LP 4000 Hz) : 0.660**  (recovery 49.1%, delta vs H = +0.100)
- **H_bf5500 clean -> 7th (LP 5500 Hz) : 0.632**  (recovery 44.7%, delta vs H = +0.072)

## Leakage status

Same StratifiedGroupKFold splits as A/B/C/D/E/F/G/H/H_pf.
Audit: `verify_no_leakage_componentmatched7_bandfocus.py`.
