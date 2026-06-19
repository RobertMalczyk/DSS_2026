# Genre classification: clean vs noisy

## Setup

- Dataset: 250 tracks (jazz=84, metal=83, pop=83) from
  `C:\Robak\DSS2026\Claude\Signal_generation\out\Model`.
- Each 30 s track sliced into 3 s windows with
  50 % overlap (hop=1.5 s) -> ~19
  windows per track, 4750 windows total.
- Features per window (48 dims): mean+std of 20 MFCCs +
  mean/std of spectral centroid, rolloff, bandwidth, and zero-crossing rate.
- Classifier: `StandardScaler -> SVC(rbf, C=10.0, gamma=scale,
  class_weight='balanced')`, fit on windows.
- Cross-validation: `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=42)` over tracks (groups = `track_id`), so no track
  appears in both train and test of the same fold. Both experiments use the
  **same** fold assignments to make the noise gap directly comparable.
- Per-track prediction is the **argmax of mean soft-probabilities** across
  that track's windows (soft-vote aggregation).

## Results (per-track accuracy, mean +/- std across folds)

| experiment       | accuracy            | jazz | metal | pop |
|------------------|---------------------|------|-------|-----|
| clean -> clean   | **0.988 +/- 0.016** | 0.99 | 0.99 | 0.99 |
| noisy -> noisy   | **0.912 +/- 0.045** | 0.90 | 0.95 | 0.88 |

- **Absolute accuracy drop from noise:** +0.076
- **Relative accuracy drop from noise:** +7.7 %

Per-genre accuracy drop (clean - noisy):
- jazz : +0.08
- metal: +0.04
- pop  : +0.11

The genre that suffered most under contamination: **pop**.

Window-level accuracy (informational, before per-track aggregation):
- clean -> clean: 0.966 +/- 0.014
- noisy -> noisy: 0.828 +/- 0.040

Confusion matrices (track-level, summed across folds): see
`confusion_clean.png` and `confusion_noisy.png`.

## Notes / assumptions

- 250 tracks is small; per-genre test counts (~16
  tracks per genre per fold) keep some jitter across seeds. The 5-fold CV at
  the track level bounds this but does not eliminate it.
- The noise model is fixed at SNR = -10 dB with mains hum, switching, broadband
  EMI, impulsive bursts, 1/f drift and 8-bit quantisation; results would
  differ at other SNRs.
- Soft-vote aggregation (mean of class probabilities) consistently equalled
  or beat hard window-vote in pilot runs, so it is used throughout.
