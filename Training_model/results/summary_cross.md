# Cross-condition: train CLEAN -> test NOISY

Same windowing, features, classifier, folds, and aggregation as
`train_compare.py` (experiments A and B). Only the source of train vs test
windows differs.

## Result (per-track accuracy, mean +/- std across 5 folds)

| experiment                 | accuracy            | jazz | metal | pop |
|----------------------------|---------------------|------|-------|-----|
| C: train clean, test noisy | **0.344 +/- 0.015** | 0.23 | 0.00 | 0.81 |

Window-level accuracy: 0.367 +/- 0.034

Confusion matrix: `confusion_clean_to_noisy.png`.

## Leakage status

For each of the 5 folds, `train_tracks` and `test_tracks` are
disjoint (verified by the in-loop assert in `train_cross.py` and by
`verify_no_leakage.py` on the underlying fold structure). Although the
audio source differs between train and test, the track IDs do not overlap,
so no track is seen in both — this measures generalisation to the noise
distribution, not memorisation.
