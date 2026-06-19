# Cross-condition: train CLEAN -> test FILTERED_2ND_STAGE

Same windowing, features, classifier, folds, and aggregation as the rest
of the baseline. Only the test-source audio changes.

Test files come from:
    C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered assist and AI 2nd stage

## Result (per-track accuracy, mean +/- std across 5 folds)

| experiment                                 | accuracy            | wrong / 250 | jazz | metal | pop |
|--------------------------------------------|---------------------|-----------------------|------|-------|-----|
| F: train clean, test filtered_2nd_stage    | **0.436 +/- 0.078** | 141                   | 0.57 | 0.33 | 0.41 |

Window-level accuracy: 0.414 +/- 0.064

Confusion matrix: `confusion_clean_to_filtered_2nd.png`.

## Placement against frozen baseline

- A clean -> clean              : 0.988  (upper bound)
- B noisy -> noisy              : 0.912  (noise-aware training)
- C clean -> noisy              : 0.344  (untreated domain-shift floor)
- D clean -> filtered_AI        : 0.404  (AI denoiser at test)
- E clean -> filtered_1st       : 0.340  (1st-stage classical denoiser at test)
- **F clean -> filtered_2nd     : 0.436**  (2nd-stage classical+AI denoiser at test)

Recovery fraction of the A-C gap closed by the 2nd-stage filter:
`(F - C) / (A - C) = (0.436 - 0.344) / (0.988 - 0.344) = 14.29%`

## Leakage status

For each of the 5 folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test windows
come from the 2nd-stage filter output, but the underlying track IDs
never overlap. Full audit: `verify_no_leakage_filtered_2nd.py`.
