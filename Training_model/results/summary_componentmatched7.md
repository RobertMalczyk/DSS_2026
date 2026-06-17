# Cross-condition: train CLEAN -> test COMPONENTMATCHED_7TH_STAGE

Same windowing, features, classifier, folds, and aggregation as the
rest of the baseline. Only the test-source audio changes.

Test files come from:
    C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered ComponentMatched 7th stage
Filename pattern: `filtered_componentmatched_7th_stage_<file_id>.wav`.

## Result (per-track accuracy, mean +/- std across 5 folds)

| experiment                                              | accuracy            | wrong / 250 | jazz | metal | pop |
|---------------------------------------------------------|---------------------|-----------------------|------|-------|-----|
| H: train clean, test componentmatched_7th_stage         | **0.560 +/- 0.036** | 110                   | 0.90 | 0.00 | 0.77 |

Window-level accuracy: 0.485 +/- 0.019

Confusion matrix: `confusion_clean_to_componentmatched7.png`.

## Placement against frozen baseline

- A clean -> clean                   : 0.988  (upper bound)
- B noisy -> noisy                   : 0.912  (noise-aware training)
- C clean -> noisy                   : 0.344  (untreated domain-shift floor)
- D clean -> filtered_AI             : 0.404  (AI denoiser at test)
- E clean -> filtered_1st            : 0.340  (classical 1st-stage denoiser at test)
- F clean -> filtered_2nd            : 0.436  (classical+AI 2nd-stage denoiser at test)
- G clean -> adaptivems_6th          : 0.492  (adaptive-MS 6th-stage denoiser at test)
- **H clean -> componentmatched_7th  : 0.560**  (component-matched 7th-stage denoiser at test)

Recovery fraction of the A-C gap closed by the ComponentMatched 7th-stage filter:
`(H - C) / (A - C) = (0.560 - 0.344) / (0.988 - 0.344) = 33.54%`

## Leakage status

For each of the 5 folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test
windows come from the 7th-stage filter output, but the underlying track
IDs never overlap. Full audit: `verify_no_leakage_componentmatched7.py`.
