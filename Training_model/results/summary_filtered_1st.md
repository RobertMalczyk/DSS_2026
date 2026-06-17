# Cross-condition: train CLEAN -> test FILTERED_1ST_STAGE

Same windowing, features, classifier, folds, and aggregation as
`train_compare.py` / `train_cross.py` / `train_cross_filtered.py`. Only
the source of the test windows changes (noisy / filtered_AI ->
filtered_1st_stage, i.e. the 1st-stage classical denoiser from
Signal_detection_assist).

Test files come from:
    C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\Filtered assist 1st stage

## Result (per-track accuracy, mean +/- std across 5 folds)

| experiment                                 | accuracy            | wrong / 250 | jazz | metal | pop |
|--------------------------------------------|---------------------|-----------------------|------|-------|-----|
| E: train clean, test filtered_1st_stage    | **0.340 +/- 0.013** | 165                   | 0.20 | 0.81 | 0.01 |

Window-level accuracy: 0.344 +/- 0.013

Confusion matrix: `confusion_clean_to_filtered_1st.png`.

## Placement against frozen baseline

- A clean -> clean              : 0.988  (upper bound)
- B noisy -> noisy              : 0.912  (noise-aware training)
- C clean -> noisy              : 0.344  (untreated domain-shift floor)
- D clean -> filtered_AI        : 0.404  (AI denoiser at test)
- **E clean -> filtered_1st     : 0.340**  (1st-stage classical denoiser at test)

Recovery fraction of the A-C gap closed by the 1st-stage filter:
`(E - C) / (A - C) = (0.340 - 0.344) / (0.988 - 0.344) = -0.62%`

## Leakage status

For each of the 5 folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test windows
come from the 1st-stage filter output, but the underlying track IDs
never overlap -- this measures generalisation to the 1st-stage-filter
distribution, not memorisation of individual tracks.
Full audit: `verify_no_leakage_filtered_1st.py`.
