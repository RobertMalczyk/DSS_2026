# Cross-condition: train CLEAN -> test FILTERED_AI

Same windowing, features, classifier, folds, and aggregation as
`train_compare.py` / `train_cross.py`. Only the source of the test
windows changes (noisy -> filtered_AI).

## Result (per-track accuracy, mean +/- std across 5 folds)

| experiment                         | accuracy            | wrong / 250 | jazz | metal | pop |
|------------------------------------|---------------------|-----------------------|------|-------|-----|
| D: train clean, test filtered_AI   | **0.404 +/- 0.064** | 149                   | 0.99 | 0.00 | 0.22 |

Window-level accuracy: 0.409 +/- 0.040

Confusion matrix: `confusion_clean_to_filtered.png`.

## Placement against frozen baseline

From `project_classifier_baseline_finding` (2026-04-22):
- A clean -> clean     : 0.988  (upper bound, no contamination at test time)
- B noisy -> noisy     : 0.912  (noise-aware training on noisy test)
- C clean -> noisy     : 0.344  (untreated domain-shift floor, ~= chance)
- **D clean -> filtered_AI : 0.404**

Recovery fraction of the C -> A gap closed by the AI filter:
`(D - C) / (A - C) = (0.404 - 0.344) / (0.988 - 0.344) = 9.32%`

## Leakage status

For each of the 5 folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test windows
come from filtered_AI/, but the underlying track IDs never overlap, so
this measures generalisation to the AI-filtered distribution, not
memorisation of individual tracks.
