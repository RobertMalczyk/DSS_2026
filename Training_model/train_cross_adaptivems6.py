"""
Cross-condition experiment G: train on CLEAN windows, test on
ADAPTIVE-MS 6TH-STAGE windows.

Test-source directory (inside the shared Model/ root):
    C:\\Robak\\DSS2026\\Claude\\Signal_generation\\out\\Model\\Filtered AdaptiveMS 6th stage\\filtered_adaptivems_6th_stage_<file_id>.wav

Same SVM pipeline, same 48-dim features, same StratifiedGroupKFold folds
as the frozen baseline, so G is directly comparable to A/B/C/D/E/F:

    A clean -> clean              (upper bound)
    B noisy -> noisy              (noise-aware training)
    C clean -> noisy              (domain-shift floor)
    D clean -> filtered_AI        (AI denoiser at test)
    E clean -> filtered_1st       (classical 1st-stage denoiser at test)
    F clean -> filtered_2nd       (classical+AI 2nd-stage denoiser at test)
    G clean -> adaptivems_6th     (adaptive mean-subtraction 6th-stage at test)

Features for the 6th-stage filter are cached at
cache/X_filtered_adaptivems_6th_stage.npy (etc.). Clean features reuse
cache/X_clean.npy unchanged.

Outputs:
    results/accuracy_table_adaptivems6.csv
    results/confusion_clean_to_adaptivems6.png
    results/raw_results_adaptivems6.json
    results/summary_adaptivems6.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from train_compare import (
    DATA_ROOT, RESULTS_DIR, GENRES, GENRE_TO_IDX,
    SEED, N_SPLITS,
    build_dataset, plot_confusion,
)
from train_cross_filtered import run_cross
from sklearn.model_selection import StratifiedGroupKFold

ADAPTIVEMS_DIRNAME = "Filtered AdaptiveMS 6th stage"
ADAPTIVEMS_PREFIX  = "filtered_adaptivems_6th_stage_"
ADAPTIVEMS_SOURCE  = "filtered_adaptivems_6th_stage"   # cache key / log label


def build_adaptivems_dataset(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from the AdaptiveMS 6th-stage folder onto the same
    window grid as clean/noisy. Synthesises a '<source>_path' column with
    paths relative to DATA_ROOT so build_dataset picks them up and caches
    them as X_filtered_adaptivems_6th_stage.npy."""
    filt_dir = DATA_ROOT / ADAPTIVEMS_DIRNAME
    if not filt_dir.is_dir():
        raise FileNotFoundError(f"AdaptiveMS 6th-stage dir not found: {filt_dir}")

    labels_aug = labels.copy()
    labels_aug[f"{ADAPTIVEMS_SOURCE}_path"] = labels_aug["file_id"].apply(
        lambda fid: f"{ADAPTIVEMS_DIRNAME}/{ADAPTIVEMS_PREFIX}{fid}.wav"
    )
    missing = [p for p in labels_aug[f"{ADAPTIVEMS_SOURCE}_path"]
               if not (DATA_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} adaptivems_6th_stage files missing "
            f"(first: {missing[0]})"
        )
    return build_dataset(labels_aug, ADAPTIVEMS_SOURCE)


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    Xc, yc, tc = build_dataset(labels, "clean")
    print(f"  clean:                          {Xc.shape}")
    Xg, yg, tg = build_adaptivems_dataset(labels)
    print(f"  filtered_adaptivems_6th_stage:  {Xg.shape}")

    assert np.array_equal(yc, yg), "y mismatch between clean and adaptivems_6th_stage"
    assert np.array_equal(tc, tg), "track_id mismatch between clean and adaptivems_6th_stage"
    print("  shared window grid OK (same y and track_id arrays)")

    unique_tracks = np.unique(tc)
    track_genre   = np.array(
        [GENRE_TO_IDX[labels.loc[int(t), "genre"]] for t in unique_tracks],
        dtype=np.int64,
    )

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre,
        groups=unique_tracks,
    ))
    print(f"Fold sizes (test tracks): {[int(len(te)) for _, te in splits]}\n")

    print("=== Experiment G: train CLEAN -> test ADAPTIVEMS_6TH_STAGE ===")
    res = run_cross(
        X_train_src=Xc, X_test_src=Xg,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label="clean->filtered_adaptivems_6th_stage",
    )

    row = {
        "experiment":      "clean_train_adaptivems_6th_stage_test",
        "mean_acc":        res["track_acc_mean"],
        "std_acc":         res["track_acc_std"],
        "window_mean_acc": res["window_acc_mean"],
        "window_std_acc":  res["window_acc_std"],
        "wrong_total":     res["wrong_total"],
        "total":           res["total"],
        **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
    }
    acc_df  = pd.DataFrame([row])
    acc_csv = RESULTS_DIR / "accuracy_table_adaptivems6.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(
        np.array(res["confusion"]),
        "train clean -> test adaptivems_6th_stage (track-level, summed across folds)",
        RESULTS_DIR / "confusion_clean_to_adaptivems6.png",
    )

    # Baseline anchors (frozen).
    A_acc = 0.988
    C_acc = 0.344
    D_acc = 0.404
    E_acc = 0.340
    F_acc = 0.436
    g,  sg = res["track_acc_mean"], res["track_acc_std"]
    recovery = ((g - C_acc) / (A_acc - C_acc)) if A_acc > C_acc else float("nan")

    summary = f"""# Cross-condition: train CLEAN -> test ADAPTIVEMS_6TH_STAGE

Same windowing, features, classifier, folds, and aggregation as the
rest of the baseline. Only the test-source audio changes.

Test files come from:
    {DATA_ROOT / ADAPTIVEMS_DIRNAME}
Filename pattern: `{ADAPTIVEMS_PREFIX}<file_id>.wav`.

## Result (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment                                        | accuracy            | wrong / {res['total']} | jazz | metal | pop |
|---------------------------------------------------|---------------------|-----------------------|------|-------|-----|
| G: train clean, test adaptivems_6th_stage         | **{g:.3f} +/- {sg:.3f}** | {res['wrong_total']}                   | {res['per_genre_acc']['jazz']:.2f} | {res['per_genre_acc']['metal']:.2f} | {res['per_genre_acc']['pop']:.2f} |

Window-level accuracy: {res['window_acc_mean']:.3f} +/- {res['window_acc_std']:.3f}

Confusion matrix: `confusion_clean_to_adaptivems6.png`.

## Placement against frozen baseline

- A clean -> clean                  : 0.988  (upper bound)
- B noisy -> noisy                  : 0.912  (noise-aware training)
- C clean -> noisy                  : 0.344  (untreated domain-shift floor)
- D clean -> filtered_AI            : {D_acc:.3f}  (AI denoiser at test)
- E clean -> filtered_1st           : {E_acc:.3f}  (classical 1st-stage denoiser at test)
- F clean -> filtered_2nd           : {F_acc:.3f}  (classical+AI 2nd-stage denoiser at test)
- **G clean -> adaptivems_6th       : {g:.3f}**  (adaptive-MS 6th-stage denoiser at test)

Recovery fraction of the A-C gap closed by the AdaptiveMS 6th-stage filter:
`(G - C) / (A - C) = ({g:.3f} - {C_acc}) / ({A_acc} - {C_acc}) = {recovery:.2%}`

## Leakage status

For each of the {N_SPLITS} folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test
windows come from the 6th-stage filter output, but the underlying track
IDs never overlap. Full audit: `verify_no_leakage_adaptivems6.py`.
"""
    (RESULTS_DIR / "summary_adaptivems6.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / "raw_results_adaptivems6.json").write_text(
        json.dumps({"clean_to_filtered_adaptivems_6th_stage": res}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_adaptivems6.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
