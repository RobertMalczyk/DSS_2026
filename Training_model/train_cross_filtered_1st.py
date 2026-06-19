"""
Cross-condition experiment E: train on CLEAN windows, test on
FILTERED_1ST_STAGE windows (classical 1st-stage denoiser output from the
sibling project Signal_detection_assist).

Test-source directory:
    C:\\Robak\\DSS2026\\Claude\\Signal_detection_assist\\Out\\Filtered assist 1st stage\\
File naming:
    filtered_assist_1st_stage_<file_id>.wav   (e.g. jazz_01 -> ...jazz_01.wav)

Same SVM pipeline, same 48-dim features, same StratifiedGroupKFold folds
as the frozen baseline (train_compare.py / train_cross.py /
train_cross_filtered.py). This makes E directly comparable to A/B/C/D:

    A clean -> clean         (upper bound, no contamination)
    B noisy -> noisy         (noise-aware training)
    C clean -> noisy         (untreated domain-shift floor)
    D clean -> filtered_AI   (AI denoiser at test time)
    E clean -> filtered_1st  (1st-stage classical denoiser at test time)

Features for the 1st-stage filter are cached at
cache/X_filtered_1st_stage.npy (etc.). Clean features reuse
cache/X_clean.npy unchanged.

Outputs (separate filenames, do not overwrite earlier results):
    results/accuracy_table_filtered_1st.csv
    results/confusion_clean_to_filtered_1st.png
    results/raw_results_filtered_1st.json
    results/summary_filtered_1st.md
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

FILTERED_1ST_DIR = Path(
    r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\Filtered assist 1st stage"
)
FILTERED_1ST_SOURCE = "filtered_1st_stage"   # cache key / log label


def build_filtered_1st_dataset(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from the 1st-stage filter folder onto the same
    window grid as clean/noisy. Uses absolute paths in the synthetic
    path column so `build_dataset` finds files outside DATA_ROOT —
    Path(DATA_ROOT) / absolute_path yields just the absolute path.
    """
    if not FILTERED_1ST_DIR.is_dir():
        raise FileNotFoundError(f"1st-stage filter dir not found: {FILTERED_1ST_DIR}")

    def abs_path_for(file_id: str) -> str:
        return str(FILTERED_1ST_DIR / f"filtered_assist_1st_stage_{file_id}.wav")

    labels_aug = labels.copy()
    labels_aug[f"{FILTERED_1ST_SOURCE}_path"] = labels_aug["file_id"].apply(abs_path_for)

    missing = [p for p in labels_aug[f"{FILTERED_1ST_SOURCE}_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} filtered_1st_stage files missing "
            f"(first: {missing[0]})"
        )

    return build_dataset(labels_aug, FILTERED_1ST_SOURCE)


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    Xc, yc, tc = build_dataset(labels, "clean")
    print(f"  clean:              {Xc.shape}")
    Xf, yf, tf = build_filtered_1st_dataset(labels)
    print(f"  filtered_1st_stage: {Xf.shape}")

    assert np.array_equal(yc, yf), "y mismatch between clean and filtered_1st_stage"
    assert np.array_equal(tc, tf), "track_id mismatch between clean and filtered_1st_stage"
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

    print("=== Experiment E: train CLEAN -> test FILTERED_1ST_STAGE ===")
    res = run_cross(
        X_train_src=Xc, X_test_src=Xf,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label="clean->filtered_1st_stage",
    )

    row = {
        "experiment":      "clean_train_filtered_1st_stage_test",
        "mean_acc":        res["track_acc_mean"],
        "std_acc":         res["track_acc_std"],
        "window_mean_acc": res["window_acc_mean"],
        "window_std_acc":  res["window_acc_std"],
        "wrong_total":     res["wrong_total"],
        "total":           res["total"],
        **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
    }
    acc_df  = pd.DataFrame([row])
    acc_csv = RESULTS_DIR / "accuracy_table_filtered_1st.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(
        np.array(res["confusion"]),
        "train clean -> test filtered_1st_stage (track-level, summed across folds)",
        RESULTS_DIR / "confusion_clean_to_filtered_1st.png",
    )

    # Anchor numbers from the frozen baseline for interpretation only.
    A_acc = 0.988
    C_acc = 0.344
    D_acc = 0.404
    e, se = res["track_acc_mean"], res["track_acc_std"]
    recovery = ((e - C_acc) / (A_acc - C_acc)) if A_acc > C_acc else float("nan")

    summary = f"""# Cross-condition: train CLEAN -> test FILTERED_1ST_STAGE

Same windowing, features, classifier, folds, and aggregation as
`train_compare.py` / `train_cross.py` / `train_cross_filtered.py`. Only
the source of the test windows changes (noisy / filtered_AI ->
filtered_1st_stage, i.e. the 1st-stage classical denoiser from
Signal_detection_assist).

Test files come from:
    {FILTERED_1ST_DIR}

## Result (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment                                 | accuracy            | wrong / {res['total']} | jazz | metal | pop |
|--------------------------------------------|---------------------|-----------------------|------|-------|-----|
| E: train clean, test filtered_1st_stage    | **{e:.3f} +/- {se:.3f}** | {res['wrong_total']}                   | {res['per_genre_acc']['jazz']:.2f} | {res['per_genre_acc']['metal']:.2f} | {res['per_genre_acc']['pop']:.2f} |

Window-level accuracy: {res['window_acc_mean']:.3f} +/- {res['window_acc_std']:.3f}

Confusion matrix: `confusion_clean_to_filtered_1st.png`.

## Placement against frozen baseline

- A clean -> clean              : 0.988  (upper bound)
- B noisy -> noisy              : 0.912  (noise-aware training)
- C clean -> noisy              : 0.344  (untreated domain-shift floor)
- D clean -> filtered_AI        : {D_acc:.3f}  (AI denoiser at test)
- **E clean -> filtered_1st     : {e:.3f}**  (1st-stage classical denoiser at test)

Recovery fraction of the A-C gap closed by the 1st-stage filter:
`(E - C) / (A - C) = ({e:.3f} - {C_acc}) / ({A_acc} - {C_acc}) = {recovery:.2%}`

## Leakage status

For each of the {N_SPLITS} folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test windows
come from the 1st-stage filter output, but the underlying track IDs
never overlap -- this measures generalisation to the 1st-stage-filter
distribution, not memorisation of individual tracks.
Full audit: `verify_no_leakage_filtered_1st.py`.
"""
    (RESULTS_DIR / "summary_filtered_1st.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / "raw_results_filtered_1st.json").write_text(
        json.dumps({"clean_to_filtered_1st_stage": res}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_filtered_1st.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
