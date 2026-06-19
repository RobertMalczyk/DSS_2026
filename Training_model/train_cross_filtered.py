"""
Cross-condition experiment D: train on CLEAN windows, test on FILTERED_AI windows.

Mirrors `train_cross.py` (experiment C: clean -> noisy) but swaps the test
source to the AI-denoised audio in
    C:\\Robak\\DSS2026\\Claude\\Signal_generation\\out\\Model\\filtered_AI\\<file_id>.wav

Purpose: measure how much an AI denoiser recovers the accuracy that
collapsed in experiment C (0.344). If filtering is effective the number
should sit between the C floor (0.344) and the A ceiling (0.988). Same
SVM pipeline, same 48-dim features, same StratifiedGroupKFold folds as
the frozen baseline in `train_compare.py`.

Features for filtered_AI are cached at cache/X_filtered_AI.npy (etc.).
Clean features reuse cache/X_clean.npy -- this script does NOT touch it.

Outputs (separate filenames, do not overwrite A/B/C results):
    results/accuracy_table_filtered.csv
    results/confusion_clean_to_filtered.png
    results/raw_results_filtered.json
    results/summary_filtered.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_compare import (
    DATA_ROOT, RESULTS_DIR, GENRES, GENRE_TO_IDX,
    SEED, N_SPLITS, SVC_C, SVC_GAMMA,
    build_dataset, aggregate_per_track, plot_confusion,
)

FILTERED_DIRNAME = "filtered_AI"


def build_filtered_dataset(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from the filtered_AI/ folder on the same window grid
    as clean/noisy. Re-uses `build_dataset` by injecting a synthetic
    '<source>_path' column, so the cache lands at cache/X_filtered_AI.npy."""
    filt_dir = DATA_ROOT / FILTERED_DIRNAME
    if not filt_dir.is_dir():
        raise FileNotFoundError(f"filtered_AI directory not found: {filt_dir}")

    labels_aug = labels.copy()
    labels_aug[f"{FILTERED_DIRNAME}_path"] = labels_aug["file_id"].apply(
        lambda fid: f"{FILTERED_DIRNAME}/{fid}.wav"
    )
    # Sanity-check every synthesised path resolves to a real file.
    missing = [p for p in labels_aug[f"{FILTERED_DIRNAME}_path"]
               if not (DATA_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} filtered_AI files missing "
            f"(first: {missing[0]})"
        )
    return build_dataset(labels_aug, FILTERED_DIRNAME)


def run_cross(
    X_train_src: np.ndarray, X_test_src: np.ndarray,
    y: np.ndarray, track_id: np.ndarray, track_genre: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> dict:
    """Train on X_train_src[train_win], test on X_test_src[test_win] where
    train_win / test_win come from the same track-level folds and are
    therefore drawn from disjoint track_ids."""
    fold_track_acc:  list[float] = []
    fold_window_acc: list[float] = []
    cm_total          = np.zeros((len(GENRES), len(GENRES)), dtype=np.int64)
    per_genre_correct = np.zeros(len(GENRES), dtype=np.int64)
    per_genre_total   = np.zeros(len(GENRES), dtype=np.int64)

    unique_tracks    = np.unique(track_id)
    track_to_windows = {int(t): np.where(track_id == t)[0] for t in unique_tracks}

    for k, (tr_idx, te_idx) in enumerate(splits):
        train_tracks = unique_tracks[tr_idx]
        test_tracks  = unique_tracks[te_idx]

        assert set(int(t) for t in train_tracks).isdisjoint(
               set(int(t) for t in test_tracks)), \
               f"fold {k+1}: train and test track sets overlap"

        tr_win = np.concatenate([track_to_windows[int(t)] for t in train_tracks])
        te_win = np.concatenate([track_to_windows[int(t)] for t in test_tracks])

        X_tr, y_tr = X_train_src[tr_win], y[tr_win]
        X_te, y_te = X_test_src [te_win], y[te_win]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(C=SVC_C, gamma=SVC_GAMMA, kernel="rbf",
                           probability=True, class_weight="balanced",
                           random_state=SEED)),
        ])
        clf.fit(X_tr, y_tr)

        win_pred = clf.predict(X_te)
        win_acc  = accuracy_score(y_te, win_pred)
        fold_window_acc.append(win_acc)

        proba    = clf.predict_proba(X_te)
        tids, track_pred = aggregate_per_track(proba, track_id[te_win])
        track_true = track_genre[tids]
        track_acc  = accuracy_score(track_true, track_pred)
        fold_track_acc.append(track_acc)

        cm_total += confusion_matrix(track_true, track_pred,
                                     labels=list(range(len(GENRES))))

        for g in range(len(GENRES)):
            mask = track_true == g
            per_genre_total[g]   += int(mask.sum())
            per_genre_correct[g] += int((track_pred[mask] == g).sum())

        print(f"  [{label}] fold {k+1}/{len(splits)}: "
              f"window_acc={win_acc:.3f}  track_acc={track_acc:.3f}  "
              f"(train_tracks={len(train_tracks)}, test_tracks={len(test_tracks)})")

    per_genre_acc = np.where(
        per_genre_total > 0, per_genre_correct / np.maximum(per_genre_total, 1), np.nan
    )
    wrong = int(per_genre_total.sum() - per_genre_correct.sum())

    return {
        "label":           label,
        "track_acc_mean":  float(np.mean(fold_track_acc)),
        "track_acc_std":   float(np.std(fold_track_acc)),
        "window_acc_mean": float(np.mean(fold_window_acc)),
        "window_acc_std":  float(np.std(fold_window_acc)),
        "fold_track_acc":  [float(x) for x in fold_track_acc],
        "per_genre_acc":   {GENRES[g]: float(per_genre_acc[g]) for g in range(len(GENRES))},
        "per_genre_correct": {GENRES[g]: int(per_genre_correct[g]) for g in range(len(GENRES))},
        "per_genre_total":   {GENRES[g]: int(per_genre_total[g])   for g in range(len(GENRES))},
        "wrong_total":       wrong,
        "total":             int(per_genre_total.sum()),
        "confusion":       cm_total.tolist(),
    }


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    Xc, yc, tc = build_dataset(labels, "clean")
    print(f"  clean:       {Xc.shape}")
    Xf, yf, tf = build_filtered_dataset(labels)
    print(f"  filtered_AI: {Xf.shape}")

    # Same window grid across sources? y and track_id must match -- otherwise
    # cross-condition inference is applying the clean-trained model to
    # misaligned (even mislabelled) test windows.
    assert np.array_equal(yc, yf), "y mismatch between clean and filtered_AI"
    assert np.array_equal(tc, tf), "track_id mismatch between clean and filtered_AI"
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

    print("=== Experiment D: train CLEAN -> test FILTERED_AI ===")
    res = run_cross(
        X_train_src=Xc, X_test_src=Xf,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label="clean->filtered_AI",
    )

    row = {
        "experiment":      "clean_train_filtered_AI_test",
        "mean_acc":        res["track_acc_mean"],
        "std_acc":         res["track_acc_std"],
        "window_mean_acc": res["window_acc_mean"],
        "window_std_acc":  res["window_acc_std"],
        "wrong_total":     res["wrong_total"],
        "total":           res["total"],
        **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
    }
    acc_df  = pd.DataFrame([row])
    acc_csv = RESULTS_DIR / "accuracy_table_filtered.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(
        np.array(res["confusion"]),
        "train clean -> test filtered_AI (track-level, summed across folds)",
        RESULTS_DIR / "confusion_clean_to_filtered.png",
    )

    # Anchor numbers from the frozen baseline (train_compare.py / train_cross.py)
    # for interpretation only -- not recomputed here.
    A_acc = 0.988
    C_acc = 0.344
    recovery = ((res["track_acc_mean"] - C_acc) / (A_acc - C_acc)) if A_acc > C_acc else float("nan")

    d, sd = res["track_acc_mean"], res["track_acc_std"]
    summary = f"""# Cross-condition: train CLEAN -> test FILTERED_AI

Same windowing, features, classifier, folds, and aggregation as
`train_compare.py` / `train_cross.py`. Only the source of the test
windows changes (noisy -> filtered_AI).

## Result (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment                         | accuracy            | wrong / {res['total']} | jazz | metal | pop |
|------------------------------------|---------------------|-----------------------|------|-------|-----|
| D: train clean, test filtered_AI   | **{d:.3f} +/- {sd:.3f}** | {res['wrong_total']}                   | {res['per_genre_acc']['jazz']:.2f} | {res['per_genre_acc']['metal']:.2f} | {res['per_genre_acc']['pop']:.2f} |

Window-level accuracy: {res['window_acc_mean']:.3f} +/- {res['window_acc_std']:.3f}

Confusion matrix: `confusion_clean_to_filtered.png`.

## Placement against frozen baseline

From `project_classifier_baseline_finding` (2026-04-22):
- A clean -> clean     : 0.988  (upper bound, no contamination at test time)
- B noisy -> noisy     : 0.912  (noise-aware training on noisy test)
- C clean -> noisy     : 0.344  (untreated domain-shift floor, ~= chance)
- **D clean -> filtered_AI : {d:.3f}**

Recovery fraction of the C -> A gap closed by the AI filter:
`(D - C) / (A - C) = ({d:.3f} - {C_acc}) / ({A_acc} - {C_acc}) = {recovery:.2%}`

## Leakage status

For each of the {N_SPLITS} folds, `train_tracks` and `test_tracks` are
disjoint (asserted in-loop). Train windows come from clean/, test windows
come from filtered_AI/, but the underlying track IDs never overlap, so
this measures generalisation to the AI-filtered distribution, not
memorisation of individual tracks.
"""
    (RESULTS_DIR / "summary_filtered.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / "raw_results_filtered.json").write_text(
        json.dumps({"clean_to_filtered_AI": res}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_filtered.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
