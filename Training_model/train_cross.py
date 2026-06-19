"""
Cross-condition experiment: train on CLEAN windows, test on NOISY windows.

Uses the same SVM pipeline, the same cached features, and the *same*
StratifiedGroupKFold splits as `train_compare.py` (experiments A and B).
For each fold:
    train_set = clean[train_tracks]
    test_set  = noisy[test_tracks]
Train- and test-track IDs are disjoint (track-level GroupKFold), so this
measures generalisation across the noise distribution, not memorisation
of individual tracks.

Outputs (separate filenames, do not overwrite A/B results):
    results/accuracy_table_cross.csv
    results/confusion_clean_to_noisy.png
    results/raw_results_cross.json
    results/summary_cross.md
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


def run_cross(
    X_train_src: np.ndarray, X_test_src: np.ndarray,
    y: np.ndarray, track_id: np.ndarray, track_genre: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> dict:
    """Train on X_train_src[train_win], test on X_test_src[test_win], where
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

        # Belt-and-braces: assert no track id is in both sets.
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

    return {
        "label":           label,
        "track_acc_mean":  float(np.mean(fold_track_acc)),
        "track_acc_std":   float(np.std(fold_track_acc)),
        "window_acc_mean": float(np.mean(fold_window_acc)),
        "window_acc_std":  float(np.std(fold_window_acc)),
        "fold_track_acc":  [float(x) for x in fold_track_acc],
        "per_genre_acc":   {GENRES[g]: float(per_genre_acc[g]) for g in range(len(GENRES))},
        "confusion":       cm_total.tolist(),
    }


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    Xc, yc, tc = build_dataset(labels, "clean")
    Xn, yn, tn = build_dataset(labels, "noisy")
    assert np.array_equal(yc, yn), "y mismatch between clean and noisy"
    assert np.array_equal(tc, tn), "track_id mismatch between clean and noisy"
    print(f"  clean: {Xc.shape}, noisy: {Xn.shape}, shared track grid OK")

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

    print("=== Experiment C: train CLEAN -> test NOISY ===")
    res = run_cross(
        X_train_src=Xc, X_test_src=Xn,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label="clean->noisy",
    )

    row = {
        "experiment":      "clean_train_noisy_test",
        "mean_acc":        res["track_acc_mean"],
        "std_acc":         res["track_acc_std"],
        "window_mean_acc": res["window_acc_mean"],
        "window_std_acc":  res["window_acc_std"],
        **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
    }
    acc_df  = pd.DataFrame([row])
    acc_csv = RESULTS_DIR / "accuracy_table_cross.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(np.array(res["confusion"]),
                   "train clean -> test noisy (track-level, summed across folds)",
                   RESULTS_DIR / "confusion_clean_to_noisy.png")

    a, sa = res["track_acc_mean"], res["track_acc_std"]

    summary = f"""# Cross-condition: train CLEAN -> test NOISY

Same windowing, features, classifier, folds, and aggregation as
`train_compare.py` (experiments A and B). Only the source of train vs test
windows differs.

## Result (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment                 | accuracy            | jazz | metal | pop |
|----------------------------|---------------------|------|-------|-----|
| C: train clean, test noisy | **{a:.3f} +/- {sa:.3f}** | {res['per_genre_acc']['jazz']:.2f} | {res['per_genre_acc']['metal']:.2f} | {res['per_genre_acc']['pop']:.2f} |

Window-level accuracy: {res['window_acc_mean']:.3f} +/- {res['window_acc_std']:.3f}

Confusion matrix: `confusion_clean_to_noisy.png`.

## Leakage status

For each of the {N_SPLITS} folds, `train_tracks` and `test_tracks` are
disjoint (verified by the in-loop assert in `train_cross.py` and by
`verify_no_leakage.py` on the underlying fold structure). Although the
audio source differs between train and test, the track IDs do not overlap,
so no track is seen in both — this measures generalisation to the noise
distribution, not memorisation.
"""
    (RESULTS_DIR / "summary_cross.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / "raw_results_cross.json").write_text(
        json.dumps({"clean_to_noisy": res}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_cross.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
