"""
Same matched-fold experiment as train_compare.py, but with XGBoost
instead of SVC. Reuses the cached feature matrices and the *exact* same
StratifiedGroupKFold splits, so the results are directly comparable to the
SVM baseline.

Outputs (separate filenames so they don't clobber the SVM run):
  results/accuracy_table_xgb.csv
  results/confusion_clean_xgb.png
  results/confusion_noisy_xgb.png
  results/raw_results_xgb.json
  results/summary_xgb.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from train_compare import (
    DATA_ROOT, RESULTS_DIR, GENRES, GENRE_TO_IDX,
    SEED, N_SPLITS,
    build_dataset, aggregate_per_track, plot_confusion,
)

# ----------------------------- xgboost hyperparameters --------------------
XGB_PARAMS = dict(
    objective       = "multi:softprob",
    num_class       = len(GENRES),
    n_estimators    = 400,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.9,
    colsample_bytree= 0.9,
    min_child_weight= 2,
    reg_lambda      = 1.0,
    tree_method     = "hist",
    eval_metric     = "mlogloss",
    random_state    = SEED,
    n_jobs          = -1,
)

# --------------------------------------------------------------------------


def run_experiment_xgb(
    X: np.ndarray, y: np.ndarray, track_id: np.ndarray,
    track_genre: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> dict:
    fold_track_acc:  list[float] = []
    fold_window_acc: list[float] = []
    cm_total          = np.zeros((len(GENRES), len(GENRES)), dtype=np.int64)
    per_genre_correct = np.zeros(len(GENRES), dtype=np.int64)
    per_genre_total   = np.zeros(len(GENRES), dtype=np.int64)

    unique_tracks = np.unique(track_id)
    track_to_windows = {int(t): np.where(track_id == t)[0] for t in unique_tracks}

    for k, (tr_idx, te_idx) in enumerate(splits):
        train_tracks = unique_tracks[tr_idx]
        test_tracks  = unique_tracks[te_idx]

        tr_win = np.concatenate([track_to_windows[int(t)] for t in train_tracks])
        te_win = np.concatenate([track_to_windows[int(t)] for t in test_tracks])

        X_tr, y_tr = X[tr_win], y[tr_win]
        X_te, y_te = X[te_win], y[te_win]

        clf = XGBClassifier(**XGB_PARAMS)
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
              f"window_acc={win_acc:.3f}  track_acc={track_acc:.3f}")

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

    # cached features (built by train_compare.py); rebuild only if missing.
    Xc, yc, tc = build_dataset(labels, "clean")
    Xn, yn, tn = build_dataset(labels, "noisy")
    assert np.array_equal(yc, yn)
    assert np.array_equal(tc, tn)

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

    print("=== XGBoost: clean -> clean ===")
    res_clean = run_experiment_xgb(Xc, yc, tc, track_genre, splits, "clean")

    print("\n=== XGBoost: noisy -> noisy ===")
    res_noisy = run_experiment_xgb(Xn, yn, tn, track_genre, splits, "noisy")

    rows = []
    for tag, res in [("clean_vs_clean", res_clean), ("noisy_vs_noisy", res_noisy)]:
        rows.append({
            "experiment":      tag,
            "mean_acc":        res["track_acc_mean"],
            "std_acc":         res["track_acc_std"],
            "window_mean_acc": res["window_acc_mean"],
            "window_std_acc":  res["window_acc_std"],
            **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
        })
    acc_df  = pd.DataFrame(rows)
    acc_csv = RESULTS_DIR / "accuracy_table_xgb.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(np.array(res_clean["confusion"]),
                   "XGBoost  clean -> clean (track-level, summed across folds)",
                   RESULTS_DIR / "confusion_clean_xgb.png")
    plot_confusion(np.array(res_noisy["confusion"]),
                   "XGBoost  noisy -> noisy (track-level, summed across folds)",
                   RESULTS_DIR / "confusion_noisy_xgb.png")

    a, sa = res_clean["track_acc_mean"], res_clean["track_acc_std"]
    b, sb = res_noisy["track_acc_mean"], res_noisy["track_acc_std"]
    abs_drop = a - b
    rel_drop = (abs_drop / a) if a > 0 else float("nan")

    summary = f"""# XGBoost: clean vs noisy

Same windowing, features, folds, and aggregation as `train_compare.py`
(see `results/summary.md`); only the classifier is swapped from
`SVC(rbf)` to `XGBClassifier`.

## XGBoost hyperparameters

```
{json.dumps(XGB_PARAMS, indent=2)}
```

## Results (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment       | accuracy            | jazz | metal | pop |
|------------------|---------------------|------|-------|-----|
| clean -> clean   | **{a:.3f} +/- {sa:.3f}** | {res_clean['per_genre_acc']['jazz']:.2f} | {res_clean['per_genre_acc']['metal']:.2f} | {res_clean['per_genre_acc']['pop']:.2f} |
| noisy -> noisy   | **{b:.3f} +/- {sb:.3f}** | {res_noisy['per_genre_acc']['jazz']:.2f} | {res_noisy['per_genre_acc']['metal']:.2f} | {res_noisy['per_genre_acc']['pop']:.2f} |

- **Absolute accuracy drop from noise:** {abs_drop:+.3f}
- **Relative accuracy drop from noise:** {rel_drop*100:+.1f} %

Window-level accuracy (informational):
- clean -> clean: {res_clean['window_acc_mean']:.3f} +/- {res_clean['window_acc_std']:.3f}
- noisy -> noisy: {res_noisy['window_acc_mean']:.3f} +/- {res_noisy['window_acc_std']:.3f}

Confusion matrices: `confusion_clean_xgb.png`, `confusion_noisy_xgb.png`.
"""
    (RESULTS_DIR / "summary_xgb.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / "raw_results_xgb.json").write_text(
        json.dumps({"clean": res_clean, "noisy": res_noisy}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_xgb.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
