"""
verify_int8_quantization.py

Read-only experiment: does int8 quantization of the v2 RBF-SVM's support
vectors preserve the H_v2 = 0.796 accuracy that backs slide 19's
"smarter beats bigger" claim?

The deck currently advertises ~250 KB peak memory for the v2 model, which
is only achievable if the 1496 trained SVs are quantized from sklearn's
default float64 to int8. The project never measured that quantization.
This script measures it.

Protocol (matches `Training_model/train_cross_v2_noisy_to_7th.py` H_v2 fold
exactly so the numbers compare against the published 0.796):
  - Train: cache/X_clean_v2.npy            (clean v2 features)
  - Test:  cache/X_filtered_componentmatched_7th_stage_v2.npy  (Stage 7 v2)
  - Pipeline: StandardScaler -> SVC(C=10, gamma='scale', rbf,
              probability=True, class_weight='balanced', random_state=42)
  - 5-fold StratifiedGroupKFold(seed=42) over track_id
  - Per-fold: fit float baseline, quantize SVs in-place to int8 (per-feature
              symmetric, scale = abs_max/127), dequantize at inference
              (libsvm path stays float internally so Platt sigmoid still maps).
  - Aggregate windows -> tracks via mean predicted probability.

Outputs (project root):
  - int8_quantization_results.json        per-fold + aggregate numbers
  - int8_quantization_report.md           human-readable summary

Nothing in Training_model/ or Robert_Malczyk.pptx is modified. The cached
.npy feature matrices are read-only.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT      = Path(r"C:\Robak\DSS2026\Claude")
CACHE_DIR = ROOT / "Training_model" / "cache"

GENRES        = ("jazz", "metal", "pop")
SEED          = 42
N_SPLITS      = 5
SVC_C         = 10.0
SVC_GAMMA     = "scale"


def load_cached(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(CACHE_DIR / f"X_{stem}.npy")
    y = np.load(CACHE_DIR / f"y_{stem}.npy")
    t = np.load(CACHE_DIR / f"t_{stem}.npy")
    return X, y, t


def aggregate_per_track(proba: np.ndarray, tids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean soft-probability across windows of the same track -> per-track pred."""
    uniq = np.unique(tids)
    preds = np.empty(len(uniq), dtype=np.int64)
    for i, tid in enumerate(uniq):
        mask = tids == tid
        preds[i] = int(np.argmax(proba[mask].mean(axis=0)))
    return uniq, preds


def quantize_svs_int8(svs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature symmetric int8 quantization.

    svs shape: (n_sv, n_features). Returns (q, scales) where
    q is int8 and dequant = q.astype(float64) * scales (broadcast).
    """
    abs_max = np.abs(svs).max(axis=0).astype(np.float64)
    abs_max[abs_max == 0.0] = 1.0
    scales = abs_max / 127.0
    q = np.round(svs / scales).astype(np.int8)
    q = np.clip(q.astype(np.int16), -127, 127).astype(np.int8)  # safety
    return q, scales


def dequantize_svs(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float64) * scales[None, :]


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(C=SVC_C, gamma=SVC_GAMMA, kernel="rbf",
                       probability=True, class_weight="balanced",
                       random_state=SEED)),
    ])


def main() -> int:
    print("=== int8 SV quantization verification (read-only) ===\n")
    t0 = time.time()

    # ---- load cached v2 features ----
    Xc, yc, tc = load_cached("clean_v2")
    Xh, yh, th = load_cached("filtered_componentmatched_7th_stage_v2")

    assert np.array_equal(yc, yh), "y mismatch clean vs 7th"
    assert np.array_equal(tc, th), "track_id mismatch clean vs 7th"
    print(f"clean_v2: X={Xc.shape}, y unique={sorted(set(yc.tolist()))}")
    print(f"7th_v2  : X={Xh.shape}\n")

    # ---- shared 5-fold split (track-level) ----
    unique_tracks = np.unique(tc)
    track_genre = np.array(
        [int(yc[np.where(tc == t)[0][0]]) for t in unique_tracks],
        dtype=np.int64,
    )
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre, groups=unique_tracks,
    ))
    print(f"Fold sizes (test tracks): {[len(te) for _, te in splits]}\n")

    track_to_windows = {int(t): np.where(tc == t)[0] for t in unique_tracks}

    fold_results = []
    fold_track_acc_fp64: list[float] = []
    fold_track_acc_int8: list[float] = []
    fold_window_acc_fp64: list[float] = []
    fold_window_acc_int8: list[float] = []
    sv_count_per_fold: list[int] = []
    fp64_bytes_per_fold: list[int] = []
    int8_bytes_per_fold: list[int] = []

    for k, (tr_idx, te_idx) in enumerate(splits):
        train_tracks = unique_tracks[tr_idx]
        test_tracks  = unique_tracks[te_idx]

        tr_win = np.concatenate([track_to_windows[int(t)] for t in train_tracks])
        te_win = np.concatenate([track_to_windows[int(t)] for t in test_tracks])

        X_tr, y_tr = Xc[tr_win], yc[tr_win]
        X_te, y_te = Xh[te_win], yh[te_win]

        # ---- fit float64 baseline ----
        clf = make_pipeline()
        clf.fit(X_tr, y_tr)
        svc = clf.named_steps["svc"]
        n_sv = int(svc.support_vectors_.shape[0])
        sv_count_per_fold.append(n_sv)

        # baseline predictions
        win_pred_fp = clf.predict(X_te)
        proba_fp = clf.predict_proba(X_te)
        _, track_pred_fp = aggregate_per_track(proba_fp, tc[te_win])
        track_true = track_genre[
            np.searchsorted(unique_tracks, np.unique(tc[te_win]))
        ]

        win_acc_fp = float(accuracy_score(y_te, win_pred_fp))
        track_acc_fp = float(accuracy_score(track_true, track_pred_fp))

        # ---- quantize SVs to int8 in a deep copy ----
        clf_q = copy.deepcopy(clf)
        svc_q = clf_q.named_steps["svc"]
        svs_fp = svc_q.support_vectors_.copy()
        q, scales = quantize_svs_int8(svs_fp)
        svs_dq = dequantize_svs(q, scales)
        # round-trip error stats (sanity)
        rt_err = float(np.max(np.abs(svs_dq - svs_fp)))
        rt_rmse = float(np.sqrt(np.mean((svs_dq - svs_fp) ** 2)))
        # write dequantized SVs back so libsvm uses them at predict
        svc_q.support_vectors_ = svs_dq.astype(svs_fp.dtype, copy=False)

        win_pred_q = clf_q.predict(X_te)
        proba_q = clf_q.predict_proba(X_te)
        _, track_pred_q = aggregate_per_track(proba_q, tc[te_win])

        win_acc_q   = float(accuracy_score(y_te, win_pred_q))
        track_acc_q = float(accuracy_score(track_true, track_pred_q))

        # ---- footprint accounting ----
        fp64_bytes = svs_fp.nbytes
        int8_bytes = q.nbytes + scales.nbytes  # int8 SVs + float64 scales
        fp64_bytes_per_fold.append(int(fp64_bytes))
        int8_bytes_per_fold.append(int(int8_bytes))

        fold_results.append({
            "fold":                 k + 1,
            "n_sv":                 n_sv,
            "train_tracks":         int(len(train_tracks)),
            "test_tracks":          int(len(test_tracks)),
            "fp64_window_acc":      win_acc_fp,
            "int8_window_acc":      win_acc_q,
            "fp64_track_acc":       track_acc_fp,
            "int8_track_acc":       track_acc_q,
            "track_acc_delta":      track_acc_q - track_acc_fp,
            "sv_roundtrip_max_err": rt_err,
            "sv_roundtrip_rmse":    rt_rmse,
            "fp64_sv_bytes":        int(fp64_bytes),
            "int8_sv_bytes":        int(int8_bytes),
        })

        fold_track_acc_fp64.append(track_acc_fp)
        fold_track_acc_int8.append(track_acc_q)
        fold_window_acc_fp64.append(win_acc_fp)
        fold_window_acc_int8.append(win_acc_q)

        print(f"  fold {k+1}/{N_SPLITS}: "
              f"n_sv={n_sv}  "
              f"track_acc fp64={track_acc_fp:.4f} int8={track_acc_q:.4f} "
              f"delta={track_acc_q - track_acc_fp:+.4f}  "
              f"window_acc fp64={win_acc_fp:.4f} int8={win_acc_q:.4f}  "
              f"sv_err_max={rt_err:.3e}")

    fp64_track_mean = float(np.mean(fold_track_acc_fp64))
    int8_track_mean = float(np.mean(fold_track_acc_int8))
    fp64_track_std  = float(np.std(fold_track_acc_fp64))
    int8_track_std  = float(np.std(fold_track_acc_int8))

    fp64_win_mean = float(np.mean(fold_window_acc_fp64))
    int8_win_mean = float(np.mean(fold_window_acc_int8))

    avg_n_sv = float(np.mean(sv_count_per_fold))
    avg_fp64_kb = float(np.mean(fp64_bytes_per_fold)) / 1024.0
    avg_int8_kb = float(np.mean(int8_bytes_per_fold)) / 1024.0

    summary = {
        "protocol": {
            "train_source": "clean_v2",
            "test_source":  "filtered_componentmatched_7th_stage_v2",
            "experiment":   "H_v2 (clean -> Stage 7)",
            "pipeline":     "StandardScaler -> SVC(rbf, C=10, gamma=scale, "
                            "probability=True, class_weight=balanced, seed=42)",
            "n_splits":     N_SPLITS,
            "seed":         SEED,
            "quantization": "per-feature symmetric int8 of support_vectors_; "
                            "dequantized at predict so libsvm/Platt path "
                            "stays float internally",
        },
        "headline": {
            "fp64_track_acc_mean": fp64_track_mean,
            "fp64_track_acc_std":  fp64_track_std,
            "int8_track_acc_mean": int8_track_mean,
            "int8_track_acc_std":  int8_track_std,
            "delta_track_acc":     int8_track_mean - fp64_track_mean,
            "fp64_window_acc_mean": fp64_win_mean,
            "int8_window_acc_mean": int8_win_mean,
        },
        "footprint_avg_per_fold": {
            "n_sv":               avg_n_sv,
            "fp64_sv_kb":         avg_fp64_kb,
            "int8_sv_kb":         avg_int8_kb,
            "compression_ratio":  avg_fp64_kb / max(avg_int8_kb, 1e-9),
        },
        "fold_results": fold_results,
    }

    out_json = ROOT / "int8_quantization_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_json.name}")

    # ---- markdown report ----
    md = []
    md.append("# int8 SV quantization — H_v2 verification\n")
    md.append(f"_Generated by `verify_int8_quantization.py` on "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}._\n")
    md.append("## Protocol\n")
    md.append("- Train: `cache/X_clean_v2.npy` (v2 features, bf4000 prefiltered clean)")
    md.append("- Test:  `cache/X_filtered_componentmatched_7th_stage_v2.npy` "
              "(v2 features, Stage 7 ComponentMatched filtered)")
    md.append("- Pipeline: `StandardScaler -> SVC(C=10, gamma='scale', kernel='rbf', "
              "probability=True, class_weight='balanced', random_state=42)`")
    md.append(f"- {N_SPLITS}-fold StratifiedGroupKFold over `track_id`, seed {SEED}")
    md.append("- Aggregation: mean predicted probability across windows per track, "
              "argmax to track-level prediction")
    md.append("- Quantization: per-feature symmetric int8 of `support_vectors_` "
              "(scale = abs_max / 127). Dequantized at predict time, so libsvm "
              "decision_function and Platt sigmoid see float values close to "
              "the original.\n")
    md.append("## Headline result\n")
    md.append(f"- **fp64 track accuracy** (baseline): "
              f"`{fp64_track_mean:.4f} ± {fp64_track_std:.4f}`")
    md.append(f"- **int8 track accuracy** (quantized SVs): "
              f"`{int8_track_mean:.4f} ± {int8_track_std:.4f}`")
    md.append(f"- **Δ track accuracy**: `{int8_track_mean - fp64_track_mean:+.4f}` "
              f"({(int8_track_mean - fp64_track_mean) * 100:+.2f} pp)\n")
    md.append(f"- fp64 window accuracy: `{fp64_win_mean:.4f}`")
    md.append(f"- int8 window accuracy: `{int8_win_mean:.4f}`\n")
    md.append("## Footprint (averaged across folds)\n")
    md.append(f"- Average SV count: `{avg_n_sv:.1f}`")
    md.append(f"- fp64 SV matrix:   `{avg_fp64_kb:.1f} KB`")
    md.append(f"- int8 SV matrix + float scales: `{avg_int8_kb:.1f} KB`")
    md.append(f"- Compression ratio: `{avg_fp64_kb / max(avg_int8_kb, 1e-9):.2f}×`\n")
    md.append("## Per-fold detail\n")
    md.append("| Fold | n_sv | fp64 track | int8 track | Δ | fp64 window | "
              "int8 window | sv max err |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in fold_results:
        md.append(
            f"| {r['fold']} | {r['n_sv']} | "
            f"{r['fp64_track_acc']:.4f} | {r['int8_track_acc']:.4f} | "
            f"{r['track_acc_delta']:+.4f} | "
            f"{r['fp64_window_acc']:.4f} | {r['int8_window_acc']:.4f} | "
            f"{r['sv_roundtrip_max_err']:.2e} |"
        )
    md.append("")
    md.append("## What this means for slide 19\n")
    md.append("- The published `H_v2 = 0.796` headline assumes float64 SVs.")
    md.append("- After int8 quantization the headline becomes the `int8 track "
              "accuracy` reported above. If `Δ` is small (< 1 pp), the slide can "
              "honestly cite ~260 KB at int8 alongside 0.79x track accuracy.")
    md.append("- If `Δ` is large, the slide should either fall back to the float "
              "footprint (~1 MB at float32) or disclose the accuracy hit.")

    out_md = ROOT / "int8_quantization_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {out_md.name}")
    print(f"\nTotal runtime: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
