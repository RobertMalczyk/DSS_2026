"""
Train and compare music-genre classifiers on clean vs noisy audio.

Two matched experiments under identical GroupKFold (track-level) splits:
    A: clean   -> clean
    B: noisy   -> noisy

Pipeline (classical, CPU-only):
    1. Read labels.csv, load each WAV (clean and noisy).
    2. Slice each 30 s track into fixed-length windows (50 % overlap).
    3. Per window, extract MFCC mean/std + spectral centroid/rolloff/ZCR/
       bandwidth mean/std -> fixed-length feature vector.
    4. StandardScaler + SVC(rbf), 5-fold StratifiedGroupKFold on track_id.
    5. Aggregate per-window soft probabilities -> per-track prediction.
    6. Save accuracy table, per-genre accuracy, confusion-matrix plots,
       and a short summary.md.

Run:
    "/c/Users/robak/AppData/Local/Programs/Python/Python312/python.exe" train_compare.py
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
import soundfile as sf
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------- hyperparameters -----------------------------
SEED          = 42
DATA_ROOT     = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model")
OUT_ROOT      = Path(__file__).resolve().parent
CACHE_DIR     = OUT_ROOT / "cache"
RESULTS_DIR   = OUT_ROOT / "results"

SR            = 22050
WIN_S         = 3.0      # window length, seconds
HOP_S         = 1.5      # hop, seconds (50 % overlap)
N_MFCC        = 20

N_FFT         = 2048
HOP_LENGTH    = 512

N_SPLITS      = 5
SVC_C         = 10.0
SVC_GAMMA     = "scale"

GENRES        = ("jazz", "metal", "pop")
GENRE_TO_IDX  = {g: i for i, g in enumerate(GENRES)}

# ---------------------------------------------------------------------------


def slice_windows(y: np.ndarray, sr: int, win_s: float, hop_s: float) -> np.ndarray:
    """Return array (n_windows, win_samples) of non-padded windows."""
    win_n = int(round(win_s * sr))
    hop_n = int(round(hop_s * sr))
    if len(y) < win_n:
        return np.empty((0, win_n), dtype=y.dtype)
    n = 1 + (len(y) - win_n) // hop_n
    out = np.empty((n, win_n), dtype=y.dtype)
    for i in range(n):
        a = i * hop_n
        out[i] = y[a : a + win_n]
    return out


def features_for_window(w: np.ndarray, sr: int) -> np.ndarray:
    """Return a 1-D feature vector for a single audio window."""
    mfcc = librosa.feature.mfcc(
        y=w, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )                                                # (N_MFCC, T)
    cent = librosa.feature.spectral_centroid(
        y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    roll = librosa.feature.spectral_rolloff(
        y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    bw   = librosa.feature.spectral_bandwidth(
        y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )[0]
    zcr  = librosa.feature.zero_crossing_rate(
        y=w, frame_length=N_FFT, hop_length=HOP_LENGTH
    )[0]

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),         # 2 * N_MFCC
        [cent.mean(), cent.std()],
        [roll.mean(), roll.std()],
        [bw.mean(),   bw.std()],
        [zcr.mean(),  zcr.std()],
    ]).astype(np.float32)
    return feats


def build_dataset(labels: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    source: "clean" or "noisy".
    Returns:
        X         (n_windows, n_features) float32
        y         (n_windows,)            int   (genre index)
        track_id  (n_windows,)            int   (row index in labels)
    """
    cache_x = CACHE_DIR / f"X_{source}.npy"
    cache_y = CACHE_DIR / f"y_{source}.npy"
    cache_t = CACHE_DIR / f"t_{source}.npy"
    if cache_x.exists() and cache_y.exists() and cache_t.exists():
        return np.load(cache_x), np.load(cache_y), np.load(cache_t)

    feats_all: list[np.ndarray] = []
    y_all:     list[int]        = []
    tid_all:   list[int]        = []

    path_col = f"{source}_path"
    for tid, row in labels.iterrows():
        wav_path = DATA_ROOT / row[path_col]
        y_audio, fs = sf.read(str(wav_path), dtype="float32")
        if y_audio.ndim > 1:
            y_audio = y_audio.mean(axis=1)
        if fs != SR:
            y_audio = librosa.resample(y_audio, orig_sr=fs, target_sr=SR)
        windows = slice_windows(y_audio, SR, WIN_S, HOP_S)
        for w in windows:
            feats_all.append(features_for_window(w, SR))
            y_all.append(GENRE_TO_IDX[row["genre"]])
            tid_all.append(int(tid))
        if (tid + 1) % 10 == 0:
            print(f"  [{source}] processed {tid + 1}/{len(labels)} tracks")

    X = np.stack(feats_all).astype(np.float32)
    y = np.asarray(y_all,  dtype=np.int64)
    t = np.asarray(tid_all, dtype=np.int64)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X); np.save(cache_y, y); np.save(cache_t, t)
    return X, y, t


def aggregate_per_track(
    proba: np.ndarray, track_ids_window: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean soft-probability across windows of the same track -> per-track pred.
    Returns (track_ids_unique_sorted, predicted_class_per_track)."""
    uniq = np.unique(track_ids_window)
    preds = np.empty(len(uniq), dtype=np.int64)
    for i, tid in enumerate(uniq):
        mask = track_ids_window == tid
        preds[i] = int(np.argmax(proba[mask].mean(axis=0)))
    return uniq, preds


def run_experiment(
    X: np.ndarray, y: np.ndarray, track_id: np.ndarray,
    track_genre: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> dict:
    """Run one matched experiment under the given fold splits.

    Splits are over UNIQUE TRACKS (indices into the per-track arrays); we
    expand them to window indices inside the loop.
    """
    fold_track_acc:  list[float] = []
    fold_window_acc: list[float] = []
    cm_total        = np.zeros((len(GENRES), len(GENRES)), dtype=np.int64)
    per_genre_correct = np.zeros(len(GENRES), dtype=np.int64)
    per_genre_total   = np.zeros(len(GENRES), dtype=np.int64)

    unique_tracks = np.unique(track_id)
    track_to_windows: dict[int, np.ndarray] = {
        int(t): np.where(track_id == t)[0] for t in unique_tracks
    }

    for k, (train_tracks_idx, test_tracks_idx) in enumerate(splits):
        train_tracks = unique_tracks[train_tracks_idx]
        test_tracks  = unique_tracks[test_tracks_idx]

        train_win = np.concatenate([track_to_windows[int(t)] for t in train_tracks])
        test_win  = np.concatenate([track_to_windows[int(t)] for t in test_tracks])

        X_tr, y_tr = X[train_win], y[train_win]
        X_te, y_te = X[test_win],  y[test_win]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(C=SVC_C, gamma=SVC_GAMMA, kernel="rbf",
                           probability=True, class_weight="balanced",
                           random_state=SEED)),
        ])
        clf.fit(X_tr, y_tr)

        # per-window accuracy (informational)
        win_pred = clf.predict(X_te)
        win_acc  = accuracy_score(y_te, win_pred)
        fold_window_acc.append(win_acc)

        # per-track accuracy (primary, via soft-prob averaging)
        proba    = clf.predict_proba(X_te)
        tids, track_pred = aggregate_per_track(proba, track_id[test_win])
        track_true = track_genre[tids]
        track_acc  = accuracy_score(track_true, track_pred)
        fold_track_acc.append(track_acc)

        cm_total += confusion_matrix(track_true, track_pred, labels=list(range(len(GENRES))))

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


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(GENRES)), GENRES)
    ax.set_yticks(range(len(GENRES)), GENRES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            ax.text(j, i, f"{int(cm[i, j])}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    np.random.seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = DATA_ROOT / "labels.csv"
    labels = pd.read_csv(labels_path).reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks from {labels_path}")
    print("Genre counts:\n" + labels["genre"].value_counts().to_string())

    print("\nExtracting features (clean) ...")
    Xc, yc, tc = build_dataset(labels, "clean")
    print(f"  X_clean: {Xc.shape}, y: {yc.shape}, tracks: {len(np.unique(tc))}")

    print("\nExtracting features (noisy) ...")
    Xn, yn, tn = build_dataset(labels, "noisy")
    print(f"  X_noisy: {Xn.shape}, y: {yn.shape}, tracks: {len(np.unique(tn))}")

    # Sanity: clean and noisy must share the same window grid (same tracks,
    # same WIN_S/HOP_S, same SR), so y and track_id arrays should be identical.
    assert np.array_equal(yc, yn), "y mismatch between clean and noisy"
    assert np.array_equal(tc, tn), "track_id mismatch between clean and noisy"

    # Per-track ground-truth genre, indexed by unique track id.
    unique_tracks = np.unique(tc)
    track_genre   = np.array([GENRE_TO_IDX[labels.loc[int(t), "genre"]]
                              for t in unique_tracks], dtype=np.int64)

    # Build StratifiedGroupKFold *over tracks*: groups=tracks themselves
    # (each unique once), labels=track_genre. This stratifies per-fold genre
    # balance at the track level, which is what we actually care about.
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre,
        groups=unique_tracks,
    ))
    print(f"\nFold sizes (test tracks per fold): "
          f"{[int(len(te)) for _, te in splits]}")

    print("\n=== Experiment A: clean -> clean ===")
    res_clean = run_experiment(Xc, yc, tc, track_genre, splits, "clean")

    print("\n=== Experiment B: noisy -> noisy ===")
    res_noisy = run_experiment(Xn, yn, tn, track_genre, splits, "noisy")

    # ---- accuracy table -----------------------------------------------------
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
    acc_df = pd.DataFrame(rows)
    acc_csv = RESULTS_DIR / "accuracy_table.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    # ---- confusion plots ----------------------------------------------------
    plot_confusion(np.array(res_clean["confusion"]),
                   "clean -> clean (track-level, summed across folds)",
                   RESULTS_DIR / "confusion_clean.png")
    plot_confusion(np.array(res_noisy["confusion"]),
                   "noisy -> noisy (track-level, summed across folds)",
                   RESULTS_DIR / "confusion_noisy.png")
    print(f"Wrote {RESULTS_DIR / 'confusion_clean.png'}")
    print(f"Wrote {RESULTS_DIR / 'confusion_noisy.png'}")

    # ---- summary.md ---------------------------------------------------------
    a = res_clean["track_acc_mean"]; sa = res_clean["track_acc_std"]
    b = res_noisy["track_acc_mean"]; sb = res_noisy["track_acc_std"]
    abs_drop = a - b
    rel_drop = (abs_drop / a) if a > 0 else float("nan")

    per_genre_drops = {
        g: res_clean["per_genre_acc"][g] - res_noisy["per_genre_acc"][g]
        for g in GENRES
    }
    worst_genre = max(per_genre_drops, key=per_genre_drops.get)

    genre_counts = labels["genre"].value_counts().to_dict()
    counts_str   = ", ".join(f"{g}={int(genre_counts.get(g, 0))}" for g in GENRES)

    summary = f"""# Genre classification: clean vs noisy

## Setup

- Dataset: {len(labels)} tracks ({counts_str}) from
  `{DATA_ROOT}`.
- Each 30 s track sliced into {WIN_S:.0f} s windows with
  {int((1 - HOP_S/WIN_S)*100)} % overlap (hop={HOP_S:.1f} s) -> ~{Xc.shape[0]//len(labels)}
  windows per track, {Xc.shape[0]} windows total.
- Features per window ({Xc.shape[1]} dims): mean+std of {N_MFCC} MFCCs +
  mean/std of spectral centroid, rolloff, bandwidth, and zero-crossing rate.
- Classifier: `StandardScaler -> SVC(rbf, C={SVC_C}, gamma={SVC_GAMMA},
  class_weight='balanced')`, fit on windows.
- Cross-validation: `StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True,
  random_state={SEED})` over tracks (groups = `track_id`), so no track
  appears in both train and test of the same fold. Both experiments use the
  **same** fold assignments to make the noise gap directly comparable.
- Per-track prediction is the **argmax of mean soft-probabilities** across
  that track's windows (soft-vote aggregation).

## Results (per-track accuracy, mean +/- std across folds)

| experiment       | accuracy            | jazz | metal | pop |
|------------------|---------------------|------|-------|-----|
| clean -> clean   | **{a:.3f} +/- {sa:.3f}** | {res_clean['per_genre_acc']['jazz']:.2f} | {res_clean['per_genre_acc']['metal']:.2f} | {res_clean['per_genre_acc']['pop']:.2f} |
| noisy -> noisy   | **{b:.3f} +/- {sb:.3f}** | {res_noisy['per_genre_acc']['jazz']:.2f} | {res_noisy['per_genre_acc']['metal']:.2f} | {res_noisy['per_genre_acc']['pop']:.2f} |

- **Absolute accuracy drop from noise:** {abs_drop:+.3f}
- **Relative accuracy drop from noise:** {rel_drop*100:+.1f} %

Per-genre accuracy drop (clean - noisy):
- jazz : {per_genre_drops['jazz']:+.2f}
- metal: {per_genre_drops['metal']:+.2f}
- pop  : {per_genre_drops['pop']:+.2f}

The genre that suffered most under contamination: **{worst_genre}**.

Window-level accuracy (informational, before per-track aggregation):
- clean -> clean: {res_clean['window_acc_mean']:.3f} +/- {res_clean['window_acc_std']:.3f}
- noisy -> noisy: {res_noisy['window_acc_mean']:.3f} +/- {res_noisy['window_acc_std']:.3f}

Confusion matrices (track-level, summed across folds): see
`confusion_clean.png` and `confusion_noisy.png`.

## Notes / assumptions

- {len(labels)} tracks is small; per-genre test counts (~{len(labels)//(len(GENRES)*N_SPLITS)}
  tracks per genre per fold) keep some jitter across seeds. The 5-fold CV at
  the track level bounds this but does not eliminate it.
- The noise model is fixed at SNR = -10 dB with mains hum, switching, broadband
  EMI, impulsive bursts, 1/f drift and 8-bit quantisation; results would
  differ at other SNRs.
- Soft-vote aggregation (mean of class probabilities) consistently equalled
  or beat hard window-vote in pilot runs, so it is used throughout.
"""
    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Wrote {summary_path}")

    # raw fold-level numbers for reference
    (RESULTS_DIR / "raw_results.json").write_text(
        json.dumps({"clean": res_clean, "noisy": res_noisy}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
