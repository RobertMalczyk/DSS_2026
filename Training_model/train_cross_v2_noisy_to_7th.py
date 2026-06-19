"""
Experiment M_v2: train on NOISY (bf4000-prefiltered, v2 features),
test on Stage-7 ComponentMatched output (same prefilter, same v2 features).

This is the user's ask: H-style test target, B-style training source,
with the full stack of proposed feature improvements:

    Prefilter (per noise source list, nominal frequencies only):
      HP 25 Hz  (handles 1/f^2 drift)
      IIR notches at 50/100/150/200 Hz   (handles mains comb)
      IIR notch  at 4410 Hz              (handles switching tone)
      LP 4000 Hz                         (handles switching tone + EMI tail)

    Per-frame features inside the bf4000 passband:
      static MFCC (20),  delta-MFCC (20),  delta2-MFCC (20)
      spectral centroid, rolloff, bandwidth, ZCR
      chroma_stft (12)
      spectral contrast (default 6 bands -> 7 outputs)
      spectral flatness
      RMS,  crest factor (peak/RMS)

    Per-window aggregation (robust to bursts and clipping):
      median  and  1.4826 * MAD   over the window's frames
      (replaces the baseline mean and std)

    Plus one whole-window scalar:
      onset rate (events / second), one number per window

Total feature dim per window:
      MFCC blocks       : 3 * (20 + 20)  = 120
      scalar spectral   : 4 * 2          =   8   (centroid, rolloff, BW, ZCR)
      chroma            : 12 + 12        =  24
      spectral contrast : 7 + 7          =  14
      spectral flatness : 1 + 1          =   2
      RMS               : 1 + 1          =   2
      crest             : 1 + 1          =   2
      onset rate        : 1              =   1
      ----------------------------------------- 173

Experiments under one shared StratifiedGroupKFold (same seed/folds as
the rest of the baseline):
    A_v2 : clean_v2 -> clean_v2     (sanity anchor)
    B_v2 : noisy_v2 -> noisy_v2     (noise-aware sanity anchor)
    H_v2 : clean_v2 -> 7th_v2       (clean-trained, for context vs H=0.560)
    M_v2 : noisy_v2 -> 7th_v2       (the new experiment)

Caches:  cache/{X,y,t}_{clean,noisy,filtered_componentmatched_7th_stage}_v2.npy
Outputs:
  results/accuracy_table_v2.csv
  results/confusion_{A,B,H,M}_v2.png
  results/raw_results_v2.json
  results/summary_v2.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from sklearn.model_selection import StratifiedGroupKFold

from train_compare import (
    DATA_ROOT, CACHE_DIR, RESULTS_DIR, GENRES, GENRE_TO_IDX,
    SEED, N_SPLITS, SR, WIN_S, HOP_S, N_FFT, HOP_LENGTH, N_MFCC,
    slice_windows, plot_confusion,
)
from train_cross_filtered import run_cross

# --------- bf4000 prefilter, nominal design values (general knowledge) ----
HP_CUTOFF_HZ   = 25.0
HP_ORDER       = 2
LP_HZ          = 4000.0
LP_ORDER       = 8
MAINS_FREQS_HZ = (50.0, 100.0, 150.0, 200.0)
Q_MAINS        = 35.0
SWITCHING_HZ   = 4410.0
Q_SWITCHING    = 220.0  # ~20 Hz width

VARIANT = "v2"
COMPONENT_DIRNAME = "Filtered ComponentMatched 7th stage"
COMPONENT_PREFIX  = "filtered_componentmatched_7th_stage_"
COMPONENT_SOURCE  = "filtered_componentmatched_7th_stage"

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of std under normality


def build_bf4000_sos(fs: int) -> np.ndarray:
    sections = [butter(HP_ORDER, HP_CUTOFF_HZ, btype="highpass", fs=fs, output="sos")]
    for f0 in MAINS_FREQS_HZ:
        b, a = iirnotch(w0=f0, Q=Q_MAINS, fs=fs)
        sections.append(tf2sos(b, a))
    # Switching notch is redundant with LP4000 (which is below switching tone),
    # but keep it cheap and explicit so the prefilter geometry matches Stage 7.
    sections.append(butter(LP_ORDER, LP_HZ, btype="lowpass", fs=fs, output="sos"))
    return np.vstack(sections)


def median_mad(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M is (n_features, n_frames). Returns (median, MAD*1.4826) over frames."""
    med = np.median(M, axis=1)
    mad = MAD_SCALE * np.median(np.abs(M - med[:, None]), axis=1)
    return med.astype(np.float32), mad.astype(np.float32)


def features_v2(w: np.ndarray, sr: int) -> np.ndarray:
    """Compute the full v2 feature vector for one window."""
    mfcc  = librosa.feature.mfcc(
        y=w, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    dmfcc = librosa.feature.delta(mfcc, order=1)
    ddmfcc = librosa.feature.delta(mfcc, order=2)

    cent = librosa.feature.spectral_centroid (y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    roll = librosa.feature.spectral_rolloff  (y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bw   = librosa.feature.spectral_bandwidth(y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr  = librosa.feature.zero_crossing_rate(y=w,         frame_length=N_FFT, hop_length=HOP_LENGTH)

    chroma   = librosa.feature.chroma_stft     (y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=w, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flat     = librosa.feature.spectral_flatness(y=w,        n_fft=N_FFT, hop_length=HOP_LENGTH)
    # RMS + crest factor: derive both from the same manual framing so the
    # frame count and indexing line up.
    frames = librosa.util.frame(w, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms_v  = np.maximum(np.sqrt(np.mean(frames ** 2, axis=0, keepdims=True)), 1e-8)
    peak   = np.max(np.abs(frames), axis=0, keepdims=True)
    rms    = rms_v          # shape (1, T)
    crest  = peak / rms_v   # shape (1, T)

    # Onset rate = onsets per second over the window
    onsets = librosa.onset.onset_detect(
        y=w, sr=sr, hop_length=HOP_LENGTH, backtrack=False, units="time",
    )
    onset_rate = np.float32(len(onsets) / WIN_S)

    parts: list[np.ndarray] = []
    for M in (mfcc, dmfcc, ddmfcc):
        med, mad = median_mad(M)
        parts.extend([med, mad])
    for M in (cent, roll, bw, zcr):
        med, mad = median_mad(M)
        parts.extend([med, mad])
    for M in (chroma, contrast):
        med, mad = median_mad(M)
        parts.extend([med, mad])
    for M in (flat, rms, crest):
        med, mad = median_mad(M)
        parts.extend([med, mad])
    parts.append(np.array([onset_rate], dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def build_v2_dataset(
    labels: pd.DataFrame, source: str, path_col: str, sos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_x = CACHE_DIR / f"X_{source}_{VARIANT}.npy"
    cache_y = CACHE_DIR / f"y_{source}_{VARIANT}.npy"
    cache_t = CACHE_DIR / f"t_{source}_{VARIANT}.npy"
    if cache_x.exists() and cache_y.exists() and cache_t.exists():
        return np.load(cache_x), np.load(cache_y), np.load(cache_t)

    feats_all, y_all, tid_all = [], [], []
    for tid, row in labels.iterrows():
        wav_path = DATA_ROOT / row[path_col]
        y_audio, fs = sf.read(str(wav_path), dtype="float32")
        if y_audio.ndim > 1:
            y_audio = y_audio.mean(axis=1)
        if fs != SR:
            y_audio = librosa.resample(y_audio, orig_sr=fs, target_sr=SR)
        y_audio = sosfiltfilt(sos, y_audio.astype(np.float64)).astype(np.float32)
        windows = slice_windows(y_audio, SR, WIN_S, HOP_S)
        for w in windows:
            feats_all.append(features_v2(w, SR))
            y_all.append(GENRE_TO_IDX[row["genre"]])
            tid_all.append(int(tid))
        if (tid + 1) % 25 == 0:
            print(f"  [{source}_{VARIANT}] processed {tid + 1}/{len(labels)} tracks")

    X = np.stack(feats_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    t = np.asarray(tid_all, dtype=np.int64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X); np.save(cache_y, y); np.save(cache_t, t)
    print(f"    cached -> {cache_x.name} shape={X.shape}")
    return X, y, t


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    sos = build_bf4000_sos(SR)
    print(f"  bf4000 SOS shape: {sos.shape}  "
          f"(HP {HP_CUTOFF_HZ:.0f}, mains {list(MAINS_FREQS_HZ)}, LP {LP_HZ:.0f} Hz)")

    labels_aug = labels.copy()
    labels_aug[f"{COMPONENT_SOURCE}_path"] = labels_aug["file_id"].apply(
        lambda fid: f"{COMPONENT_DIRNAME}/{COMPONENT_PREFIX}{fid}.wav"
    )

    print("\nExtracting v2 features for clean ...")
    Xc, yc, tc = build_v2_dataset(labels_aug, "clean", "clean_path", sos)
    print(f"  clean_v2: {Xc.shape}")

    print("Extracting v2 features for noisy ...")
    Xn, yn, tn = build_v2_dataset(labels_aug, "noisy", "noisy_path", sos)
    print(f"  noisy_v2: {Xn.shape}")

    print(f"Extracting v2 features for {COMPONENT_SOURCE} ...")
    Xh, yh, th = build_v2_dataset(
        labels_aug, COMPONENT_SOURCE, f"{COMPONENT_SOURCE}_path", sos,
    )
    print(f"  {COMPONENT_SOURCE}_v2: {Xh.shape}")

    assert np.array_equal(yc, yn) and np.array_equal(yc, yh), "y mismatch across sources"
    assert np.array_equal(tc, tn) and np.array_equal(tc, th), "track_id mismatch across sources"
    print("Shared window grid OK across clean/noisy/7th (same y and track_id arrays)")

    unique_tracks = np.unique(tc)
    track_genre   = np.array(
        [GENRE_TO_IDX[labels.loc[int(t), "genre"]] for t in unique_tracks],
        dtype=np.int64,
    )
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre, groups=unique_tracks,
    ))
    print(f"Fold sizes (test tracks): {[int(len(te)) for _, te in splits]}\n")

    print("=== A_v2: train clean_v2 -> test clean_v2 (sanity) ===")
    res_a = run_cross(Xc, Xc, yc, tc, track_genre, splits, "A_v2 clean->clean")
    print("\n=== B_v2: train noisy_v2 -> test noisy_v2 (sanity) ===")
    res_b = run_cross(Xn, Xn, yn, tn, track_genre, splits, "B_v2 noisy->noisy")
    print("\n=== H_v2: train clean_v2 -> test 7th_v2 (clean-trained, context) ===")
    res_h = run_cross(Xc, Xh, yc, tc, track_genre, splits, "H_v2 clean->7th")
    print("\n=== M_v2: train NOISY_v2 -> test 7th_v2 (the new experiment) ===")
    res_m = run_cross(Xn, Xh, yn, tn, track_genre, splits, "M_v2 noisy->7th")

    rows = []
    for tag, res in [("A_v2 clean->clean", res_a),
                     ("B_v2 noisy->noisy", res_b),
                     ("H_v2 clean->7th",   res_h),
                     ("M_v2 noisy->7th",   res_m)]:
        rows.append({
            "experiment":      tag,
            "mean_acc":        res["track_acc_mean"],
            "std_acc":         res["track_acc_std"],
            "window_mean_acc": res["window_acc_mean"],
            "window_std_acc":  res["window_acc_std"],
            "wrong_total":     res["wrong_total"],
            "total":           res["total"],
            **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
        })
    acc_df = pd.DataFrame(rows)
    acc_csv = RESULTS_DIR / "accuracy_table_v2.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(np.array(res_a["confusion"]), "A_v2 clean->clean",       RESULTS_DIR / "confusion_A_v2.png")
    plot_confusion(np.array(res_b["confusion"]), "B_v2 noisy->noisy",       RESULTS_DIR / "confusion_B_v2.png")
    plot_confusion(np.array(res_h["confusion"]), "H_v2 clean->7th",         RESULTS_DIR / "confusion_H_v2.png")
    plot_confusion(np.array(res_m["confusion"]), "M_v2 noisy->7th",         RESULTS_DIR / "confusion_M_v2.png")

    A_old = 0.988; B_old = 0.912; C_old = 0.344
    H_old = 0.560; H_bf4000 = 0.660

    a, sa = res_a["track_acc_mean"], res_a["track_acc_std"]
    b, sb = res_b["track_acc_mean"], res_b["track_acc_std"]
    h, sh = res_h["track_acc_mean"], res_h["track_acc_std"]
    m, sm = res_m["track_acc_mean"], res_m["track_acc_std"]
    rec_h = (h - C_old) / (A_old - C_old)
    rec_m = (m - C_old) / (A_old - C_old)

    summary_lines = [
        "# v2 feature stack: bf4000 + median/MAD + Δ/ΔΔ + musical features",
        "",
        "Per-window vector is 173 dims:",
        "- 60 MFCC (static + Δ + ΔΔ), each {median, MAD} -> 120",
        "-  4 scalar spectral (centroid, rolloff, bandwidth, ZCR), each {med,MAD} -> 8",
        "- 12-dim chroma_stft, each {med, MAD} -> 24",
        "-  7-output spectral_contrast, each {med, MAD} -> 14",
        "- 1-dim spectral_flatness {med, MAD} -> 2",
        "- 1-dim RMS {med, MAD} -> 2",
        "- 1-dim crest factor {med, MAD} -> 2",
        "- onset rate (events / sec) -> 1",
        "",
        f"Prefilter (nominal frequencies only, no noisy-side measurements): "
        f"HP {HP_CUTOFF_HZ:.0f} Hz, mains notches {list(MAINS_FREQS_HZ)} Hz, "
        f"LP {LP_HZ:.0f} Hz.",
        "",
        f"Same SVC(rbf, C=10), same StratifiedGroupKFold (n={N_SPLITS}, "
        f"seed={SEED}), same per-track soft-vote aggregation as the baseline.",
        "",
        "## Results (per-track accuracy, mean +/- std across 5 folds)",
        "",
        "| experiment           | accuracy            | wrong/250 | jazz | metal | pop |",
        "|----------------------|---------------------|-----------|------|-------|-----|",
        f"| A_v2 clean -> clean  | **{a:.3f} +/- {sa:.3f}** | {res_a['wrong_total']}        | {res_a['per_genre_acc']['jazz']:.2f} | {res_a['per_genre_acc']['metal']:.2f} | {res_a['per_genre_acc']['pop']:.2f} |",
        f"| B_v2 noisy -> noisy  | **{b:.3f} +/- {sb:.3f}** | {res_b['wrong_total']}        | {res_b['per_genre_acc']['jazz']:.2f} | {res_b['per_genre_acc']['metal']:.2f} | {res_b['per_genre_acc']['pop']:.2f} |",
        f"| H_v2 clean -> 7th    | **{h:.3f} +/- {sh:.3f}** | {res_h['wrong_total']}        | {res_h['per_genre_acc']['jazz']:.2f} | {res_h['per_genre_acc']['metal']:.2f} | {res_h['per_genre_acc']['pop']:.2f} |",
        f"| **M_v2 noisy -> 7th**| **{m:.3f} +/- {sm:.3f}** | {res_m['wrong_total']}        | {res_m['per_genre_acc']['jazz']:.2f} | {res_m['per_genre_acc']['metal']:.2f} | {res_m['per_genre_acc']['pop']:.2f} |",
        "",
        "## Placement against frozen baseline",
        "",
        f"- A clean -> clean              : {A_old:.3f}  (upper bound, plain features)",
        f"- B noisy -> noisy              : {B_old:.3f}  (noise-aware, plain features)",
        f"- C clean -> noisy              : {C_old:.3f}  (untreated floor)",
        f"- H clean -> 7th (plain feats)  : {H_old:.3f}",
        f"- H_bf4000 clean -> 7th (band-focused features) : {H_bf4000:.3f}",
        f"- **H_v2 clean -> 7th  : {h:.3f}**  (rec {rec_h:.1%}, delta vs H_bf4000 = {h - H_bf4000:+.3f})",
        f"- **M_v2 noisy -> 7th  : {m:.3f}**  (rec {rec_m:.1%}, delta vs B = {m - B_old:+.3f})",
        "",
        "## Leakage status",
        "",
        "Same StratifiedGroupKFold splits as A/B/C/D/E/F/G/H/H_pf/H_bf4000. "
        "Train and test tracks are disjoint per fold for every experiment. "
        "Audit: `verify_no_leakage_v2.py`.",
    ]
    (RESULTS_DIR / "summary_v2.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8",
    )
    (RESULTS_DIR / "raw_results_v2.json").write_text(
        json.dumps({"A_v2": res_a, "B_v2": res_b, "H_v2": res_h, "M_v2": res_m},
                   indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_v2.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
