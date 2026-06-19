"""
Cross-condition experiment H_pf: like H (clean -> componentmatched_7th)
but the FEATURE EXTRACTOR is matched to the Stage-7 linear prefilter.

Motivation
----------
Stage-7's linear chain is: zero-phase Butterworth HP at 25 Hz, plus
narrow IIR notches at the mains comb (~50/100/150/200 Hz) and the
switching tone (~4410 Hz). The clean-trained baseline H = 0.560
suffers because clean MFCCs see narrowband peaks at those frequencies
that the filter has carved out of the test audio. We close that
specific domain shift by applying the SAME linear prefilter to the
CLEAN waveform before extracting features, so train and test features
both come from spectra with the same five carve-outs.

Wiener (Stage-7's stage D) is amplitude-dependent and not a fixed
linear filter, so it is not replayed here -- on already-clean audio
the Wiener gain is near unity in high-SNR bins anyway.

Important: the prefilter geometry uses NOMINAL design frequencies
(25 Hz cutoff; 50/100/150/200 Hz mains comb; 4410 Hz switching tone),
not values detected from noisy audio. We never read noisy/*.wav here.

Outputs:
    cache/X_clean_compmatch.npy
    cache/X_filtered_componentmatched_7th_stage_compmatch.npy
    results/accuracy_table_componentmatched7_prefilter.csv
    results/confusion_clean_to_componentmatched7_prefilter.png
    results/raw_results_componentmatched7_prefilter.json
    results/summary_componentmatched7_prefilter.md
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
    SEED, N_SPLITS, SR, WIN_S, HOP_S,
    slice_windows, features_for_window, plot_confusion,
)
from train_cross_filtered import run_cross

# ---------------- Stage-7 linear prefilter, nominal design values ----------
HP_CUTOFF_HZ   = 25.0
HP_ORDER       = 2
MAINS_FREQS_HZ = (50.0, 100.0, 150.0, 200.0)
Q_MAINS        = 35.0
SWITCHING_HZ   = 4410.0
Q_SWITCHING    = 220.0           # ~20 Hz -3 dB width at 4410 Hz

VARIANT        = "compmatch"
COMPONENT_DIRNAME = "Filtered ComponentMatched 7th stage"
COMPONENT_PREFIX  = "filtered_componentmatched_7th_stage_"
COMPONENT_SOURCE  = "filtered_componentmatched_7th_stage"


def build_prefilter_sos(fs: int) -> np.ndarray:
    """Build SOS for HP(25) + notches at the nominal design frequencies."""
    sections = [butter(HP_ORDER, HP_CUTOFF_HZ, btype="highpass", fs=fs, output="sos")]
    for f0 in MAINS_FREQS_HZ:
        b, a = iirnotch(w0=f0, Q=Q_MAINS, fs=fs)
        sections.append(tf2sos(b, a))
    b, a = iirnotch(w0=SWITCHING_HZ, Q=Q_SWITCHING, fs=fs)
    sections.append(tf2sos(b, a))
    return np.vstack(sections)


def build_prefiltered_dataset(
    labels: pd.DataFrame, source: str, path_col: str, sos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror train_compare.build_dataset but apply `sos` (zero-phase) to
    each waveform before window slicing. Cache key gets a `_{VARIANT}` suffix."""
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
        # Apply matched prefilter (zero-phase) before windowing.
        y_audio = sosfiltfilt(sos, y_audio.astype(np.float64)).astype(np.float32)
        windows = slice_windows(y_audio, SR, WIN_S, HOP_S)
        for w in windows:
            feats_all.append(features_for_window(w, SR))
            y_all.append(GENRE_TO_IDX[row["genre"]])
            tid_all.append(int(tid))
        if (tid + 1) % 25 == 0:
            print(f"  [{source}_{VARIANT}] processed {tid + 1}/{len(labels)} tracks")

    X = np.stack(feats_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    t = np.asarray(tid_all, dtype=np.int64)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X); np.save(cache_y, y); np.save(cache_t, t)
    return X, y, t


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    sos = build_prefilter_sos(SR)
    print(f"  prefilter SOS shape: {sos.shape}  "
          f"(HP {HP_CUTOFF_HZ:.0f} Hz, mains {MAINS_FREQS_HZ}, sw {SWITCHING_HZ:.0f} Hz)")

    # Synthesise the path column for the 7th-stage source so we don't need
    # to touch labels.csv schema.
    labels_aug = labels.copy()
    labels_aug[f"{COMPONENT_SOURCE}_path"] = labels_aug["file_id"].apply(
        lambda fid: f"{COMPONENT_DIRNAME}/{COMPONENT_PREFIX}{fid}.wav"
    )

    print("Extracting prefiltered clean features ...")
    Xc, yc, tc = build_prefiltered_dataset(
        labels_aug, "clean", "clean_path", sos,
    )
    print(f"  clean_{VARIANT}: {Xc.shape}")

    print(f"Extracting prefiltered {COMPONENT_SOURCE} features ...")
    Xh, yh, th = build_prefiltered_dataset(
        labels_aug, COMPONENT_SOURCE, f"{COMPONENT_SOURCE}_path", sos,
    )
    print(f"  {COMPONENT_SOURCE}_{VARIANT}: {Xh.shape}")

    assert np.array_equal(yc, yh), "y mismatch (clean vs filtered_7th, prefiltered)"
    assert np.array_equal(tc, th), "track_id mismatch (clean vs filtered_7th, prefiltered)"
    print("  shared window grid OK")

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

    # ---- Sanity anchor: A' (clean_pf -> clean_pf) should still be near 0.99
    print("=== Anchor A' : train CLEAN_pf -> test CLEAN_pf (sanity) ===")
    res_a = run_cross(
        X_train_src=Xc, X_test_src=Xc,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label="clean_pf->clean_pf",
    )

    print("\n=== Experiment H_pf: train CLEAN_pf -> test FILTERED_7TH_pf ===")
    res_h = run_cross(
        X_train_src=Xc, X_test_src=Xh,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label="clean_pf->filtered_7th_pf",
    )

    rows = []
    for tag, res in [("clean_pf_vs_clean_pf", res_a),
                     ("clean_pf_train_componentmatched_7th_pf_test", res_h)]:
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
    acc_df  = pd.DataFrame(rows)
    acc_csv = RESULTS_DIR / "accuracy_table_componentmatched7_prefilter.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(
        np.array(res_h["confusion"]),
        "train clean_pf -> test componentmatched_7th_pf (track-level)",
        RESULTS_DIR / "confusion_clean_to_componentmatched7_prefilter.png",
    )

    # ---- Anchors / commentary
    A_acc = 0.988
    C_acc = 0.344
    G_acc = 0.492
    H_acc = 0.560

    h, sh = res_h["track_acc_mean"], res_h["track_acc_std"]
    a, sa = res_a["track_acc_mean"], res_a["track_acc_std"]
    recovery = ((h - C_acc) / (A_acc - C_acc)) if A_acc > C_acc else float("nan")
    delta_h  = h - H_acc

    summary = f"""# Cross-condition: train CLEAN_pf -> test COMPONENTMATCHED_7TH_pf

Same windowing, classifier, folds, and aggregation as the rest of the
baseline. Only the **feature extractor** changes: each waveform is run
through a fixed linear prefilter that mirrors the linear stages of the
Stage-7 denoiser, then the standard 48-dim MFCC + spectral features are
extracted from the prefiltered waveform.

Prefilter (nominal design frequencies, no noisy-side detection):
- Zero-phase Butterworth HP, cutoff {HP_CUTOFF_HZ:.0f} Hz, order {HP_ORDER}
- IIR notches at the mains comb {list(MAINS_FREQS_HZ)} Hz, Q={Q_MAINS}
- IIR notch at the switching tone {SWITCHING_HZ:.0f} Hz, Q={Q_SWITCHING}

(Stage 7's adaptive Wiener is not replayed; on already-clean audio its
gain is near unity in high-SNR bins.)

## Results (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment                                                | accuracy            | wrong / {res_h['total']} | jazz | metal | pop |
|-----------------------------------------------------------|---------------------|-----------------------|------|-------|-----|
| A_pf : train clean_pf, test clean_pf (sanity)             | **{a:.3f} +/- {sa:.3f}** | {res_a['wrong_total']}                   | {res_a['per_genre_acc']['jazz']:.2f} | {res_a['per_genre_acc']['metal']:.2f} | {res_a['per_genre_acc']['pop']:.2f} |
| H_pf : train clean_pf, test componentmatched_7th_pf       | **{h:.3f} +/- {sh:.3f}** | {res_h['wrong_total']}                   | {res_h['per_genre_acc']['jazz']:.2f} | {res_h['per_genre_acc']['metal']:.2f} | {res_h['per_genre_acc']['pop']:.2f} |

Window-level accuracy (H_pf): {res_h['window_acc_mean']:.3f} +/- {res_h['window_acc_std']:.3f}
Confusion matrix: `confusion_clean_to_componentmatched7_prefilter.png`.

## Placement against frozen baseline

- A clean -> clean                       : 0.988  (upper bound)
- C clean -> noisy                       : 0.344  (untreated floor)
- G clean -> adaptivems_6th              : {G_acc:.3f}
- H clean -> componentmatched_7th        : {H_acc:.3f}  (no feature change)
- **H_pf clean_pf -> componentmatched_7th_pf : {h:.3f}**  (Stage-7-matched features)

Recovery fraction of the A-C gap closed by Stage-7 + matched features:
`(H_pf - C) / (A - C) = ({h:.3f} - {C_acc}) / ({A_acc} - {C_acc}) = {recovery:.2%}`

Delta vs H (same Stage-7 audio, plain features): `H_pf - H = {delta_h:+.3f}`.

## Leakage status

For each fold, train_tracks and test_tracks are disjoint (asserted in-loop).
Train windows come from clean/ (prefiltered), test windows come from the
7th-stage filter output (prefiltered), but the underlying track IDs never
overlap. Full audit: `verify_no_leakage_componentmatched7_prefilter.py`.
"""
    (RESULTS_DIR / "summary_componentmatched7_prefilter.md").write_text(
        summary, encoding="utf-8"
    )
    (RESULTS_DIR / "raw_results_componentmatched7_prefilter.json").write_text(
        json.dumps({
            "clean_pf_vs_clean_pf": res_a,
            "clean_pf_to_filtered_componentmatched_7th_stage_pf": res_h,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_componentmatched7_prefilter.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
