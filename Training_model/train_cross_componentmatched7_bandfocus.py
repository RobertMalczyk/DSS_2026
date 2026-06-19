"""
Cross-condition experiment H_bf: like H_pf, but the matched prefilter
is extended into a true BAND-PASS so every downstream feature (MFCC,
centroid, rolloff, bandwidth, ZCR) is computed only from the band
where Stage 7 best preserves music.

Where the band edges come from
------------------------------
Stage-7 design (no noisy-side measurement consulted here):
- HP cutoff at 25 Hz       -> set fmin to 25 Hz.
- Mains comb 50/100/150/200 Hz -> stays inside the passband but is
  notched out (same as H_pf).
- Switching tone 4410 Hz    -> set fmax just below it (default 4000 Hz)
  so the entire low-SNR / heavily Wiener-attenuated band above the
  switching tone is excluded from the feature support.

We try TWO band-pass widths to bracket the trade-off:
- LP_HZ_TIGHT = 4000  (just below switching tone; loses some brilliance
                       but keeps the cleanest Wiener passband)
- LP_HZ_WIDE  = 5500  (above switching tone, but well below the
                       broadband-EMI tail; keeps more high-frequency
                       timbre at the cost of including the Wiener-
                       attenuated 4-5 kHz region)

Both train and test audio are bandpassed identically before features.
The Wiener stage of the actual filter is amplitude-dependent and is
NOT replayed here -- on clean audio it would be near unity in the
chosen passband anyway.

Outputs:
    cache/X_clean_compmatch_bf{lp}.npy
    cache/X_filtered_componentmatched_7th_stage_compmatch_bf{lp}.npy
    results/accuracy_table_componentmatched7_bandfocus.csv
    results/confusion_clean_to_componentmatched7_bf{lp}.png
    results/raw_results_componentmatched7_bandfocus.json
    results/summary_componentmatched7_bandfocus.md
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
LP_ORDER       = 8
MAINS_FREQS_HZ = (50.0, 100.0, 150.0, 200.0)
Q_MAINS        = 35.0
SWITCHING_HZ   = 4410.0
Q_SWITCHING    = 220.0           # ~20 Hz -3 dB width at 4410 Hz

LP_HZ_TIGHT    = 4000.0          # below switching tone
LP_HZ_WIDE     = 5500.0          # above switching tone, below EMI tail

COMPONENT_DIRNAME = "Filtered ComponentMatched 7th stage"
COMPONENT_PREFIX  = "filtered_componentmatched_7th_stage_"
COMPONENT_SOURCE  = "filtered_componentmatched_7th_stage"


def build_bandpass_sos(fs: int, lp_hz: float) -> np.ndarray:
    """HP(25) + mains/switching notches + LP(lp_hz)."""
    sections = [butter(HP_ORDER, HP_CUTOFF_HZ, btype="highpass", fs=fs, output="sos")]
    for f0 in MAINS_FREQS_HZ:
        b, a = iirnotch(w0=f0, Q=Q_MAINS, fs=fs)
        sections.append(tf2sos(b, a))
    if lp_hz < SWITCHING_HZ:
        # LP already removes the switching tone; the notch is redundant.
        pass
    else:
        b, a = iirnotch(w0=SWITCHING_HZ, Q=Q_SWITCHING, fs=fs)
        sections.append(tf2sos(b, a))
    sections.append(butter(LP_ORDER, lp_hz, btype="lowpass", fs=fs, output="sos"))
    return np.vstack(sections)


def build_bandpassed_dataset(
    labels: pd.DataFrame, source: str, path_col: str,
    sos: np.ndarray, variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_x = CACHE_DIR / f"X_{source}_{variant}.npy"
    cache_y = CACHE_DIR / f"y_{source}_{variant}.npy"
    cache_t = CACHE_DIR / f"t_{source}_{variant}.npy"
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
            feats_all.append(features_for_window(w, SR))
            y_all.append(GENRE_TO_IDX[row["genre"]])
            tid_all.append(int(tid))
        if (tid + 1) % 25 == 0:
            print(f"  [{source}_{variant}] processed {tid + 1}/{len(labels)} tracks")

    X = np.stack(feats_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    t = np.asarray(tid_all, dtype=np.int64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X); np.save(cache_y, y); np.save(cache_t, t)
    return X, y, t


def run_one_band(labels_aug: pd.DataFrame, lp_hz: float,
                 splits, track_genre, tag: str) -> dict:
    sos = build_bandpass_sos(SR, lp_hz)
    print(f"\n=== Bandfocus LP={lp_hz:.0f} Hz, SOS shape {sos.shape} ===")

    variant = f"compmatch_bf{int(lp_hz)}"

    print(f"Extracting clean ({variant}) ...")
    Xc, yc, tc = build_bandpassed_dataset(labels_aug, "clean", "clean_path", sos, variant)
    print(f"Extracting {COMPONENT_SOURCE} ({variant}) ...")
    Xh, yh, th = build_bandpassed_dataset(
        labels_aug, COMPONENT_SOURCE, f"{COMPONENT_SOURCE}_path", sos, variant,
    )
    assert np.array_equal(yc, yh), f"y mismatch ({variant})"
    assert np.array_equal(tc, th), f"track_id mismatch ({variant})"

    print(f"--- Anchor A_{tag}: train clean_{variant} -> test clean_{variant} ---")
    res_a = run_cross(Xc, Xc, yc, tc, track_genre, splits, label=f"clean_{tag}->clean_{tag}")

    print(f"--- Experiment H_{tag}: train clean_{variant} -> test 7th_{variant} ---")
    res_h = run_cross(Xc, Xh, yc, tc, track_genre, splits, label=f"clean_{tag}->7th_{tag}")

    return {"variant": variant, "lp_hz": lp_hz, "anchor": res_a, "cross": res_h}


def main() -> int:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")

    labels_aug = labels.copy()
    labels_aug[f"{COMPONENT_SOURCE}_path"] = labels_aug["file_id"].apply(
        lambda fid: f"{COMPONENT_DIRNAME}/{COMPONENT_PREFIX}{fid}.wav"
    )

    unique_tracks_full = labels.index.to_numpy()
    track_genre_full   = np.array(
        [GENRE_TO_IDX[g] for g in labels["genre"]], dtype=np.int64,
    )
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks_full), 1)),
        y=track_genre_full, groups=unique_tracks_full,
    ))

    results = []
    for lp_hz, tag in [(LP_HZ_TIGHT, "bf4000"), (LP_HZ_WIDE, "bf5500")]:
        results.append(run_one_band(labels_aug, lp_hz, splits, track_genre_full, tag))

    rows = []
    for r in results:
        for kind, res in [("anchor_clean->clean", r["anchor"]),
                          ("cross_clean->7th",   r["cross"])]:
            rows.append({
                "variant":         r["variant"],
                "lp_hz":           r["lp_hz"],
                "experiment":      kind,
                "mean_acc":        res["track_acc_mean"],
                "std_acc":         res["track_acc_std"],
                "window_mean_acc": res["window_acc_mean"],
                "window_std_acc":  res["window_acc_std"],
                "wrong_total":     res["wrong_total"],
                "total":           res["total"],
                **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
            })
    acc_df = pd.DataFrame(rows)
    acc_csv = RESULTS_DIR / "accuracy_table_componentmatched7_bandfocus.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    for r in results:
        plot_confusion(
            np.array(r["cross"]["confusion"]),
            f"clean_{r['variant']} -> 7th_{r['variant']} (track-level)",
            RESULTS_DIR / f"confusion_clean_to_componentmatched7_{r['variant']}.png",
        )

    A_acc = 0.988
    C_acc = 0.344
    G_acc = 0.492
    H_acc = 0.560
    H_pf  = 0.572

    summary_lines = ["# Cross-condition (Stage-7 audio) with band-pass-focused features",
                     "",
                     "Same SVM, folds, and aggregation as the rest of the baseline.",
                     "Both train (clean) and test (Stage-7 output) are run through the",
                     f"same matched band-pass prefilter (HP {HP_CUTOFF_HZ:.0f} Hz, "
                     f"mains notches {list(MAINS_FREQS_HZ)} Hz, optional switching "
                     f"notch at {SWITCHING_HZ:.0f} Hz, LP at the band edge) BEFORE "
                     "feature extraction.",
                     "",
                     "## Results",
                     "",
                     "| variant | LP (Hz) | A_bf (clean->clean) | H_bf (clean->7th) | wrong/250 | jazz | metal | pop |",
                     "|---------|---------|---------------------|-------------------|-----------|------|-------|-----|"]
    for r in results:
        a = r["anchor"]; c = r["cross"]
        summary_lines.append(
            f"| {r['variant']} | {int(r['lp_hz'])} | "
            f"{a['track_acc_mean']:.3f} +/- {a['track_acc_std']:.3f} | "
            f"**{c['track_acc_mean']:.3f} +/- {c['track_acc_std']:.3f}** | "
            f"{c['wrong_total']} | "
            f"{c['per_genre_acc']['jazz']:.2f} | "
            f"{c['per_genre_acc']['metal']:.2f} | "
            f"{c['per_genre_acc']['pop']:.2f} |"
        )
    summary_lines += [
        "",
        "## Placement against frozen baseline",
        "",
        f"- A clean -> clean                          : {A_acc:.3f}  (upper bound)",
        f"- C clean -> noisy                          : {C_acc:.3f}  (untreated floor)",
        f"- G clean -> adaptivems_6th                 : {G_acc:.3f}",
        f"- H clean -> componentmatched_7th           : {H_acc:.3f}  (no feature change)",
        f"- H_pf clean_pf -> componentmatched_7th_pf  : {H_pf:.3f}  (linear prefilter only)",
    ]
    for r in results:
        h = r["cross"]["track_acc_mean"]
        rec = (h - C_acc) / (A_acc - C_acc)
        summary_lines.append(
            f"- **H_{r['variant'].split('_')[-1]} clean -> 7th (LP {int(r['lp_hz'])} Hz) : "
            f"{h:.3f}**  (recovery {rec:.1%}, delta vs H = {h - H_acc:+.3f})"
        )
    summary_lines += [
        "",
        "## Leakage status",
        "",
        "Same StratifiedGroupKFold splits as A/B/C/D/E/F/G/H/H_pf.",
        "Audit: `verify_no_leakage_componentmatched7_bandfocus.py`.",
    ]

    (RESULTS_DIR / "summary_componentmatched7_bandfocus.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8",
    )
    (RESULTS_DIR / "raw_results_componentmatched7_bandfocus.json").write_text(
        json.dumps([{
            "variant": r["variant"], "lp_hz": r["lp_hz"],
            "anchor_clean_to_clean": r["anchor"],
            "cross_clean_to_7th":    r["cross"],
        } for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / 'summary_componentmatched7_bandfocus.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
