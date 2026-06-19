"""
Cross-condition experiments E' and F' under a 3 kHz feature-bandlimit.

For each audio waveform (clean, filtered_1st_stage, filtered_2nd_stage)
a zero-phase 8th-order Butterworth low-pass filter at 3 kHz is applied
*before* the same 3 s windowing and the same 48-dim MFCC+spectral
feature extractor used in the frozen baseline. This guarantees that no
feature depends on spectral content above 3 kHz (on either the train or
the test side).

Experiments (both train on bandlimited clean):
    E':  clean_fcutoff3khz  ->  filtered_1st_stage_fcutoff3khz
    F':  clean_fcutoff3khz  ->  filtered_2nd_stage_fcutoff3khz

Pipeline is otherwise unchanged from train_compare.py:
    - 48-dim feature vector per 3 s window (50 % overlap)
    - StandardScaler -> SVC(rbf, C=10, gamma='scale',
                            class_weight='balanced', probability=True)
    - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    - Per-track aggregation via mean soft probabilities

Caches (new):
    cache/X_clean_fcutoff3khz.npy              (y_, t_)
    cache/X_filtered_1st_stage_fcutoff3khz.npy (y_, t_)
    cache/X_filtered_2nd_stage_fcutoff3khz.npy (y_, t_)

Outputs (separate filenames, do not overwrite earlier results):
    results/accuracy_table_fcutoff3khz.csv
    results/confusion_fcutoff3khz_clean_to_filtered_1st.png
    results/confusion_fcutoff3khz_clean_to_filtered_2nd.png
    results/raw_results_fcutoff3khz.json
    results/summary_fcutoff3khz.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold

from train_compare import (
    DATA_ROOT, CACHE_DIR, RESULTS_DIR, GENRES, GENRE_TO_IDX,
    SEED, N_SPLITS, SR, WIN_S, HOP_S,
    slice_windows, features_for_window, plot_confusion,
)
from train_cross_filtered import run_cross
from train_cross_filtered_1st import FILTERED_1ST_DIR
from train_cross_filtered_2nd import FILTERED_2ND_DIRNAME

# ----------------------------- hyperparameters -----------------------------
LPF_CUTOFF_HZ = 3000.0
LPF_ORDER     = 8
CUTOFF_TAG    = "fcutoff3khz"   # cache-key suffix + filename tag
# ---------------------------------------------------------------------------


def _lpf_sos() -> np.ndarray:
    # sample-rate-aware SOS form -- stable for order 8 unlike ba form
    return butter(
        LPF_ORDER, LPF_CUTOFF_HZ, btype="lowpass", fs=SR, output="sos"
    )


def lowpass(y: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth LPF at LPF_CUTOFF_HZ. Output length == input."""
    return sosfiltfilt(_lpf_sos(), y).astype(np.float32, copy=False)


def build_bandlimited_dataset(
    labels: pd.DataFrame,
    path_resolver: Callable[[str], Path],
    cache_key: str,
    source_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read WAVs via path_resolver(file_id), LPF, window, extract features.

    Re-uses `slice_windows` and `features_for_window` from train_compare
    so the feature dimensionality/layout matches the baseline exactly.
    Results are cached under cache/X_{cache_key}.npy (plus y_, t_)."""
    cx = CACHE_DIR / f"X_{cache_key}.npy"
    cy = CACHE_DIR / f"y_{cache_key}.npy"
    ct = CACHE_DIR / f"t_{cache_key}.npy"
    if cx.exists() and cy.exists() and ct.exists():
        return np.load(cx), np.load(cy), np.load(ct)

    feats: list[np.ndarray] = []
    y_all: list[int]        = []
    t_all: list[int]        = []
    for tid, row in labels.iterrows():
        wav_path = path_resolver(row["file_id"])
        if not wav_path.is_file():
            raise FileNotFoundError(f"missing {source_label} file: {wav_path}")
        audio, fs = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if fs != SR:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=SR)
        audio = lowpass(audio)
        for w in slice_windows(audio, SR, WIN_S, HOP_S):
            feats.append(features_for_window(w, SR))
            y_all.append(GENRE_TO_IDX[row["genre"]])
            t_all.append(int(tid))
        if (tid + 1) % 25 == 0:
            print(f"  [{source_label}] LPF+features: {tid + 1}/{len(labels)} tracks")

    X = np.stack(feats).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    t = np.asarray(t_all, dtype=np.int64)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cx, X); np.save(cy, y); np.save(ct, t)
    return X, y, t


# ----- path resolvers (one per audio source) --------------------------------
def resolve_clean(fid: str) -> Path:
    return DATA_ROOT / "clean" / f"{fid}.wav"


def resolve_filtered_1st(fid: str) -> Path:
    return FILTERED_1ST_DIR / f"filtered_assist_1st_stage_{fid}.wav"


def resolve_filtered_2nd(fid: str) -> Path:
    return DATA_ROOT / FILTERED_2ND_DIRNAME / f"{fid}.wav"


# ----- reporting helpers ----------------------------------------------------
def result_row(res: dict, tag: str) -> dict:
    return {
        "experiment":      tag,
        "mean_acc":        res["track_acc_mean"],
        "std_acc":         res["track_acc_std"],
        "window_mean_acc": res["window_acc_mean"],
        "window_std_acc":  res["window_acc_std"],
        "wrong_total":     res["wrong_total"],
        "total":           res["total"],
        **{f"acc_{g}": res["per_genre_acc"][g] for g in GENRES},
    }


def main() -> int:
    np.random.seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    print(f"Loaded {len(labels)} tracks")
    print(f"LPF: Butterworth order {LPF_ORDER}, cutoff {LPF_CUTOFF_HZ:.0f} Hz, "
          f"fs={SR} Hz -> Wn={LPF_CUTOFF_HZ/(SR/2):.3f}")

    print("\nFeature extraction (bandlimited to 3 kHz):")
    Xc, yc, tc = build_bandlimited_dataset(
        labels, resolve_clean,         f"clean_{CUTOFF_TAG}",              "clean")
    print(f"  clean_{CUTOFF_TAG}:               {Xc.shape}")

    X1, y1, t1 = build_bandlimited_dataset(
        labels, resolve_filtered_1st,  f"filtered_1st_stage_{CUTOFF_TAG}", "filtered_1st_stage")
    print(f"  filtered_1st_stage_{CUTOFF_TAG}:  {X1.shape}")

    X2, y2, t2 = build_bandlimited_dataset(
        labels, resolve_filtered_2nd,  f"filtered_2nd_stage_{CUTOFF_TAG}", "filtered_2nd_stage")
    print(f"  filtered_2nd_stage_{CUTOFF_TAG}:  {X2.shape}")

    # Shared window grid across all three sources -- otherwise per-track
    # aggregation and fold splits are not comparable.
    for tag, y_other, t_other in [("filtered_1st", y1, t1), ("filtered_2nd", y2, t2)]:
        assert np.array_equal(yc, y_other), f"y mismatch clean vs {tag}"
        assert np.array_equal(tc, t_other), f"track_id mismatch clean vs {tag}"
    print("  shared window grid OK (identical y and track_id arrays)")

    # ---- folds (same as baseline) ------------------------------------------
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

    # ---- experiments -------------------------------------------------------
    print("=== E' : train CLEAN_3kHz -> test FILTERED_1ST_STAGE_3kHz ===")
    res_e = run_cross(
        X_train_src=Xc, X_test_src=X1,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label=f"clean_{CUTOFF_TAG}->filtered_1st_stage_{CUTOFF_TAG}",
    )
    print("\n=== F' : train CLEAN_3kHz -> test FILTERED_2ND_STAGE_3kHz ===")
    res_f = run_cross(
        X_train_src=Xc, X_test_src=X2,
        y=yc, track_id=tc, track_genre=track_genre,
        splits=splits, label=f"clean_{CUTOFF_TAG}->filtered_2nd_stage_{CUTOFF_TAG}",
    )

    rows = [
        result_row(res_e, f"clean_{CUTOFF_TAG}_train_filtered_1st_stage_test"),
        result_row(res_f, f"clean_{CUTOFF_TAG}_train_filtered_2nd_stage_test"),
    ]
    acc_df  = pd.DataFrame(rows)
    acc_csv = RESULTS_DIR / f"accuracy_table_{CUTOFF_TAG}.csv"
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nWrote {acc_csv}")
    print(acc_df.to_string(index=False))

    plot_confusion(
        np.array(res_e["confusion"]),
        f"3 kHz-bandlimited: clean -> filtered_1st_stage (track-level)",
        RESULTS_DIR / f"confusion_{CUTOFF_TAG}_clean_to_filtered_1st.png",
    )
    plot_confusion(
        np.array(res_f["confusion"]),
        f"3 kHz-bandlimited: clean -> filtered_2nd_stage (track-level)",
        RESULTS_DIR / f"confusion_{CUTOFF_TAG}_clean_to_filtered_2nd.png",
    )

    # ---- summary -----------------------------------------------------------
    e,  se = res_e["track_acc_mean"], res_e["track_acc_std"]
    f2, sf_ = res_f["track_acc_mean"], res_f["track_acc_std"]

    # Full-bandwidth anchors (from frozen baseline / earlier runs)
    A_fb  = 0.988; C_fb = 0.344
    E_fb  = 0.340; F_fb = 0.436
    rec_e = (e  - C_fb) / (A_fb - C_fb)
    rec_f = (f2 - C_fb) / (A_fb - C_fb)

    summary = f"""# Cross-condition under 3 kHz feature-bandlimit

Same pipeline as the frozen baseline, but every waveform (train and
test) is low-pass filtered at {LPF_CUTOFF_HZ:.0f} Hz (Butterworth order
{LPF_ORDER}, zero-phase) **before** windowing and feature extraction.
This guarantees that no feature depends on spectral content above
{LPF_CUTOFF_HZ/1000:.0f} kHz on either side of the train/test boundary.

## Results (per-track accuracy, mean +/- std across {N_SPLITS} folds)

| experiment                                             | accuracy            | wrong / {res_e['total']} | jazz | metal | pop |
|--------------------------------------------------------|---------------------|-----------------------|------|-------|-----|
| E' : clean_3kHz -> filtered_1st_stage_3kHz             | **{e :.3f} +/- {se :.3f}** | {res_e['wrong_total']}                   | {res_e['per_genre_acc']['jazz']:.2f} | {res_e['per_genre_acc']['metal']:.2f} | {res_e['per_genre_acc']['pop']:.2f} |
| F' : clean_3kHz -> filtered_2nd_stage_3kHz             | **{f2:.3f} +/- {sf_:.3f}** | {res_f['wrong_total']}                   | {res_f['per_genre_acc']['jazz']:.2f} | {res_f['per_genre_acc']['metal']:.2f} | {res_f['per_genre_acc']['pop']:.2f} |

Window-level accuracy:
- E' : {res_e['window_acc_mean']:.3f} +/- {res_e['window_acc_std']:.3f}
- F' : {res_f['window_acc_mean']:.3f} +/- {res_f['window_acc_std']:.3f}

## Comparison to full-bandwidth baseline

| pair                    | full-bw | 3 kHz  | delta  |
|-------------------------|--------:|-------:|-------:|
| E  vs  E'               |  {E_fb:.3f}  | {e :.3f} | {e -E_fb:+.3f} |
| F  vs  F'               |  {F_fb:.3f}  | {f2:.3f} | {f2-F_fb:+.3f} |

Recovery fraction of the full-bandwidth A-C gap closed under bandlimit:
- E' : (E' - C_fb) / (A_fb - C_fb) = {rec_e:.2%}
- F' : (F' - C_fb) / (A_fb - C_fb) = {rec_f:.2%}

(Anchors A_fb = {A_fb}, C_fb = {C_fb} are the clean and domain-shift
points from the full-bandwidth frozen baseline, not re-run here.)

## Leakage status

Folds are byte-identical to the baseline StratifiedGroupKFold
(random_state={SEED}, n_splits={N_SPLITS}), so E' and F' are directly
comparable to the full-bandwidth experiments. Full audit:
`verify_no_leakage_fcutoff3khz.py`.

## Confusion matrices

- `confusion_{CUTOFF_TAG}_clean_to_filtered_1st.png`
- `confusion_{CUTOFF_TAG}_clean_to_filtered_2nd.png`
"""
    (RESULTS_DIR / f"summary_{CUTOFF_TAG}.md").write_text(summary, encoding="utf-8")
    (RESULTS_DIR / f"raw_results_{CUTOFF_TAG}.json").write_text(
        json.dumps(
            {
                f"clean_{CUTOFF_TAG}_to_filtered_1st_stage_{CUTOFF_TAG}": res_e,
                f"clean_{CUTOFF_TAG}_to_filtered_2nd_stage_{CUTOFF_TAG}": res_f,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_DIR / f'summary_{CUTOFF_TAG}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
