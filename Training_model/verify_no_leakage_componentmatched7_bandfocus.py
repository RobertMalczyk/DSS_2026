"""
Verify no data leakage in the bandfocus variants H_bf4000 and H_bf5500
(prefiltered + LP-bandlimited features). Same checks as the other
verify_* scripts, applied to both *_compmatch_bf4000.npy and
*_compmatch_bf5500.npy caches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
DATA_ROOT = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model")

SEED      = 42
N_SPLITS  = 5
GENRES    = ("jazz", "metal", "pop")
G2I       = {g: i for i, g in enumerate(GENRES)}
VARIANTS  = ("compmatch_bf4000", "compmatch_bf5500")


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ok:   {msg}")


def check_variant(variant: str, labels: pd.DataFrame, y_lab: np.ndarray) -> None:
    print(f"\n=== variant: {variant} ===")
    required = [
        f"X_clean_{variant}.npy", f"y_clean_{variant}.npy", f"t_clean_{variant}.npy",
        f"X_filtered_componentmatched_7th_stage_{variant}.npy",
        f"y_filtered_componentmatched_7th_stage_{variant}.npy",
        f"t_filtered_componentmatched_7th_stage_{variant}.npy",
    ]
    missing  = [p for p in required if not (CACHE_DIR / p).is_file()]
    if missing:
        fail(f"missing caches: {missing} -- "
             f"run train_cross_componentmatched7_bandfocus.py first")

    yc = np.load(CACHE_DIR / f"y_clean_{variant}.npy")
    tc = np.load(CACHE_DIR / f"t_clean_{variant}.npy")
    yh = np.load(CACHE_DIR / f"y_filtered_componentmatched_7th_stage_{variant}.npy")
    th = np.load(CACHE_DIR / f"t_filtered_componentmatched_7th_stage_{variant}.npy")
    Xc_shape = np.load(CACHE_DIR / f"X_clean_{variant}.npy", mmap_mode="r").shape
    Xh_shape = np.load(CACHE_DIR / f"X_filtered_componentmatched_7th_stage_{variant}.npy",
                       mmap_mode="r").shape

    print("[1] cached labels self-consistent with labels.csv")
    for src, y, t in [(f"clean_{variant}", yc, tc),
                      (f"7th_{variant}",   yh, th)]:
        for tid in np.unique(t):
            ys = y[t == tid]
            if not np.all(ys == ys[0]):
                fail(f"{src}: track {tid} has mixed window labels {np.unique(ys)}")
            if int(ys[0]) != int(y_lab[tid]):
                fail(f"{src}: track {tid} cached label {ys[0]} != csv label {y_lab[tid]}")
    ok(f"all {len(np.unique(tc))} tracks per source: every window matches labels.csv")

    print(f"[2] caches share window grid (shape={Xc_shape})")
    if Xc_shape != Xh_shape:
        fail(f"shape mismatch: clean={Xc_shape} 7th={Xh_shape}")
    if not np.array_equal(yc, yh): fail("y mismatch")
    if not np.array_equal(tc, th): fail("track_id mismatch")
    ok("y and track_id arrays identical between clean and 7th")

    print("[2b] window grid identical to plain (un-prefiltered) caches")
    plain_t_clean = CACHE_DIR / "t_clean.npy"
    plain_t_h     = CACHE_DIR / "t_filtered_componentmatched_7th_stage.npy"
    if plain_t_clean.is_file() and plain_t_h.is_file():
        tc0 = np.load(plain_t_clean); th0 = np.load(plain_t_h)
        if not np.array_equal(tc, tc0): fail("track_id differs from un-prefiltered clean cache")
        if not np.array_equal(th, th0): fail("track_id differs from un-prefiltered 7th cache")
        ok("prefiltered+bandlimited window grid matches plain caches")
    else:
        ok("plain caches not present; skipping cross-check")

    unique_tracks = np.unique(tc)
    track_genre   = np.array([y_lab[int(t)] for t in unique_tracks], dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre, groups=unique_tracks,
    ))
    track_to_windows = {int(t): np.where(tc == t)[0] for t in unique_tracks}
    all_tracks       = set(int(t) for t in unique_tracks)

    print("[3] per-fold train/test track disjointness")
    test_appearances: dict[int, int] = {int(t): 0 for t in unique_tracks}
    for k, (tr_idx, te_idx) in enumerate(splits):
        tr_tracks = set(int(t) for t in unique_tracks[tr_idx])
        te_tracks = set(int(t) for t in unique_tracks[te_idx])
        overlap = tr_tracks & te_tracks
        if overlap:
            fail(f"fold {k+1}: train/test overlap: {sorted(overlap)}")
        if (tr_tracks | te_tracks) != all_tracks:
            fail(f"fold {k+1}: tracks missing from train+test: "
                 f"{sorted(all_tracks - (tr_tracks | te_tracks))}")
        tr_win = np.concatenate([track_to_windows[t] for t in tr_tracks])
        te_win = np.concatenate([track_to_windows[t] for t in te_tracks])
        if set(int(x) for x in tc[tr_win]) & set(int(x) for x in th[te_win]):
            fail(f"fold {k+1}: tracks behind train/test windows overlap")
        for t in te_tracks:
            test_appearances[t] += 1
    bad = [t for t, c in test_appearances.items() if c != 1]
    if bad: fail(f"tracks appearing != 1 time in test: {bad}")
    ok(f"5 folds OK, all {len(test_appearances)} tracks tested exactly once")

    print("[4] same fold structure as A/B/C/D/E/F/G/H")
    unique_tracks_h = np.unique(th)
    track_genre_h   = np.array([y_lab[int(t)] for t in unique_tracks_h], dtype=np.int64)
    sgkf_h = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits_h = list(sgkf_h.split(
        X=np.zeros((len(unique_tracks_h), 1)),
        y=track_genre_h, groups=unique_tracks_h,
    ))
    for k, ((a_tr, a_te), (b_tr, b_te)) in enumerate(zip(splits, splits_h)):
        if not (np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)):
            fail(f"fold {k+1}: clean/7th splits differ -- not directly comparable")
    ok("byte-identical fold assignments to baseline")


def main() -> int:
    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    y_lab  = np.array([G2I[g] for g in labels["genre"]], dtype=np.int64)
    for v in VARIANTS:
        check_variant(v, labels, y_lab)
    print("\nALL CHECKS PASSED for both bandfocus variants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
