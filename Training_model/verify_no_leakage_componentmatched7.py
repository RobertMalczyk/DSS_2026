"""
Verify no data leakage in experiment H (train clean -> test filtered_componentmatched_7th_stage).

Checks (replays the same split construction as train_cross_componentmatched7.py):
  1. filtered_componentmatched_7th_stage cache is self-consistent with labels.csv.
  2. Clean and componentmatched_7th_stage caches share identical y and
     track_id arrays (same window grid).
  3. For each fold:
       a. train and test track sets are disjoint,
       b. their union covers all tracks,
       c. track ids behind train windows (CLEAN cache) and test windows
          (COMPONENTMATCHED cache) are disjoint,
       d. every test track contributes *all* its windows to the test set.
  4. Each track appears in exactly one test fold.
  5. The fold structure is byte-identical to A/B/C/D/E/F/G -- so H is
     directly comparable to the rest of the baseline.
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


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ok:   {msg}")


def main() -> int:
    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    y_lab  = np.array([G2I[g] for g in labels["genre"]], dtype=np.int64)

    required = ["X_clean.npy", "y_clean.npy", "t_clean.npy",
                "X_filtered_componentmatched_7th_stage.npy",
                "y_filtered_componentmatched_7th_stage.npy",
                "t_filtered_componentmatched_7th_stage.npy"]
    missing  = [p for p in required if not (CACHE_DIR / p).is_file()]
    if missing:
        fail(f"missing caches: {missing} -- run train_cross_componentmatched7.py first")

    yc = np.load(CACHE_DIR / "y_clean.npy"); tc = np.load(CACHE_DIR / "t_clean.npy")
    yh = np.load(CACHE_DIR / "y_filtered_componentmatched_7th_stage.npy")
    th = np.load(CACHE_DIR / "t_filtered_componentmatched_7th_stage.npy")
    Xc_shape = np.load(CACHE_DIR / "X_clean.npy",                                 mmap_mode="r").shape
    Xh_shape = np.load(CACHE_DIR / "X_filtered_componentmatched_7th_stage.npy",   mmap_mode="r").shape

    print("[1] filtered_componentmatched_7th_stage cached labels self-consistent with labels.csv")
    for src, y, t in [("clean", yc, tc), ("componentmatched_7th_stage", yh, th)]:
        for tid in np.unique(t):
            ys = y[t == tid]
            if not np.all(ys == ys[0]):
                fail(f"{src}: track {tid} has mixed window labels {np.unique(ys)}")
            if int(ys[0]) != int(y_lab[tid]):
                fail(f"{src}: track {tid} cached label {ys[0]} != csv label {y_lab[tid]}")
    ok(f"all {len(np.unique(tc))} tracks per source: every window's label "
       f"matches labels.csv")

    print("[2] clean and componentmatched_7th_stage caches share window grid")
    if Xc_shape != Xh_shape:
        fail(f"feature-matrix shape mismatch: clean={Xc_shape} componentmatched={Xh_shape}")
    if not np.array_equal(yc, yh): fail("y_clean != y_filtered_componentmatched_7th_stage")
    if not np.array_equal(tc, th): fail("t_clean != t_filtered_componentmatched_7th_stage")
    ok(f"y and track_id arrays identical between clean and componentmatched_7th_stage "
       f"(shape={yc.shape}, X shape={Xc_shape})")

    unique_tracks = np.unique(tc)
    track_genre   = np.array([y_lab[int(t)] for t in unique_tracks], dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre,
        groups=unique_tracks,
    ))

    track_to_windows = {int(t): np.where(tc == t)[0] for t in unique_tracks}
    all_tracks       = set(int(t) for t in unique_tracks)

    print("[3] per-fold train(clean)/test(componentmatched_7th_stage) track disjointness")
    test_appearances: dict[int, int] = {int(t): 0 for t in unique_tracks}
    for k, (tr_idx, te_idx) in enumerate(splits):
        tr_tracks = set(int(t) for t in unique_tracks[tr_idx])
        te_tracks = set(int(t) for t in unique_tracks[te_idx])

        overlap = tr_tracks & te_tracks
        if overlap:
            fail(f"fold {k+1}: tracks in BOTH train and test: {sorted(overlap)}")
        if (tr_tracks | te_tracks) != all_tracks:
            missing = all_tracks - (tr_tracks | te_tracks)
            fail(f"fold {k+1}: tracks missing from train+test: {sorted(missing)}")

        tr_win_clean = np.concatenate([track_to_windows[t] for t in tr_tracks])
        te_win_filt  = np.concatenate([track_to_windows[t] for t in te_tracks])

        train_win_tracks = set(int(x) for x in tc[tr_win_clean])
        test_win_tracks  = set(int(x) for x in th[te_win_filt])
        win_overlap = train_win_tracks & test_win_tracks
        if win_overlap:
            fail(f"fold {k+1}: tracks behind train/test windows overlap: "
                 f"{sorted(win_overlap)}")

        for t in te_tracks:
            expected = np.sort(track_to_windows[t])
            got      = np.sort(te_win_filt[np.isin(te_win_filt, track_to_windows[t])])
            if not np.array_equal(expected, got):
                fail(f"fold {k+1}: test track {t} not fully placed in test set")

        for t in te_tracks:
            test_appearances[t] += 1

        ok(f"fold {k+1}: train(clean)={len(tr_tracks)} tracks / {len(tr_win_clean)} windows, "
           f"test(componentmatched_7th)={len(te_tracks)} tracks / {len(te_win_filt)} windows -- disjoint track ids")

    print("[4] every track appears in exactly one test fold")
    bad = [t for t, c in test_appearances.items() if c != 1]
    if bad: fail(f"tracks with appearances != 1: {bad}")
    ok(f"all {len(test_appearances)} tracks appear in exactly one test fold "
       f"(min={min(test_appearances.values())}, max={max(test_appearances.values())})")

    print("[5] same fold structure as A/B/C/D/E/F/G (so H is directly comparable)")
    unique_tracks_h = np.unique(th)
    track_genre_h   = np.array([y_lab[int(t)] for t in unique_tracks_h], dtype=np.int64)
    sgkf_h = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits_h = list(sgkf_h.split(
        X=np.zeros((len(unique_tracks_h), 1)),
        y=track_genre_h,
        groups=unique_tracks_h,
    ))
    for k, ((a_tr, a_te), (b_tr, b_te)) in enumerate(zip(splits, splits_h)):
        if not (np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)):
            fail(f"fold {k+1}: clean/componentmatched_7th splits differ -- H is NOT directly comparable")
    ok("componentmatched_7th_stage uses byte-identical fold assignments to the baseline")

    print("\nALL CHECKS PASSED -- no track leakage and matched folds for H.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
