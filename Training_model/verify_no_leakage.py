"""
Verify no data leakage in the train/test setup of train_compare.py.

Checks (replays the *same* split construction as train_compare.py):
  1. Every cached window's track_id matches the cached genre label.
  2. Clean and noisy caches share identical y and track_id arrays
     (so they really are matched windows of the same source tracks).
  3. For each fold:
       a. train and test track sets are disjoint  (no track leakage),
       b. their union covers all 50 tracks,
       c. no train *window index* appears in the test set, and
          vice versa,
       d. for every test track, *all* of its windows are in test
          (the split is by track, not by window).
  4. Across all folds, every track appears in exactly one test fold.
  5. The fold structure is identical for the clean and noisy
     experiments (same fold IDs -> same train/test track sets).
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

    yc = np.load(CACHE_DIR / "y_clean.npy"); tc = np.load(CACHE_DIR / "t_clean.npy")
    yn = np.load(CACHE_DIR / "y_noisy.npy"); tn = np.load(CACHE_DIR / "t_noisy.npy")

    print("[1] cached labels self-consistent with labels.csv")
    for src, y, t in [("clean", yc, tc), ("noisy", yn, tn)]:
        for tid in np.unique(t):
            ys = y[t == tid]
            if not np.all(ys == ys[0]):
                fail(f"{src}: track {tid} has mixed window labels {np.unique(ys)}")
            if int(ys[0]) != int(y_lab[tid]):
                fail(f"{src}: track {tid} cached label {ys[0]} != csv label {y_lab[tid]}")
    ok(f"all {len(np.unique(tc))} tracks per source: every window's label "
       f"matches labels.csv")

    print("[2] clean and noisy caches share window grid")
    if not np.array_equal(yc, yn): fail("y_clean != y_noisy")
    if not np.array_equal(tc, tn): fail("t_clean != t_noisy")
    ok(f"y and track_id arrays identical between clean and noisy "
       f"(shape={yc.shape})")

    # rebuild the EXACT same splits as train_compare.py
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

    print("[3] per-fold track and window disjointness")
    test_appearances: dict[int, int] = {int(t): 0 for t in unique_tracks}
    for k, (tr_idx, te_idx) in enumerate(splits):
        tr_tracks = set(int(t) for t in unique_tracks[tr_idx])
        te_tracks = set(int(t) for t in unique_tracks[te_idx])

        # 3a: disjoint track sets
        overlap = tr_tracks & te_tracks
        if overlap:
            fail(f"fold {k+1}: tracks in BOTH train and test: {sorted(overlap)}")
        # 3b: cover all tracks
        if (tr_tracks | te_tracks) != all_tracks:
            missing = all_tracks - (tr_tracks | te_tracks)
            fail(f"fold {k+1}: tracks missing from train+test: {sorted(missing)}")

        tr_win = np.concatenate([track_to_windows[t] for t in tr_tracks])
        te_win = np.concatenate([track_to_windows[t] for t in te_tracks])

        # 3c: disjoint window index sets
        wsetA = set(int(i) for i in tr_win)
        wsetB = set(int(i) for i in te_win)
        winov = wsetA & wsetB
        if winov:
            fail(f"fold {k+1}: window indices in both train and test: "
                 f"{len(winov)} indices")

        # 3d: every test track contributes ALL its windows to test
        for t in te_tracks:
            if not np.array_equal(np.sort(track_to_windows[t]),
                                  np.sort(te_win[np.isin(te_win, track_to_windows[t])])):
                fail(f"fold {k+1}: test track {t} not fully placed in test set")

        for t in te_tracks:
            test_appearances[t] += 1

        ok(f"fold {k+1}: train={len(tr_tracks)} tracks / {len(tr_win)} windows, "
           f"test={len(te_tracks)} tracks / {len(te_win)} windows -- disjoint")

    print("[4] every track appears in exactly one test fold")
    bad = [t for t, c in test_appearances.items() if c != 1]
    if bad: fail(f"tracks with appearances != 1: {bad}")
    ok(f"all {len(test_appearances)} tracks appear in exactly one test fold "
       f"(min={min(test_appearances.values())}, max={max(test_appearances.values())})")

    print("[5] same fold structure used for clean and noisy")
    # train_compare.py constructs `splits` once and feeds it to BOTH experiments.
    # Re-derive splits from the noisy cache and verify they're identical.
    unique_tracks_n = np.unique(tn)
    track_genre_n   = np.array([y_lab[int(t)] for t in unique_tracks_n], dtype=np.int64)
    sgkf_n = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits_n = list(sgkf_n.split(
        X=np.zeros((len(unique_tracks_n), 1)),
        y=track_genre_n,
        groups=unique_tracks_n,
    ))
    for k, ((a_tr, a_te), (b_tr, b_te)) in enumerate(zip(splits, splits_n)):
        if not (np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)):
            fail(f"fold {k+1}: clean/noisy splits differ -- gap is NOT directly comparable")
    ok("clean and noisy experiments use byte-identical fold assignments")

    print("\nALL CHECKS PASSED -- no track leakage and matched folds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
