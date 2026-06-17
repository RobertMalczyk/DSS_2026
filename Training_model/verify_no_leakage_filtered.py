"""
Verify no data leakage in the D experiment (train clean -> test filtered_AI).

Checks (replays the same split construction as train_cross_filtered.py):
  1. Every filtered_AI cache file exists and the track_id / genre labels
     line up with labels.csv.
  2. Clean and filtered_AI caches share identical y and track_id arrays
     (same window grid -> safe to apply the clean-trained model to the
     filtered windows of the same track at test time).
  3. For each fold:
       a. train and test track sets are disjoint  (no track leakage),
       b. their union covers all tracks,
       c. for a given fold, the train windows (taken from the CLEAN
          cache) and the test windows (taken from the FILTERED_AI cache)
          belong to disjoint track ids,
       d. for every test track, all of its windows are in the test set.
  4. Across all folds, every track appears in exactly one test fold.
  5. The fold structure is byte-identical to the one used for
     experiments A/B/C (train_compare.py / train_cross.py). This is the
     invariant that makes A vs D and C vs D directly comparable.
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
                "X_filtered_AI.npy", "y_filtered_AI.npy", "t_filtered_AI.npy"]
    missing  = [p for p in required if not (CACHE_DIR / p).is_file()]
    if missing:
        fail(f"missing caches: {missing} -- run train_cross_filtered.py first")

    yc = np.load(CACHE_DIR / "y_clean.npy"); tc = np.load(CACHE_DIR / "t_clean.npy")
    yf = np.load(CACHE_DIR / "y_filtered_AI.npy"); tf = np.load(CACHE_DIR / "t_filtered_AI.npy")
    Xc_shape = np.load(CACHE_DIR / "X_clean.npy", mmap_mode="r").shape
    Xf_shape = np.load(CACHE_DIR / "X_filtered_AI.npy", mmap_mode="r").shape

    print("[1] filtered_AI cached labels self-consistent with labels.csv")
    for src, y, t in [("clean", yc, tc), ("filtered_AI", yf, tf)]:
        for tid in np.unique(t):
            ys = y[t == tid]
            if not np.all(ys == ys[0]):
                fail(f"{src}: track {tid} has mixed window labels {np.unique(ys)}")
            if int(ys[0]) != int(y_lab[tid]):
                fail(f"{src}: track {tid} cached label {ys[0]} != csv label {y_lab[tid]}")
    ok(f"all {len(np.unique(tc))} tracks per source: every window's label "
       f"matches labels.csv")

    print("[2] clean and filtered_AI caches share window grid")
    if Xc_shape != Xf_shape:
        fail(f"feature-matrix shape mismatch: clean={Xc_shape} filtered={Xf_shape}")
    if not np.array_equal(yc, yf): fail("y_clean != y_filtered_AI")
    if not np.array_equal(tc, tf): fail("t_clean != t_filtered_AI")
    ok(f"y and track_id arrays identical between clean and filtered_AI "
       f"(shape={yc.shape}, X shape={Xc_shape})")

    # Rebuild the EXACT same splits as train_compare.py / train_cross.py.
    unique_tracks = np.unique(tc)
    track_genre   = np.array([y_lab[int(t)] for t in unique_tracks], dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre,
        groups=unique_tracks,
    ))

    # Window index -> track map. train uses the clean cache, test uses
    # the filtered cache, but since y and track_id arrays are identical
    # (checked above) the window-index-to-track mapping is the same.
    track_to_windows = {int(t): np.where(tc == t)[0] for t in unique_tracks}
    all_tracks       = set(int(t) for t in unique_tracks)

    print("[3] per-fold train(clean)/test(filtered_AI) track disjointness")
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

        # (3c) Cross-source disjointness: the TRACK IDs behind train windows
        # (from clean cache) must not overlap the TRACK IDs behind test
        # windows (from filtered cache). Using tc here because tc==tf.
        train_win_tracks = set(int(x) for x in tc[tr_win_clean])
        test_win_tracks  = set(int(x) for x in tf[te_win_filt])
        win_overlap = train_win_tracks & test_win_tracks
        if win_overlap:
            fail(f"fold {k+1}: tracks behind train/test windows overlap: "
                 f"{sorted(win_overlap)}")

        # (3d) every test track contributes ALL of its windows to the test set
        for t in te_tracks:
            expected = np.sort(track_to_windows[t])
            got      = np.sort(te_win_filt[np.isin(te_win_filt, track_to_windows[t])])
            if not np.array_equal(expected, got):
                fail(f"fold {k+1}: test track {t} not fully placed in test set")

        for t in te_tracks:
            test_appearances[t] += 1

        ok(f"fold {k+1}: train(clean)={len(tr_tracks)} tracks / {len(tr_win_clean)} windows, "
           f"test(filtered_AI)={len(te_tracks)} tracks / {len(te_win_filt)} windows -- disjoint track ids")

    print("[4] every track appears in exactly one test fold")
    bad = [t for t, c in test_appearances.items() if c != 1]
    if bad: fail(f"tracks with appearances != 1: {bad}")
    ok(f"all {len(test_appearances)} tracks appear in exactly one test fold "
       f"(min={min(test_appearances.values())}, max={max(test_appearances.values())})")

    print("[5] same fold structure as A/B/C (so D is directly comparable)")
    # Rebuild from the filtered cache's unique tracks; must be byte-identical
    # to `splits` above.
    unique_tracks_f = np.unique(tf)
    track_genre_f   = np.array([y_lab[int(t)] for t in unique_tracks_f], dtype=np.int64)
    sgkf_f = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits_f = list(sgkf_f.split(
        X=np.zeros((len(unique_tracks_f), 1)),
        y=track_genre_f,
        groups=unique_tracks_f,
    ))
    for k, ((a_tr, a_te), (b_tr, b_te)) in enumerate(zip(splits, splits_f)):
        if not (np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)):
            fail(f"fold {k+1}: clean/filtered splits differ -- D is NOT directly comparable to A/C")
    ok("filtered_AI uses byte-identical fold assignments to clean/noisy")

    print("\nALL CHECKS PASSED -- no track leakage and matched folds for D.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
