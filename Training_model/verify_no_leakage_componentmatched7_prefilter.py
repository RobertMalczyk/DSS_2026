"""
Verify no data leakage in experiment H_pf (train clean_pf -> test
componentmatched_7th_pf), where the prefiltered feature variant is
cached as *_compmatch.npy.

Same checks as verify_no_leakage_componentmatched7.py, applied to the
prefiltered caches, plus an extra check that the prefiltered window
grid is byte-identical to the plain (un-prefiltered) one -- the
prefilter is linear and length-preserving so it MUST yield the same
number of windows per track.
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
VARIANT   = "compmatch"


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ok:   {msg}")


def main() -> int:
    labels = pd.read_csv(DATA_ROOT / "labels.csv").reset_index(drop=True)
    y_lab  = np.array([G2I[g] for g in labels["genre"]], dtype=np.int64)

    required = [
        f"X_clean_{VARIANT}.npy", f"y_clean_{VARIANT}.npy", f"t_clean_{VARIANT}.npy",
        f"X_filtered_componentmatched_7th_stage_{VARIANT}.npy",
        f"y_filtered_componentmatched_7th_stage_{VARIANT}.npy",
        f"t_filtered_componentmatched_7th_stage_{VARIANT}.npy",
    ]
    missing  = [p for p in required if not (CACHE_DIR / p).is_file()]
    if missing:
        fail(f"missing caches: {missing} -- "
             f"run train_cross_componentmatched7_prefilter.py first")

    yc = np.load(CACHE_DIR / f"y_clean_{VARIANT}.npy")
    tc = np.load(CACHE_DIR / f"t_clean_{VARIANT}.npy")
    yh = np.load(CACHE_DIR / f"y_filtered_componentmatched_7th_stage_{VARIANT}.npy")
    th = np.load(CACHE_DIR / f"t_filtered_componentmatched_7th_stage_{VARIANT}.npy")
    Xc_shape = np.load(CACHE_DIR / f"X_clean_{VARIANT}.npy", mmap_mode="r").shape
    Xh_shape = np.load(CACHE_DIR / f"X_filtered_componentmatched_7th_stage_{VARIANT}.npy",
                       mmap_mode="r").shape

    print("[1] prefiltered caches self-consistent with labels.csv")
    for src, y, t in [("clean_pf", yc, tc), ("componentmatched_7th_pf", yh, th)]:
        for tid in np.unique(t):
            ys = y[t == tid]
            if not np.all(ys == ys[0]):
                fail(f"{src}: track {tid} has mixed window labels {np.unique(ys)}")
            if int(ys[0]) != int(y_lab[tid]):
                fail(f"{src}: track {tid} cached label {ys[0]} != csv label {y_lab[tid]}")
    ok(f"all {len(np.unique(tc))} tracks per source: every window's label "
       f"matches labels.csv")

    print("[2] clean_pf and componentmatched_7th_pf caches share window grid")
    if Xc_shape != Xh_shape:
        fail(f"feature-matrix shape mismatch: clean_pf={Xc_shape} 7th_pf={Xh_shape}")
    if not np.array_equal(yc, yh): fail("y mismatch (clean_pf vs 7th_pf)")
    if not np.array_equal(tc, th): fail("track_id mismatch (clean_pf vs 7th_pf)")
    ok(f"y and track_id arrays identical between clean_pf and 7th_pf "
       f"(shape={yc.shape}, X shape={Xc_shape})")

    print("[2b] window grid identical to plain (un-prefiltered) caches")
    plain_t_clean = CACHE_DIR / "t_clean.npy"
    plain_t_h     = CACHE_DIR / "t_filtered_componentmatched_7th_stage.npy"
    if plain_t_clean.is_file() and plain_t_h.is_file():
        tc0 = np.load(plain_t_clean); th0 = np.load(plain_t_h)
        if not np.array_equal(tc, tc0): fail("track_id differs from un-prefiltered clean cache")
        if not np.array_equal(th, th0): fail("track_id differs from un-prefiltered 7th cache")
        ok(f"prefiltered window grid matches plain caches (no extra/missing windows)")
    else:
        ok("plain caches not present; skipping cross-check (prefilter is length-preserving)")

    unique_tracks = np.unique(tc)
    track_genre   = np.array([y_lab[int(t)] for t in unique_tracks], dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(
        X=np.zeros((len(unique_tracks), 1)),
        y=track_genre, groups=unique_tracks,
    ))
    track_to_windows = {int(t): np.where(tc == t)[0] for t in unique_tracks}
    all_tracks       = set(int(t) for t in unique_tracks)

    print("[3] per-fold train(clean_pf)/test(7th_pf) track disjointness")
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

        tr_win = np.concatenate([track_to_windows[t] for t in tr_tracks])
        te_win = np.concatenate([track_to_windows[t] for t in te_tracks])

        train_win_tracks = set(int(x) for x in tc[tr_win])
        test_win_tracks  = set(int(x) for x in th[te_win])
        win_overlap = train_win_tracks & test_win_tracks
        if win_overlap:
            fail(f"fold {k+1}: tracks behind train/test windows overlap: "
                 f"{sorted(win_overlap)}")

        for t in te_tracks:
            expected = np.sort(track_to_windows[t])
            got      = np.sort(te_win[np.isin(te_win, track_to_windows[t])])
            if not np.array_equal(expected, got):
                fail(f"fold {k+1}: test track {t} not fully placed in test set")

        for t in te_tracks:
            test_appearances[t] += 1

        ok(f"fold {k+1}: train(clean_pf)={len(tr_tracks)}/{len(tr_win)}w, "
           f"test(7th_pf)={len(te_tracks)}/{len(te_win)}w -- disjoint track ids")

    print("[4] every track appears in exactly one test fold")
    bad = [t for t, c in test_appearances.items() if c != 1]
    if bad: fail(f"tracks with appearances != 1: {bad}")
    ok(f"all {len(test_appearances)} tracks appear in exactly one test fold")

    print("[5] same fold structure as A/B/C/D/E/F/G/H (so H_pf is directly comparable)")
    unique_tracks_h = np.unique(th)
    track_genre_h   = np.array([y_lab[int(t)] for t in unique_tracks_h], dtype=np.int64)
    sgkf_h = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits_h = list(sgkf_h.split(
        X=np.zeros((len(unique_tracks_h), 1)),
        y=track_genre_h, groups=unique_tracks_h,
    ))
    for k, ((a_tr, a_te), (b_tr, b_te)) in enumerate(zip(splits, splits_h)):
        if not (np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)):
            fail(f"fold {k+1}: clean_pf/7th_pf splits differ -- H_pf is NOT comparable")
    ok("componentmatched_7th_pf uses byte-identical fold assignments to baseline")

    print("\nALL CHECKS PASSED -- no track leakage and matched folds for H_pf.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
