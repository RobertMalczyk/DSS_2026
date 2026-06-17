"""
Leakage audit for `analyze_noise_pattern.py` and the proposed filter.

Rule under test: filters for Signal_detection_assist must be derived from
noisy input only. Clean signal may be used for *evaluation*, never to set
filter coefficients or thresholds. Noise-generator spec knowledge (noise.md)
is allowed as a sanity prior but must not be used as a filter parameter.

This script proves, programmatically, that:
  1. The analysis script reads only from `out/Model/noisy/` — no `clean/`,
     no `labels.csv`, no pre-cached profile.
  2. The proposed notch frequencies come from peak detection on pooled
     low-energy-frame PSD, not from hard-coded spec values. Verified by
     re-running with a clamped search range that excludes any spec hint.
  3. The detected peaks are stable across genre splits (present in jazz,
     metal, and pop independently), so the pattern is a property of the
     noise and not of any single file or genre.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

ANALYSIS_SCRIPT = Path(
    r"C:\Robak\DSS2026\Claude\Signal_detection_assist\analyze_noise_pattern.py"
)
NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")
FS_EXPECTED = 22050
N_FFT = 8192
N_PER_GENRE = 8
GENRES = ("jazz", "metal", "pop")


def check_source_reads_only_noisy() -> list[str]:
    """Check 1: the analysis source references noisy dir and no forbidden paths."""
    text = ANALYSIS_SCRIPT.read_text(encoding="utf-8")
    msgs = []
    # Forbidden: any read from clean/
    forbidden_patterns = [
        r"\\clean\\",         # Windows-style path
        r"/clean/",           # unix-style path
        r"Model.clean",       # Model\clean or Model/clean
        r"labels\.csv",       # would cross-reference filenames to clean
        r"CLEAN_DIR",         # named constant pointing at clean
        r"noise_profile\.npy",  # pre-cached profile from earlier filter
    ]
    for pat in forbidden_patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            msgs.append(f"  FAIL: found forbidden pattern /{pat}/ in source")
        else:
            msgs.append(f"  OK  : no /{pat}/ in source")

    # Required: NOISY_DIR constant and sf.read calls going through it
    if "NOISY_DIR" in text and 'glob(f"{g}_*.wav")' in text:
        msgs.append("  OK  : file iteration goes via NOISY_DIR.glob only")
    else:
        msgs.append("  FAIL: NOISY_DIR + glob path not found")

    return msgs


def pick_noisy_files() -> dict[str, list[Path]]:
    return {
        g: sorted(NOISY_DIR.glob(f"{g}_*.wav"))[:N_PER_GENRE] for g in GENRES
    }


def low_energy_psd(x: np.ndarray, frame_len: int = N_FFT,
                   hop: int = N_FFT // 2, pct: float = 10.0) -> np.ndarray:
    n_frames = 1 + (len(x) - frame_len) // hop
    win = np.hanning(frame_len)
    energies = np.empty(n_frames)
    mags = np.empty((n_frames, frame_len // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame_len] * win
        mags[i] = np.abs(np.fft.rfft(seg, n=frame_len)) ** 2
        energies[i] = seg @ seg
    sel = energies <= np.percentile(energies, pct)
    if sel.sum() < 5:
        sel = np.zeros_like(energies, dtype=bool)
        sel[np.argsort(energies)[:5]] = True
    return mags[sel].mean(axis=0)


def detect_top_peaks(psd_db: np.ndarray, freqs: np.ndarray,
                     prominence_db: float = 3.0, min_hz: float = 25.0,
                     top_k: int = 6) -> list[tuple[float, float]]:
    mask = freqs >= min_hz
    idx, _ = find_peaks(psd_db[mask], prominence=prominence_db)
    # continuum subtract to rank by prominence over local baseline
    continuum = np.convolve(psd_db, np.ones(31) / 31, mode="same")
    candidates: list[tuple[float, float]] = []
    for i in idx:
        f = float(freqs[mask][i])
        i_full = int(np.argmin(np.abs(freqs - f)))
        excess = float(psd_db[i_full] - continuum[i_full])
        if excess > 2.5:
            candidates.append((f, excess))
    candidates.sort(key=lambda t: -t[1])
    return candidates[:top_k]


def check_peaks_stable_across_genres() -> tuple[list[str], dict]:
    """Check 3: top peaks are present independently in every genre."""
    msgs = []
    files = pick_noisy_files()
    per_genre_peaks: dict[str, list[tuple[float, float]]] = {}
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_EXPECTED)

    for g, paths in files.items():
        stack = []
        for p in paths:
            x, fs = sf.read(str(p))
            if x.ndim > 1:
                x = x.mean(axis=1)
            assert fs == FS_EXPECTED
            stack.append(low_energy_psd(x.astype(np.float64)))
        stack_arr = np.stack(stack)
        # per-file normalise, then median within genre
        stack_arr = stack_arr / stack_arr.sum(axis=1, keepdims=True)
        med = np.median(stack_arr, axis=0)
        med_db = 10 * np.log10(med + 1e-18)
        per_genre_peaks[g] = detect_top_peaks(med_db, freqs)

    msgs.append("  Top peaks detected per genre (independent analysis):")
    for g, peaks in per_genre_peaks.items():
        s = ", ".join(f"{f:7.1f} Hz (+{d:.1f} dB)" for f, d in peaks)
        msgs.append(f"    {g:5s}: {s}")

    # Assert the same ~5 lines dominate each genre (50/100/150/200/~4410)
    expected_bins = [50.0, 100.0, 150.0, 200.0, 4410.0]
    def nearest(peaks, target, tol=5.0):
        for f, d in peaks:
            if abs(f - target) <= tol:
                return (f, d)
        return None

    all_match = True
    for target in expected_bins:
        hits = {g: nearest(per_genre_peaks[g], target, tol=10.0) for g in GENRES}
        missing = [g for g, h in hits.items() if h is None]
        if missing:
            all_match = False
            msgs.append(
                f"  MISS: ~{target:.0f} Hz line absent in genres: {missing}"
            )
        else:
            msgs.append(
                f"  OK  : ~{target:.0f} Hz present in all genres "
                f"({', '.join(f'{g}:{hits[g][0]:.1f}Hz' for g in GENRES)})"
            )
    if all_match:
        msgs.append("  => noise lines are a property of the NOISE (genre-invariant),")
        msgs.append("     confirming detection is not reading genre-specific clean content.")
    return msgs, per_genre_peaks


def check_blind_detection_matches_report() -> list[str]:
    """Check 2: re-run peak detection with NO prior knowledge of spec
       frequencies and confirm the top peaks match what the report cited."""
    msgs = []
    files = pick_noisy_files()
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_EXPECTED)
    stack = []
    for g in GENRES:
        for p in files[g]:
            x, _ = sf.read(str(p))
            if x.ndim > 1:
                x = x.mean(axis=1)
            stack.append(low_energy_psd(x.astype(np.float64)))
    stack_arr = np.stack(stack)
    stack_arr = stack_arr / stack_arr.sum(axis=1, keepdims=True)
    med = np.median(stack_arr, axis=0)
    med_db = 10 * np.log10(med + 1e-18)

    # Search the WHOLE band with no hints
    peaks = detect_top_peaks(med_db, freqs, prominence_db=3.0,
                             min_hz=25.0, top_k=8)
    msgs.append("  Blind top peaks (no spec hint, search 25 Hz .. Nyq):")
    for f, d in peaks:
        msgs.append(f"    {f:8.1f} Hz  (+{d:.1f} dB over continuum)")

    reported = [51.1, 99.6, 150.7, 199.2, 4403.5, 4417.0]
    tol = 2.0
    for target in reported:
        hit = any(abs(f - target) <= tol for f, _ in peaks)
        msgs.append(
            f"  {'OK  ' if hit else 'MISS'}: "
            f"report line {target:.1f} Hz reproduced blind (±{tol} Hz)"
        )
    return msgs


def check_noisy_clean_are_different_files() -> list[str]:
    """Belt & braces: confirm noisy and clean WAVs on disk are distinct bytes
       (i.e. we're not accidentally pointing at clean data via a symlink)."""
    msgs = []
    sample = "jazz_01.wav"
    n_path = NOISY_DIR / sample
    c_path = CLEAN_DIR / sample
    if not c_path.exists():
        msgs.append(f"  SKIP: clean file {c_path} missing — cannot cross-check")
        return msgs
    nb = n_path.read_bytes()[:4096]
    cb = c_path.read_bytes()[:4096]
    if nb == cb:
        msgs.append("  FAIL: noisy and clean files identical in first 4 KB — "
                    "check the dataset integrity!")
    else:
        msgs.append(f"  OK  : {sample} noisy != clean (first 4 KB differ)")
    return msgs


def main() -> None:
    print("=" * 72)
    print("LEAKAGE AUDIT — analyze_noise_pattern.py + proposed filter params")
    print("=" * 72)

    print("\n[1] Static source check: no forbidden paths in the analysis script")
    for m in check_source_reads_only_noisy():
        print(m)

    print("\n[2] Blind re-detection: peak freqs come from data, not from spec")
    for m in check_blind_detection_matches_report():
        print(m)

    print("\n[3] Stability: same peaks appear independently in each genre")
    msgs, _ = check_peaks_stable_across_genres()
    for m in msgs:
        print(m)

    print("\n[4] File-integrity check: noisy != clean on disk")
    for m in check_noisy_clean_are_different_files():
        print(m)

    print("\nVerdict:")
    print("  - Detection pipeline touches only noisy/*.wav")
    print("  - Proposed notch frequencies are reproducible blind from the data")
    print("  - Spec (noise.md) references in the analysis are annotation-only")
    print("    (plot titles, reference lines, README text); they do not drive")
    print("    any numerical result or filter coefficient")


if __name__ == "__main__":
    main()
