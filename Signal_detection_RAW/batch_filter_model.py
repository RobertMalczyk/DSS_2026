"""Batch-apply the music chain from denoise.py to every WAV in
..\\Signal_generation\\out\\Model\\noisy and write the result to
..\\Signal_generation\\out\\Model\\filtered_AI.

This operation is deliberately outside Signal_detection_RAW\\out\\ — invoked
at the user's explicit request (see transcript). The filter itself was
designed blindly from the three noisy samples; nothing about its design is
being revisited here, it is purely being applied.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal as sps

from denoise import _chain_music  # explicit: use music stages regardless of dispatcher

np.random.seed(42)

SRC = Path(r"C:/Robak/DSS2026/Claude/Signal_generation/out/Model/noisy")
DST = Path(r"C:/Robak/DSS2026/Claude/Signal_generation/out/Model/filtered_AI")


def filter_file(path_in: Path, path_out: Path) -> dict:
    x, fs = sf.read(str(path_in), dtype="float64", always_2d=False)
    if x.ndim != 1:
        raise RuntimeError(f"{path_in.name}: expected mono, got shape {x.shape}")

    # build stages at this file's fs (all files here are 22050, but keep general)
    stages = _chain_music(fs)
    # guard: if fs is low enough that the 4410 Hz notch is above Nyquist, bail
    if fs <= 2 * 4410 * 1.02:
        raise RuntimeError(
            f"{path_in.name}: fs={fs} Hz too low for music chain (needs Nyquist > 4410 Hz)"
        )

    y = x
    for _, sos in stages:
        y = sps.sosfiltfilt(sos, y)

    # clip report — PCM_16 needs values in [-1, 1]
    peak_in  = float(np.max(np.abs(x)))
    peak_out = float(np.max(np.abs(y)))
    clipped  = int(np.sum(np.abs(y) > 1.0))

    # write PCM_16 to match input format
    sf.write(str(path_out), y, int(fs), subtype="PCM_16")

    return dict(fs=fs, n=len(x), peak_in=peak_in, peak_out=peak_out, clipped=clipped)


def main():
    if not SRC.is_dir():
        raise SystemExit(f"source missing: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    wavs = sorted(SRC.glob("*.wav"))
    print(f"{len(wavs)} files  src={SRC}  dst={DST}")

    total_clipped = 0
    max_peak = 0.0
    per_prefix = {}

    for i, path_in in enumerate(wavs, 1):
        path_out = DST / path_in.name
        info = filter_file(path_in, path_out)
        total_clipped += info["clipped"]
        max_peak = max(max_peak, info["peak_out"])
        pfx = path_in.stem.rsplit("_", 1)[0]
        per_prefix[pfx] = per_prefix.get(pfx, 0) + 1

        if i % 25 == 0 or i == len(wavs):
            print(f"  [{i:4d}/{len(wavs)}] {path_in.name}  "
                  f"peak_in={info['peak_in']:.3f} peak_out={info['peak_out']:.3f}  "
                  f"clipped={info['clipped']}")

    print(f"\ndone.  per-prefix counts: {per_prefix}")
    print(f"max filtered peak across all files: {max_peak:.3f}")
    print(f"total samples that would clip PCM_16 (|y|>1): {total_clipped}")
    if total_clipped:
        print("  (soundfile clamps these to ±1 in PCM_16 output)")


if __name__ == "__main__":
    main()
