"""2nd-stage batch filter for the DSS2026 Model dataset.

Pipeline per file: noisy -> music chain (from denoise.py) -> additional
zero-phase IIR LP with effective 10th-order magnitude (= Butter ord 5 SOS
applied with sosfiltfilt, cutoff 3 kHz).

Source:  ..\\Signal_generation\\out\\Model\\noisy
Dest:    ..\\Signal_generation\\out\\Model\\Filtered assist and AI 2nd stage

The music chain in denoise.py is not modified — the extra LP lives only
here, as a 2nd-stage on top.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal as sps

from denoise import _chain_music

np.random.seed(42)

SRC = Path(r"C:/Robak/DSS2026/Claude/Signal_generation/out/Model/noisy")
DST = Path(r"C:/Robak/DSS2026/Claude/Signal_generation/out/Model/Filtered assist and AI 2nd stage")

# effective 10th-order zero-phase LP => order 5 SOS + sosfiltfilt
EXTRA_LP_ORDER_SINGLE = 5            # effective doubles under sosfiltfilt
EXTRA_LP_CUTOFF_HZ   = 3000.0


def extra_lp_sos(fs):
    return sps.butter(EXTRA_LP_ORDER_SINGLE, EXTRA_LP_CUTOFF_HZ,
                      btype="low", fs=fs, output="sos")


def filter_file(path_in: Path, path_out: Path) -> dict:
    x, fs = sf.read(str(path_in), dtype="float64", always_2d=False)
    if x.ndim != 1:
        raise RuntimeError(f"{path_in.name}: expected mono, got {x.shape}")
    if fs <= 2 * EXTRA_LP_CUTOFF_HZ * 1.02:
        raise RuntimeError(
            f"{path_in.name}: fs={fs} Hz too low for 3 kHz LP (needs Nyquist > 3 kHz)"
        )

    y = x
    # stage 1: music chain (unchanged from denoise.py)
    for _, sos in _chain_music(fs):
        y = sps.sosfiltfilt(sos, y)
    # stage 2: extra LP, effective 10th order, zero-phase
    y = sps.sosfiltfilt(extra_lp_sos(fs), y)

    clipped = int(np.sum(np.abs(y) > 1.0))
    sf.write(str(path_out), y, int(fs), subtype="PCM_16")

    return dict(fs=fs, n=len(x),
                peak_in=float(np.max(np.abs(x))),
                peak_out=float(np.max(np.abs(y))),
                clipped=clipped)


def main():
    if not SRC.is_dir():
        raise SystemExit(f"source missing: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    wavs = sorted(SRC.glob("*.wav"))
    print(f"{len(wavs)} files")
    print(f"  src = {SRC}")
    print(f"  dst = {DST}")
    print(f"  stage-2 LP: Butter ord {EXTRA_LP_ORDER_SINGLE} @ "
          f"{EXTRA_LP_CUTOFF_HZ:.0f} Hz  (effective ord "
          f"{2 * EXTRA_LP_ORDER_SINGLE} magnitude under sosfiltfilt, phase = 0)")

    total_clipped = 0
    max_peak = 0.0
    for i, path_in in enumerate(wavs, 1):
        info = filter_file(path_in, DST / path_in.name)
        total_clipped += info["clipped"]
        max_peak = max(max_peak, info["peak_out"])
        if i % 25 == 0 or i == len(wavs):
            print(f"  [{i:4d}/{len(wavs)}] {path_in.name}  "
                  f"peak_in={info['peak_in']:.3f} peak_out={info['peak_out']:.3f}  "
                  f"clipped={info['clipped']}")

    print(f"\ndone.")
    print(f"  max filtered peak: {max_peak:.3f}")
    print(f"  total PCM_16 clip samples: {total_clipped}")


if __name__ == "__main__":
    main()
