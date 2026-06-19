"""Stage 1 filter: zero-phase Butterworth low-pass at 3 kHz, effective order 10.

Design:
  - 5th-order Butterworth prototype (low distortion: flat passband, no ripple).
  - Applied via forward-backward filtering (scipy.signal.sosfiltfilt).
  - Forward + backward passes cancel phase -> 0 degrees at every frequency.
  - Effective magnitude response = squared prototype = 10th-order rolloff.

Input : every .wav in C:\\Robak\\DSS2026\\Claude\\Signal_generation\\out\\Model\\noisy
Output: same filename with prefix "filtered_assist_1st_stage_" in
        C:\\Robak\\DSS2026\\Claude\\Signal_detection_assist\\Out\\Filtered assist 1st stage
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

IN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\Filtered assist 1st stage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_HZ = 3000.0
PROTOTYPE_ORDER = 5     # effective order after filtfilt = 2 * 5 = 10
PREFIX = "filtered_assist_1st_stage_"


def design_filter(sr: int):
    return butter(PROTOTYPE_ORDER, CUTOFF_HZ, btype="low", fs=sr, output="sos")


def apply_zero_phase(sos, x):
    # sosfiltfilt handles multi-channel along axis=0 if given 2D (n_samples, n_channels).
    return sosfiltfilt(sos, x, axis=0)


def process_one(path: Path, sos_cache: dict):
    info = sf.info(str(path))
    sr = info.samplerate
    if sr not in sos_cache:
        sos_cache[sr] = design_filter(sr)
    sos = sos_cache[sr]

    x, sr_read = sf.read(str(path), always_2d=False)
    assert sr_read == sr
    y = apply_zero_phase(sos, x).astype(x.dtype, copy=False)

    out_path = OUT_DIR / f"{PREFIX}{path.name}"
    sf.write(str(out_path), y, sr, subtype=info.subtype)
    return out_path, sr, len(x)


def main():
    files = sorted(IN_DIR.glob("*.wav"))
    print(f"Found {len(files)} wav files in {IN_DIR}")
    print(f"Filter: Butterworth prototype order {PROTOTYPE_ORDER}, cutoff {CUTOFF_HZ:.0f} Hz, "
          f"zero-phase via sosfiltfilt (effective order {2*PROTOTYPE_ORDER}).")
    print(f"Output -> {OUT_DIR}")
    sos_cache: dict = {}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        out_path, sr, n = process_one(p, sos_cache)
        if i % 20 == 0 or i == len(files):
            print(f"  [{i:3d}/{len(files)}] {out_path.name}  (sr={sr}, {n} samples)")
    print(f"Done in {time.time()-t0:.1f}s. Wrote {len(files)} files.")


if __name__ == "__main__":
    main()
