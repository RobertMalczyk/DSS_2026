"""Detailed inspection pass: find PSD knees, catalogue tonals, examine time shape."""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal as sps
import matplotlib.pyplot as plt

np.random.seed(42)

ROOT = Path(__file__).parent
SAMPLES = ROOT.parent / "Signal_generation" / "out" / "Samples"
OUT = ROOT / "out"

FILES = {
    "ecg": SAMPLES / "ecg_noisy.wav",
    "vibration": SAMPLES / "vibration_noisy.wav",
    "music": SAMPLES / "music_noisy.wav",
}


def detailed(name, x, fs):
    n = len(x)
    print(f"\n=== {name}  fs={fs}  n={n} ===")

    # Remove mean for spectrum so DC does not dominate median estimates
    xz = x - x.mean()

    # Fine Welch for tonal detection
    nper = min(8192, n // 2) if n > 8192 else n // 2
    f, Pxx = sps.welch(xz, fs, nperseg=nper, window="hann", detrend=False)
    Pdb = 10 * np.log10(Pxx + 1e-30)

    # robust floor: median in a moving window (to get local floor near each tonal)
    #   gives a better threshold than global median
    win = max(5, len(f) // 50)
    from scipy.ndimage import median_filter
    local_floor = median_filter(Pdb, size=win)

    excess = Pdb - local_floor
    thr = 12.0  # dB above local floor

    peaks_idx, _ = sps.find_peaks(excess, height=thr, distance=2)
    peaks = sorted([(f[i], Pdb[i], excess[i]) for i in peaks_idx], key=lambda t: -t[2])
    print(f"  tonals (>local floor + {thr} dB), sorted by excess:")
    for fp, pdb, ex in peaks[:20]:
        print(f"    {fp:9.2f} Hz   {pdb:+6.1f} dB   +{ex:4.1f} dB over local")

    # Where does broadband energy drop? find smoothed PSD and the frequency at which
    # smoothed PSD falls below the floor + N dB above its minimum value.
    smoothed = median_filter(Pdb, size=max(5, len(f) // 20))
    # Sort freqs and find the first spot (above a safe low-freq margin) where
    # smoothed is within +3 dB of the tail median.
    tail = smoothed[f > 0.6 * fs / 2]
    tail_med = np.median(tail)
    # Knee: smallest freq beyond which smoothed stays within 3 dB of tail median.
    within = (smoothed <= tail_med + 3) & (f > fs / 50)
    knee = None
    for i in range(len(f)):
        if within[i] and within[i:].all():
            knee = f[i]
            break
    print(f"  tail smoothed-PSD median (upper 40% band): {tail_med:.1f} dB")
    print(f"  PSD knee (smoothed PSD descends to within 3 dB of tail): "
          f"{knee:.2f} Hz" if knee else "  no clear knee")

    # Mean and low-freq structure
    print(f"  mean={x.mean():+.4e}  std(zero-mean)={xz.std():.4e}")

    # Draw a reference plot with floor + knee
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(f, Pdb, lw=0.6, label="PSD")
    ax[0].plot(f, local_floor, lw=0.8, label="local floor (median filter)")
    if knee:
        ax[0].axvline(knee, color="r", ls="--", label=f"knee ~{knee:.0f} Hz")
    for fp, pdb, ex in peaks[:10]:
        ax[0].plot(fp, pdb, "rx", ms=6)
    ax[0].set_title(f"{name} PSD detail")
    ax[0].set_xlabel("Hz")
    ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # log-x view helps for music/vibration
    ax[1].semilogx(f[1:], Pdb[1:], lw=0.6)
    ax[1].semilogx(f[1:], local_floor[1:], lw=0.8)
    for fp, pdb, _ in peaks[:10]:
        ax[1].plot(fp, pdb, "rx", ms=6)
    ax[1].set_title("PSD log-x")
    ax[1].set_xlabel("Hz (log)")
    ax[1].set_ylabel("dB")
    ax[1].grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / f"inspect_detail_{name}.png", dpi=110)
    plt.close(fig)

    return {"knee": knee, "tail_db": tail_med, "peaks": peaks[:20]}


def main():
    for name, path in FILES.items():
        x, fs = sf.read(path, dtype="float64")
        detailed(name, x, fs)


if __name__ == "__main__":
    main()
