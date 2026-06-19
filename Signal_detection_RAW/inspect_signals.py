"""Inspect the three noisy signals: time, PSD, spectrogram, basic stats.

Writes out/inspect_<name>.png and prints measurements to stdout.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal as sps
import matplotlib.pyplot as plt

np.random.seed(42)

ROOT = Path(__file__).parent
SAMPLES = ROOT.parent / "Signal_generation" / "out" / "Samples"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

FILES = {
    "ecg": SAMPLES / "ecg_noisy.wav",
    "vibration": SAMPLES / "vibration_noisy.wav",
    "music": SAMPLES / "music_noisy.wav",
}


def describe(name, x, fs):
    n = len(x)
    dur = n / fs
    mu, sd = x.mean(), x.std()
    p50, p95, p99 = np.percentile(np.abs(x), [50, 95, 99])
    print(f"\n=== {name} ===")
    print(f"  fs={fs} Hz  n={n}  dur={dur:.3f} s")
    print(f"  mean={mu:+.4e}  std={sd:.4e}  min={x.min():+.4e}  max={x.max():+.4e}")
    print(f"  |x| p50={p50:.4e}  p95={p95:.4e}  p99={p99:.4e}  peak/std={np.abs(x).max()/sd:.2f}")

    # Welch PSD, long window for good resolution
    nper = min(4096, n // 4) if n > 4096 else n // 2
    f, Pxx = sps.welch(x, fs, nperseg=nper, window="hann", detrend=False)
    # log PSD
    Pdb = 10 * np.log10(Pxx + 1e-30)

    # locate tonal peaks: PSD lines > median + 15 dB
    med = np.median(Pdb)
    peaks_idx, props = sps.find_peaks(Pdb, height=med + 15, distance=3)
    peaks = sorted(
        [(f[i], Pdb[i]) for i in peaks_idx], key=lambda t: -t[1]
    )[:15]
    print(f"  PSD median={med:.1f} dB   tonal peaks (>med+15 dB):")
    for fp, pdb in peaks:
        print(f"    {fp:9.2f} Hz   {pdb:+6.1f} dB")

    # broadband floor near Nyquist (upper quartile of freqs)
    hi = f > 0.75 * fs / 2
    if hi.any():
        floor_hi = np.median(Pdb[hi])
        print(f"  median PSD in top 25% of band: {floor_hi:.1f} dB")

    # DC / low-freq content
    lo = f < 0.02 * fs / 2
    if lo.any():
        print(f"  median PSD in bottom 2% of band: {np.median(Pdb[lo]):.1f} dB")

    # Figure
    t = np.arange(n) / fs
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))
    ax[0].plot(t, x, lw=0.4)
    ax[0].set_title(f"{name} noisy — time domain")
    ax[0].set_xlabel("s")
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(f, Pxx + 1e-30)
    ax[1].set_title("Welch PSD")
    ax[1].set_xlabel("Hz")
    ax[1].set_ylabel("PSD")
    ax[1].grid(which="both", alpha=0.3)

    nfft_sg = min(1024, n // 8)
    f_sg, t_sg, S = sps.spectrogram(
        x, fs, nperseg=nfft_sg, noverlap=nfft_sg // 2, window="hann"
    )
    ax[2].pcolormesh(t_sg, f_sg, 10 * np.log10(S + 1e-30), shading="auto", cmap="magma")
    ax[2].set_title("Spectrogram (dB)")
    ax[2].set_xlabel("s")
    ax[2].set_ylabel("Hz")

    fig.tight_layout()
    fig.savefig(OUT / f"inspect_{name}.png", dpi=110)
    plt.close(fig)

    return f, Pxx, peaks


def main():
    for name, path in FILES.items():
        x, fs = sf.read(path, dtype="float64")
        assert x.ndim == 1, f"{name} not mono"
        describe(name, x, fs)


if __name__ == "__main__":
    main()
