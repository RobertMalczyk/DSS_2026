"""One-shot: magnitude FFT of a single noise realisation from make_noise()."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signal_contamination import make_noise, SNR_TARGET_DB, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# Pick one representative configuration. The noise model needs fs and n to
# size its filters and burst schedule; the clean array is used only to set
# the final α-scale. Using a unit-RMS dummy makes the noise amplitude
# independent of any particular signal.
FS = 12_000
DURATION_S = 3.0


def main():
    n = int(FS * DURATION_S)
    rng = np.random.default_rng(SEED)
    dummy_clean = rng.standard_normal(n)              # unit-RMS, unit-variance
    _, _, noise = make_noise(dummy_clean, FS)

    X = np.fft.rfft(noise * np.hanning(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    mag_db = 20 * np.log10(np.abs(X) / n + 1e-20)

    rms = float(np.sqrt(np.mean(noise ** 2)))
    peak = float(np.max(np.abs(noise)))
    print(f"Noise realisation: fs={FS} Hz, N={n}, duration={DURATION_S} s, "
          f"RMS={rms:.4g}, peak={peak:.4g}")
    print(f"FFT peak |N(f)| = {mag_db.max():.1f} dB at {f[mag_db.argmax()]:.2f} Hz, "
          f"median = {np.median(mag_db):.1f} dB")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(f, mag_db, lw=0.5, color="C3")
    ax.set_xlim(0, FS / 2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|N(f)| [dB]")
    ax.set_title(f"Noise FFT  (fs = {FS} Hz, N = {n}, "
                 f"target SNR {SNR_TARGET_DB:+.1f} dB against unit-RMS dummy)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    out_path = os.path.join(OUT, "noise_fft.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
