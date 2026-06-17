"""Per-signal FFT overlay: clean signal vs the noise added to it.

Shows directly which parts of each signal's spectrum survive the noise and
which parts get buried.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signal_contamination import (
    load_ecg, load_vibration, load_music, make_noise, SNR_TARGET_DB,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def mag_db(x, fs):
    n = x.size
    X = np.fft.rfft(x * np.hanning(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, 20 * np.log10(np.abs(X) / n + 1e-20)


def main():
    loaders = [
        ("ECG",       load_ecg,       "mV"),
        ("Vibration", load_vibration, "g"),
        ("Music",     load_music,     "amplitude"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    for ax, (name, fn, units) in zip(axes, loaders):
        clean, fs, *_ = fn()
        _, _, noise = make_noise(clean, fs)

        f_s, sig_db = mag_db(clean, fs)
        f_n, n_db   = mag_db(noise, fs)

        ax.plot(f_s, sig_db, lw=0.6, color="C0",
                label=f"clean signal  (RMS {np.sqrt(np.mean(clean**2)):.3g} {units})")
        ax.plot(f_n, n_db, lw=0.6, color="C3", alpha=0.8,
                label=f"noise  (RMS {np.sqrt(np.mean(noise**2)):.3g} {units})")
        ax.set_title(f"{name} — clean vs noise FFT  "
                     f"(fs = {fs} Hz, SNR target {SNR_TARGET_DB:+.1f} dB)")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(f"|X(f)| [dB re 1 {units}]")
        ax.set_xlim(0, fs / 2)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

        buried = (n_db > sig_db).sum()
        print(f"[{name}] fs={fs} Hz, N={clean.size} — "
              f"noise bin > signal bin in {buried}/{len(f_s)} bins "
              f"({100 * buried / len(f_s):.1f}%)")

    fig.suptitle("How the noise covers each clean signal in the frequency domain",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT, "signal_vs_noise_fft.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
