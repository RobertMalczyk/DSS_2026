"""Compare filtered outputs from Signal_detection_RAW with the matched clean
references. One subplot per signal, FFT magnitude in dB."""
import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CLEAN_DIR = os.path.join(OUT, "Samples", "Clean")
FILTERED_DIR = os.path.abspath(
    os.path.join(HERE, "..", "Signal_detection_RAW", "out")
)


def mag_db(x, fs):
    n = x.size
    X = np.fft.rfft(x * np.hanning(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, 20 * np.log10(np.abs(X) / n + 1e-20)


def main():
    tags = ["ecg", "vibration", "music"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    for ax, tag in zip(axes, tags):
        clean_p = os.path.join(CLEAN_DIR, f"{tag}_clean.wav")
        filt_p  = os.path.join(FILTERED_DIR, f"{tag}_filtered.wav")
        clean, fs_c = sf.read(clean_p, dtype="float64")
        filt, fs_f  = sf.read(filt_p,  dtype="float64")

        n = min(clean.size, filt.size)
        if fs_c != fs_f or clean.size != filt.size:
            print(f"[{tag}] fs/length mismatch "
                  f"(clean {clean.size}@{fs_c} vs filt {filt.size}@{fs_f}); "
                  f"truncating to N={n}")
        clean, filt, fs = clean[:n], filt[:n], fs_c

        f_c, clean_db = mag_db(clean, fs)
        f_f, filt_db  = mag_db(filt,  fs)

        ax.plot(f_c, clean_db, lw=0.6, color="C0", label="clean reference")
        ax.plot(f_f, filt_db,  lw=0.6, color="C2", alpha=0.8, label="filtered output")
        ax.set_title(f"{tag} — filtered vs clean FFT  "
                     f"(fs = {fs} Hz, N = {n})")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("|X(f)| [dB]")
        ax.set_xlim(0, fs / 2)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

        err = filt_db - clean_db
        rms_spec_err = float(np.sqrt(np.mean(err ** 2)))
        rms_time_err = float(np.sqrt(np.mean((filt - clean) ** 2)))
        rms_clean    = float(np.sqrt(np.mean(clean ** 2)))
        print(f"[{tag}] N={n} @ {fs} Hz — "
              f"|X_filt - X_clean| RMS (dB) = {rms_spec_err:.2f} ; "
              f"time-domain RMS error = {rms_time_err:.4g} "
              f"({20*np.log10(rms_time_err / rms_clean):+.1f} dB re clean RMS)")

    fig.suptitle("Filtered output vs clean reference — FFT magnitude per signal",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT, "filtered_vs_clean_fft.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
