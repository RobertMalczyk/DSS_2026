"""
Slide 8 figure -- "Recovering the noise spec without reading it."

Three panels on one figure:

  TOP (wide)
    Pooled noise-only PSD across 24 noisy tracks (8 per genre).
    Computed by averaging the |STFT|^2 of the bottom-decile RMS frames
    per file (pseudo-VAD), normalising per file, then taking the
    median across files.  The discovered peaks (find_peaks with a
    prominence threshold) are marked with red dashed lines and labels
    -- these are the lines a downstream filter will notch out.

  BOTTOM-LEFT
    Welch PSD of one starter track (jazz_01) before and after the
    Stage-7 ComponentMatched filter.  Shows the discovered lines
    being carved out and the broadband floor pulled down.

  BOTTOM-RIGHT
    SNR(noisy) vs SNR(stage7) across the 9-track starter set
    (3 each of jazz / metal / pop), with the +8.5 dB delta callout.

Inputs  : Signal_generation/out/Model/{clean,noisy}/*.wav
          Signal_generation/out/Model/Filtered ComponentMatched 7th stage/*.wav
Output  : presentation_peak_discovery.png (next to this script)

The peak discovery and SNR formulas mirror analyze_noise_pattern.py
and filter_stage7_componentmatched.py exactly so the numbers on the
slide match the README under
Signal_detection_assist/Out/COMPONENT_MATCHED_v1/.
"""

import os
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL = HERE / "Signal_generation" / "out" / "Model"
NOISY_DIR = MODEL / "noisy"
CLEAN_DIR = MODEL / "clean"
STAGE7_DIR = MODEL / "Filtered ComponentMatched 7th stage"
STAGE7_PREFIX = "filtered_componentmatched_7th_stage_"

GENRES = ("jazz", "metal", "pop")
N_PER_GENRE = 8
N_FFT = 8192
FS_EXPECTED = 22050

STARTER_FILES = [f"{g}_{i:02d}.wav" for g in GENRES for i in (1, 2, 3)]


def load_mono(path):
    x, fs = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), fs


def low_energy_psd(x, fs, frame_len=N_FFT, hop=N_FFT // 2, pct=10.0):
    """Mean |STFT|^2 over the bottom `pct` percent of frames by energy."""
    n = len(x)
    n_frames = 1 + (n - frame_len) // hop
    win = np.hanning(frame_len)
    energies = np.empty(n_frames)
    mags = np.empty((n_frames, frame_len // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop: i * hop + frame_len] * win
        spec = np.fft.rfft(seg, n=frame_len)
        mag2 = np.abs(spec) ** 2
        mags[i] = mag2
        energies[i] = seg @ seg
    thresh = np.percentile(energies, pct)
    sel = energies <= thresh
    if sel.sum() < 5:
        sel = np.zeros_like(energies, dtype=bool)
        sel[np.argsort(energies)[:5]] = True
    psd = mags[sel].mean(axis=0)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / fs)
    return freqs, psd


def detect_peaks(freqs, psd_db, prominence_db=3.0, min_hz=25.0):
    mask = freqs >= min_hz
    idx, _ = sps.find_peaks(psd_db[mask], prominence=prominence_db)
    out = [(float(freqs[mask][i]), float(psd_db[mask][i])) for i in idx]
    out.sort(key=lambda t: -t[1])
    return out


def snr_db(ref, est):
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64)
    est = est[:n].astype(np.float64)
    return 10.0 * np.log10(ref @ ref / (np.sum((ref - est) ** 2) + 1e-20))


def pool_noise_psd():
    """Pool noise-only PSDs across 8 files per genre, return median."""
    per_file = {g: [] for g in GENRES}
    for g in GENRES:
        files = sorted(NOISY_DIR.glob(f"{g}_*.wav"))[:N_PER_GENRE]
        for p in files:
            x, fs = load_mono(p)
            freqs, psd = low_energy_psd(x, fs)
            per_file[g].append(psd)
    medians = []
    for g in GENRES:
        stack = np.stack(per_file[g])
        stack = stack / stack.sum(axis=1, keepdims=True)
        medians.append(np.median(stack, axis=0))
    pooled = np.median(np.stack(medians), axis=0)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_EXPECTED)
    return freqs, pooled


def select_top_peaks(freqs, pooled_psd_db, k=6):
    """Mirror analyze_noise_pattern: prominence over a 31-bin moving avg."""
    peaks = detect_peaks(freqs, pooled_psd_db, prominence_db=3.0)
    continuum = np.convolve(pooled_psd_db, np.ones(31) / 31, mode="same")
    cleaned = []
    for f, d in peaks:
        i = int(np.argmin(np.abs(freqs - f)))
        prom = d - continuum[i]
        if prom > 2.5:
            cleaned.append((f, d, prom))
    cleaned.sort(key=lambda t: -t[2])
    return cleaned[:k]


def starter_set_snr():
    """Compute (snr_in, snr_out) on the 9 starter tracks, mirroring stage 7."""
    snr_in, snr_out = [], []
    labels = []
    for name in STARTER_FILES:
        clean, _ = load_mono(CLEAN_DIR / name)
        noisy, _ = load_mono(NOISY_DIR / name)
        filt, _ = load_mono(STAGE7_DIR / f"{STAGE7_PREFIX}{name}")
        snr_in.append(snr_db(clean, noisy))
        snr_out.append(snr_db(clean, filt))
        labels.append(name.replace(".wav", ""))
    return np.array(snr_in), np.array(snr_out), labels


def welch_psd(x, fs, nperseg=4096):
    f, P = sps.welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return f, P


def main():
    print("Pooling noise-only PSD across 24 noisy tracks ...")
    freqs, pooled = pool_noise_psd()
    pooled_db = 10 * np.log10(pooled + 1e-18)
    top_peaks = select_top_peaks(freqs, pooled_db, k=6)
    print("  discovered peaks (Hz, dB, prominence):")
    for f, d, p in top_peaks:
        print(f"    {f:8.1f} Hz   {d:+7.2f} dB   prominence {p:+5.2f} dB")

    print("\nLoading one starter track for before/after PSD ...")
    track = "jazz_01.wav"
    clean, fs = load_mono(CLEAN_DIR / track)
    noisy, _ = load_mono(NOISY_DIR / track)
    filt, _ = load_mono(STAGE7_DIR / f"{STAGE7_PREFIX}{track}")

    print("\nComputing SNR across the 9-track starter set ...")
    snr_in, snr_out, labels = starter_set_snr()
    delta_mean = float(np.mean(snr_out) - np.mean(snr_in))
    print(f"  mean noisy  SNR = {np.mean(snr_in):+.2f} dB")
    print(f"  mean stage7 SNR = {np.mean(snr_out):+.2f} dB")
    print(f"  delta            = {delta_mean:+.2f} dB")

    # ----------------------------------- figure ----------------------------- #
    fig = plt.figure(figsize=(13, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    # --- TOP: discovered peaks on the pooled noise PSD ---
    ax_top.semilogy(freqs, pooled, color="black", lw=1.0)
    pmin = max(pooled.min(), 1e-9)
    pmax = pooled.max()
    ax_top.set_xlim(0, FS_EXPECTED / 2)
    ax_top.set_ylim(pmin * 0.7, pmax * 12)
    ax_top.set_xlabel("frequency [Hz]")
    ax_top.set_ylabel("normalised PSD (log)")
    ax_top.set_title("Pooled noise-only PSD across 24 noisy music tracks "
                     "(low-energy frames, no clean reference)",
                     fontsize=11)
    ax_top.grid(True, alpha=0.3, which="both")

    # Numbered markers at each discovered peak, sorted by frequency.
    # The numbers index a clean legend box on the right side of the panel.
    sorted_peaks = sorted(top_peaks, key=lambda t: t[0])
    for n, (f, d, prom) in enumerate(sorted_peaks, start=1):
        ax_top.axvline(f, color="crimson", ls="--", lw=0.9, alpha=0.85)
        ax_top.plot(f, pmax * 4.5, marker="v", color="crimson",
                    markersize=8, clip_on=False)
        ax_top.text(f, pmax * 6.0, str(n),
                    ha="center", va="bottom",
                    fontsize=9, weight="bold", color="crimson")

    legend_lines = ["#  freq [Hz]   prominence over continuum"]
    for n, (f, d, prom) in enumerate(sorted_peaks, start=1):
        legend_lines.append(f"{n}     {f:7.1f}        +{prom:5.2f} dB")
    legend_text = "\n".join(legend_lines)
    ax_top.text(
        0.985, 0.97, legend_text,
        transform=ax_top.transAxes,
        ha="right", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="crimson", alpha=0.95),
    )

    spec_text = (
        "spec (noise.md — NOT consulted): mains 50/100/150/200 Hz, "
        "switching tone @ 4410 Hz\n"
        "every coefficient below was derived from the noisy audio alone"
    )
    ax_top.text(
        0.015, 0.97, spec_text,
        transform=ax_top.transAxes,
        ha="left", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0",
                  ec="0.6", alpha=0.95),
    )

    # --- BOTTOM-LEFT: stage 7 carving the spectrum on jazz_01 ---
    f_n, P_n = welch_psd(noisy, fs)
    f_f, P_f = welch_psd(filt, fs)
    ax_bl.semilogy(f_n, P_n, color="C1", lw=1.0, label="noisy", alpha=0.85)
    ax_bl.semilogy(f_f, P_f, color="C0", lw=1.0, label="stage 7 filtered")
    for f, _, _ in top_peaks:
        ax_bl.axvline(f, color="crimson", ls=":", lw=0.7, alpha=0.55)
    ax_bl.set_xlim(0, FS_EXPECTED / 2)
    ax_bl.set_xlabel("frequency [Hz]")
    ax_bl.set_ylabel("Welch PSD (log)")
    ax_bl.set_title(f"Welch PSD before / after Stage 7 — {track}",
                    fontsize=10)
    ax_bl.legend(loc="upper right", fontsize=9)
    ax_bl.grid(True, alpha=0.3, which="both")

    # --- BOTTOM-RIGHT: SNR bar comparison on starter set ---
    x = np.arange(len(labels))
    width = 0.4
    ax_br.bar(x - width / 2, snr_in, width, label="noisy",
              color="C1", alpha=0.85)
    ax_br.bar(x + width / 2, snr_out, width, label="stage 7",
              color="C0")
    ax_br.axhline(0, color="0.4", lw=0.6)
    ax_br.set_xticks(x)
    ax_br.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_br.set_ylabel("SNR [dB]")
    ax_br.set_title("SNR before / after Stage 7 on 9-track starter set",
                    fontsize=10)
    ax_br.grid(True, alpha=0.3, axis="y")
    ax_br.legend(loc="lower right", fontsize=9)

    badge = (
        f"mean noisy:  {np.mean(snr_in):+.2f} dB\n"
        f"mean stage7: {np.mean(snr_out):+.2f} dB\n"
        f"--> +{delta_mean:.2f} dB"
    )
    ax_br.text(
        0.02, 0.97, badge,
        transform=ax_br.transAxes,
        ha="left", va="top",
        fontsize=10, family="monospace", weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ddffdd",
                  ec="0.5", alpha=0.95),
    )

    fig.suptitle(
        "Recovering the noise spec without reading it  —  "
        "peaks discovered from noisy audio drive a Stage 7 chain that "
        f"adds {delta_mean:+.1f} dB SNR",
        fontsize=12, weight="bold",
    )

    out_path = HERE / "presentation_peak_discovery.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
