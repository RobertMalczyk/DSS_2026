"""
Characterise the noise pattern directly from the noisy music files.

Non-oracle: reads only `out/Model/noisy/*.wav`. The clean signal is not
touched. Goal is to empirically identify which noise components are
present, at which frequencies, and with which temporal behaviour, so a
targeted filter can be designed afterwards.

Outputs go to:
    Signal_detection_assist/Out/NOISE_PATTERN_ANALYSIS/
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, welch

NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\NOISE_PATTERN_ANALYSIS")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_EXPECTED = 22050
N_FFT = 8192  # high freq resolution: ~2.7 Hz/bin at 22050
N_PER_GENRE = 8
GENRES = ("jazz", "metal", "pop")

# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def pick_files() -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {g: [] for g in GENRES}
    for g in GENRES:
        cands = sorted(NOISY_DIR.glob(f"{g}_*.wav"))[:N_PER_GENRE]
        files[g] = cands
    return files


def load(path: Path) -> tuple[np.ndarray, int]:
    x, fs = sf.read(str(path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == FS_EXPECTED, f"{path}: fs={fs}"
    return x.astype(np.float64), fs


# --------------------------------------------------------------------------- #
# Low-energy frame selection (pseudo-VAD) — used to isolate noise-dominated   #
# frames from each noisy file without any reference to the clean signal.      #
# --------------------------------------------------------------------------- #
def low_energy_psd(x: np.ndarray, fs: int,
                   frame_len: int = 2048, hop: int = 1024,
                   pct: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Time-average |STFT|² over the bottom `pct` percent of frames by energy."""
    n = len(x)
    n_frames = 1 + (n - frame_len) // hop
    win = np.hanning(frame_len)
    energies = np.empty(n_frames)
    mags = np.empty((n_frames, frame_len // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame_len] * win
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


# --------------------------------------------------------------------------- #
# Narrowband line detection (hum, switching tone)                             #
# --------------------------------------------------------------------------- #
def detect_peaks(freqs: np.ndarray, psd_db: np.ndarray,
                 prominence_db: float = 6.0, min_hz: float = 25.0
                 ) -> list[tuple[float, float]]:
    mask = freqs >= min_hz
    idx, _ = find_peaks(psd_db[mask], prominence=prominence_db)
    out = [(float(freqs[mask][i]), float(psd_db[mask][i])) for i in idx]
    out.sort(key=lambda t: -t[1])
    return out


# --------------------------------------------------------------------------- #
# Impulsive-burst detection via smoothed-envelope thresholding                #
# --------------------------------------------------------------------------- #
def burst_stats(x: np.ndarray, fs: int) -> dict:
    """Count & size of bursts using a 5-ms-window envelope + MAD threshold."""
    w = int(0.005 * fs)
    # RMS envelope
    padded = np.pad(x**2, (w // 2, w - w // 2 - 1), mode="edge")
    kernel = np.ones(w) / w
    env = np.sqrt(np.convolve(padded, kernel, mode="valid"))
    # Robust baseline
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    # Bursts are samples where envelope > med + k·MAD
    k = 8.0
    thresh = med + k * mad
    above = env > thresh
    # Merge adjacent samples into events (≥2 ms gap separates events)
    gap = int(0.002 * fs)
    events: list[tuple[int, int]] = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and (above[j] or np.any(above[j : min(j + gap, n)])):
                j += 1
            events.append((i, j))
            i = j
        else:
            i += 1
    durations_ms = [(b - a) / fs * 1000 for a, b in events]
    peak_ratios = [float(env[a:b].max() / (med + 1e-12)) for a, b in events]
    return {
        "n_events": len(events),
        "rate_per_s": len(events) / (len(x) / fs),
        "median_duration_ms": float(np.median(durations_ms)) if events else 0.0,
        "median_peak_ratio": float(np.median(peak_ratios)) if events else 0.0,
        "envelope_median": float(med),
        "envelope_mad": float(mad),
        "envelope_threshold": float(thresh),
    }


# --------------------------------------------------------------------------- #
# Low-frequency drift content                                                 #
# --------------------------------------------------------------------------- #
def drift_fraction(freqs: np.ndarray, psd: np.ndarray,
                   cutoff_hz: float = 30.0) -> float:
    """Fraction of total noise-frame power below `cutoff_hz`."""
    total = np.trapezoid(psd, freqs)
    lf = np.trapezoid(psd[freqs <= cutoff_hz], freqs[freqs <= cutoff_hz])
    return float(lf / total) if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    files = pick_files()
    per_file_peaks: list[dict] = []
    per_file_bursts: list[dict] = []
    per_file_drift: list[dict] = []
    pooled_psd: dict[str, list[np.ndarray]] = {g: [] for g in GENRES}

    for genre, paths in files.items():
        for p in paths:
            x, fs = load(p)
            freqs, psd = low_energy_psd(x, fs, frame_len=N_FFT, hop=N_FFT // 2)
            pooled_psd[genre].append(psd)

            psd_db = 10 * np.log10(psd + 1e-12)
            peaks = detect_peaks(freqs, psd_db, prominence_db=6.0)
            bursts = burst_stats(x, fs)
            lf_frac = drift_fraction(freqs, psd, cutoff_hz=30.0)

            per_file_peaks.append(
                {
                    "file": p.stem,
                    "genre": genre,
                    "top_peaks_hz": ";".join(f"{f:.1f}" for f, _ in peaks[:10]),
                    "top_peaks_db": ";".join(f"{d:.1f}" for _, d in peaks[:10]),
                }
            )
            per_file_bursts.append({"file": p.stem, "genre": genre, **bursts})
            per_file_drift.append(
                {"file": p.stem, "genre": genre, "drift_frac_below_30Hz": lf_frac}
            )

    # Pool PSDs across files, normalise per-file, median across genre
    median_psd = {}
    for g in GENRES:
        stack = np.stack(pooled_psd[g])
        stack = stack / stack.sum(axis=1, keepdims=True)  # per-file unit-sum
        median_psd[g] = np.median(stack, axis=0)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_EXPECTED)

    # Overall median across genres (since noise is genre-invariant)
    global_med = np.median(
        np.stack([median_psd[g] for g in GENRES]), axis=0
    )
    global_med_db = 10 * np.log10(global_med + 1e-18)

    # --------------------------------------------------------------------- #
    # PLOT 1: pooled noise-frame PSD with detected lines                    #
    # --------------------------------------------------------------------- #
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    ax = axes[0]
    for g in GENRES:
        ax.semilogy(freqs, median_psd[g], label=f"{g} (median of {N_PER_GENRE})",
                    alpha=0.7)
    ax.semilogy(freqs, global_med, color="k", lw=2, label="pooled median")
    ax.set_xlim(0, FS_EXPECTED / 2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("normalised PSD")
    ax.set_title("Low-energy-frame PSD (noise-dominated), pooled across files")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.semilogy(freqs, global_med, color="k", lw=1.5)
    ax.set_xlim(0, 500)  # zoom to mains + harmonics
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("normalised PSD")
    ax.set_title("Zoom 0–500 Hz — 50 Hz mains harmonics visible?")
    ax.grid(True, alpha=0.3)
    for h in (50, 100, 150, 200, 250, 300):
        ax.axvline(h, color="r", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_pooled_noise_psd.png", dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------------- #
    # PLOT 2: zoom near expected switching tone at 0.4*Nyquist = 4410 Hz?   #
    #        (audio in upstream doc says 9600 Hz for fs=48000. Here fs is   #
    #         22050 so expected f_sw = 0.4 * 11025 = 4410 Hz.)              #
    # --------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(freqs, global_med, color="k", lw=1.2)
    ax.set_xlim(3500, 5500)
    ax.axvline(4410, color="orange", ls="--", label="expected f_sw = 0.4·Nyq")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("normalised PSD")
    ax.set_title("Zoom 3.5–5.5 kHz — switching tone?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_switching_zoom.png", dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------------- #
    # PLOT 3: burst rate distribution per file                              #
    # --------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(10, 4))
    rates = [b["rate_per_s"] for b in per_file_bursts]
    colors = {"jazz": "C0", "metal": "C3", "pop": "C2"}
    xs = np.arange(len(per_file_bursts))
    ax.bar(xs, rates,
           color=[colors[b["genre"]] for b in per_file_bursts])
    ax.axhline(3.0, color="k", ls="--",
               label="spec (3/s from noise.md)")
    ax.set_xticks(xs)
    ax.set_xticklabels([b["file"] for b in per_file_bursts],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("detected bursts / s (env > med+8·MAD)")
    ax.set_title("Impulsive-burst rate per file")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_burst_rates.png", dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------------- #
    # PLOT 4: drift fraction per file                                       #
    # --------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(10, 4))
    fracs = [d["drift_frac_below_30Hz"] for d in per_file_drift]
    ax.bar(xs, fracs,
           color=[colors[d["genre"]] for d in per_file_drift])
    ax.set_xticks(xs)
    ax.set_xticklabels([d["file"] for d in per_file_drift],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("fraction of noise-frame power below 30 Hz")
    ax.set_title("Low-frequency drift content (1/f² suspected)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_drift_fraction.png", dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------------- #
    # Detect global peaks in pooled PSD — these should be present in every  #
    # file (hum + switching) and thus be the safest targets for a notch.    #
    # --------------------------------------------------------------------- #
    global_peaks = detect_peaks(freqs, global_med_db, prominence_db=3.0)
    # Keep only peaks clearly above continuum
    continuum = np.convolve(
        global_med_db, np.ones(31) / 31, mode="same"
    )
    clean_peaks = []
    for f, d in global_peaks:
        i = int(np.argmin(np.abs(freqs - f)))
        if d - continuum[i] > 2.5:
            clean_peaks.append((f, d, d - continuum[i]))
    clean_peaks.sort(key=lambda t: -t[2])

    # --------------------------------------------------------------------- #
    # Write CSV summaries                                                   #
    # --------------------------------------------------------------------- #
    with (OUT_DIR / "per_file_peaks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_file_peaks[0]))
        w.writeheader()
        w.writerows(per_file_peaks)
    with (OUT_DIR / "per_file_bursts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_file_bursts[0]))
        w.writeheader()
        w.writerows(per_file_bursts)
    with (OUT_DIR / "per_file_drift.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_file_drift[0]))
        w.writeheader()
        w.writerows(per_file_drift)

    # --------------------------------------------------------------------- #
    # README with findings                                                  #
    # --------------------------------------------------------------------- #
    lines = [
        "NOISE PATTERN ANALYSIS — read from noisy music files only (non-oracle)",
        "=" * 72,
        "",
        f"Sample: {N_PER_GENRE} files x {len(GENRES)} genres = "
        f"{N_PER_GENRE * len(GENRES)} files.",
        f"Method: low-energy frame (bottom 10% by RMS) PSD pooling, "
        f"per-file unit-sum normalisation, median across files.",
        "",
        "----- TOP GLOBAL SPECTRAL LINES (pooled across all files) -----",
        "freq_Hz   | PSD_dB   | prominence_over_continuum_dB",
    ]
    for f, d, p in clean_peaks[:15]:
        lines.append(f"{f:8.1f}  | {d:7.2f}  | {p:6.2f}")

    # Burst summary
    mean_rate = np.mean([b["rate_per_s"] for b in per_file_bursts])
    med_dur = np.median([b["median_duration_ms"] for b in per_file_bursts])
    mean_drift = np.mean([d["drift_frac_below_30Hz"] for d in per_file_drift])
    lines.extend(
        [
            "",
            "----- IMPULSIVE BURST ESTIMATE (envelope > median + 8·MAD) -----",
            f"mean rate across files          : {mean_rate:.2f} /s "
            f"(spec says 3 /s)",
            f"median event duration           : {med_dur:.1f} ms "
            f"(spec says ~20 ms incl. tail)",
            "",
            "----- LOW-FREQUENCY DRIFT -----",
            f"mean fraction of noise power    : "
            f"{mean_drift*100:.1f}% below 30 Hz",
            "",
            "----- SANITY CROSS-CHECK vs noise.md -----",
            "noise.md says the composite noise contains:",
            "  1. mains 50 Hz + harmonics {100, 150, 200}",
            "  2. switching tone at 0.4 * Nyquist (= 4410 Hz @ fs=22050)",
            "  3. broadband EMI, bandpassed [0.05, 0.95]*Nyquist",
            "  4. impulsive bursts at ~3 /s",
            "  5. 1/f^2 drift (dominant low-frequency)",
            "  6. 8-bit quantisation (negligible)",
            "  7. +/- 3*peak clipping",
            "",
            "Look for these in the plots and the peak list above. Any ",
            "narrowband line that lines up with 50,100,150,200 Hz (or ",
            "~4410 Hz) confirms component 1/2 and justifies a notch.",
        ]
    )
    (OUT_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
