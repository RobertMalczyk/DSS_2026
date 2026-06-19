"""Continuous-wavelet scalograms (time-frequency wavelet spectrograms).

Two comparisons are produced per file:
  1. compare_wavelets_<name>.png   - 5 different mother wavelets, same freq range
  2. compare_cmor_<name>.png       - complex Morlet with 4 bandwidth settings

Audio is trimmed to a 5 s mid-segment to keep CWT runtimes reasonable
(CWT is O(N * num_scales); on full 30 s at 22050 Hz each plot can take minutes).
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

IN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\WAVELET")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["jazz_01.wav", "metal_01.wav", "pop_01.wav"]

SEGMENT_SEC = 5.0
SEGMENT_START_SEC = 10.0  # skip potential silence / intro

NUM_SCALES = 96
F_MIN = 40.0
F_MAX = 8000.0

# 5 wavelet families for the main comparison.
WAVELETS = [
    "cmor1.5-1.0",   # complex Morlet, moderate bandwidth - the MIR workhorse
    "morl",          # real Morlet, no bandwidth param
    "mexh",          # Mexican hat (Ricker) - narrow time support, broad freq
    "gaus4",         # 4th derivative of Gaussian - oscillatory, real-valued
    "cgau4",         # complex 4th Gaussian derivative - analytic version of gaus4
]

# cmor bandwidth sweep (bandwidth B - center freq C). Larger B -> longer in time, narrower in freq.
CMOR_VARIANTS = [
    "cmor0.5-1.0",   # very narrow in time, broad in frequency
    "cmor1.5-1.0",   # default-ish
    "cmor3.0-1.0",   # broader time support, sharper frequency
    "cmor6.0-1.0",   # very long time support, near pure tones
]


def load_segment(path: Path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    start = int(SEGMENT_START_SEC * sr)
    end = start + int(SEGMENT_SEC * sr)
    end = min(end, len(x))
    return x[start:end], sr, start / sr


def freqs_to_scales(wavelet, freqs, sr):
    # scale = central_freq(wavelet) / (f * dt) = central_freq * sr / f
    fc = pywt.central_frequency(wavelet)
    return fc * sr / freqs


def cwt_scalogram_db(x, sr, wavelet):
    freqs = np.geomspace(F_MIN, F_MAX, NUM_SCALES)
    scales = freqs_to_scales(wavelet, freqs, sr)
    t0 = time.time()
    coefs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / sr)
    dt = time.time() - t0
    mag = np.abs(coefs)
    ref = mag.max() if mag.max() > 0 else 1.0
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
    db = np.maximum(db, -80.0)
    return freqs, db, dt


def plot_panels(name, sr, seg_start, panels, subtitle, out_path):
    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    time_axis = seg_start + np.arange(panels[0][2].shape[1]) / sr
    im = None
    for ax, (label, freqs, db, cwt_time) in zip(axes, panels):
        im = ax.pcolormesh(time_axis, freqs, db, shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        ax.set_yscale("log")
        ax.set_ylim(F_MIN, F_MAX)
        ax.set_title(f"{label}   (cwt {cwt_time:.1f} s)", fontsize=10)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{subtitle} - {name}  (segment {seg_start:.1f}-{seg_start+SEGMENT_SEC:.1f}s)",
                 fontsize=13)
    fig.colorbar(im, ax=axes.tolist(), label="dB (rel. max)", shrink=0.8)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_notes(sr):
    path = OUT_DIR / "README.txt"
    lines = [
        "Continuous wavelet transform (CWT) scalograms",
        "=============================================",
        "",
        f"Source files: jazz_01.wav, metal_01.wav, pop_01.wav (sr = {sr} Hz).",
        f"Segment analyzed: {SEGMENT_START_SEC:.1f} s to {SEGMENT_START_SEC+SEGMENT_SEC:.1f} s (5 s slice).",
        f"Frequency grid: {NUM_SCALES} log-spaced scales, {F_MIN:.0f} Hz to {F_MAX:.0f} Hz.",
        "Color: |CWT coefficient| in dB, relative to per-plot max, floor -80 dB.",
        "",
        "Why only 5 seconds?",
        "  CWT is O(N * num_scales). At 22 050 Hz, a 30 s clip with 96 scales takes",
        "  significantly longer than STFT. A 5 s central slice is enough to judge",
        "  time-frequency trade-offs and stylistic differences between wavelets.",
        "",
        "CWT vs STFT - what is different",
        "-------------------------------",
        "STFT uses a fixed window -> constant df at all frequencies.",
        "CWT scales the mother wavelet -> constant Q: resolution is FINE IN TIME at",
        "high frequencies (short wavelet) and FINE IN FREQUENCY at low frequencies",
        "(long wavelet). This matches how music is structured: bass notes hold,",
        "cymbals and transients are fast. Low-frequency bass lines become much more",
        "readable than in a linear-frequency STFT.",
        "",
        "",
        "Figure 1: compare_wavelets_<file>.png  - 5 mother wavelets",
        "----------------------------------------------------------",
        "",
        "cmor1.5-1.0  (complex Morlet, bandwidth=1.5, center=1.0)",
        "  The de-facto standard for music CWT. Complex-valued -> magnitude is",
        "  phase-invariant and smooth. Good balance between time and freq resolution.",
        "  Horizontal harmonic lines appear clean; transients stay compact.",
        "",
        "morl  (real Morlet)",
        "  Same carrier as cmor but real-valued. Magnitude shows interference",
        "  stripes when harmonics beat -> looks noisier. Useful for edge/phase",
        "  studies, worse for spectrogram reading.",
        "",
        "mexh  (Mexican hat / Ricker)",
        "  Second derivative of a Gaussian. No oscillation inside the envelope,",
        "  so it is poor for resolving harmonics but excellent for detecting sharp",
        "  events (onsets, glitches). Expect blurred harmonic bands and punchy",
        "  transient columns.",
        "",
        "gaus4  (4th derivative of Gaussian, real)",
        "  A few oscillations in the envelope -> partway between mexh and morl.",
        "  Real-valued, so it also shows interference patterns. Sharper than",
        "  mexh at isolating harmonics but less clean than cmor.",
        "",
        "cgau4  (complex 4th Gaussian derivative)",
        "  Analytic (complex) counterpart of gaus4. Magnitude is smooth like cmor,",
        "  so this is a good side-by-side to morl vs cmor: same trade-off but with",
        "  Gaussian derivatives.",
        "",
        "",
        "Figure 2: compare_cmor_<file>.png  - cmor bandwidth sweep",
        "---------------------------------------------------------",
        "",
        "The complex Morlet is parameterized cmorB-C:",
        "  B = bandwidth (controls time support / frequency resolution)",
        "  C = center frequency (normalized; kept at 1.0 throughout)",
        "",
        "Effect of B (time-frequency Heisenberg trade-off):",
        "  B = 0.5  -> short wavelet in time, broad in frequency.",
        "             Transients sharp; harmonics bleed into wide bands.",
        "  B = 1.5  -> balanced (standard setting).",
        "  B = 3.0  -> longer wavelet; harmonic stacks become very tight,",
        "             transient edges soften.",
        "  B = 6.0  -> extreme frequency resolution; near pure-tone analysis,",
        "             time structure is lost (single drum hit smears across",
        "             many frames).",
        "",
        "Genre reading tips",
        "------------------",
        "jazz   : cmor3.0-1.0 reveals brass/piano harmonic stacks nicely;",
        "         cmor1.5 still shows cymbals crisply.",
        "metal  : cmor1.5 preserves kick/snare attacks needed to resolve double-bass;",
        "         mexh highlights every transient.",
        "pop    : cmor1.5 is the safest default; low B (0.5) exposes sibilance",
        "         and kick transients; high B (>=3) emphasizes sustained vocals.",
        "",
        "Bottom line",
        "-----------",
        "For the subsequent genre-preserving denoising, cmor1.5-1.0 is the",
        "recommended scalogram. If you want to emphasize tonal content, use",
        "cmor3.0-1.0; for onset/transient emphasis, cmor0.5-1.0 or mexh.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.name}")


def main():
    sr_seen = None
    for fname in FILES:
        path = IN_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        x, sr, t0 = load_segment(path)
        sr_seen = sr
        print(f"\n== {path.stem}  (sr={sr}, {len(x)} samples = {len(x)/sr:.2f} s from {t0:.1f}s)")

        # Figure 1: wavelets comparison
        panels = []
        for w in WAVELETS:
            f, db, dt = cwt_scalogram_db(x, sr, w)
            print(f"  {w:<15s} cwt took {dt:.1f}s, shape={db.shape}")
            panels.append((w, f, db, dt))
        plot_panels(path.stem, sr, t0, panels,
                    "CWT scalograms - mother wavelet comparison",
                    OUT_DIR / f"compare_wavelets_{path.stem}.png")
        print(f"  wrote compare_wavelets_{path.stem}.png")

        # Figure 2: cmor bandwidth sweep
        panels2 = []
        for w in CMOR_VARIANTS:
            f, db, dt = cwt_scalogram_db(x, sr, w)
            print(f"  {w:<15s} cwt took {dt:.1f}s, shape={db.shape}")
            panels2.append((w, f, db, dt))
        plot_panels(path.stem, sr, t0, panels2,
                    "CWT scalograms - complex Morlet bandwidth sweep",
                    OUT_DIR / f"compare_cmor_{path.stem}.png")
        print(f"  wrote compare_cmor_{path.stem}.png")

    if sr_seen is not None:
        write_notes(sr_seen)


if __name__ == "__main__":
    main()
