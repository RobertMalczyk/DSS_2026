"""Compare STFT with different window sizes (nperseg) on one file per genre.

Keeps a constant 75% overlap ratio and Hann window; only nperseg varies.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib.pyplot as plt

IN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\STFT_WINDOW_COMPARISON")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["jazz_01.wav", "metal_01.wav", "pop_01.wav"]

# Significantly different window sizes, default=2048 is included.
NPERSEGS = [256, 1024, 2048, 4096, 8192, 16384]
OVERLAP_RATIO = 0.75
DB_FLOOR = -100.0


def load_mono(path: Path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def compute_stft_db(x, sr, nperseg):
    noverlap = int(nperseg * OVERLAP_RATIO)
    f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
    mag = np.abs(Z)
    ref = mag.max() if mag.max() > 0 else 1.0
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
    return f, t, np.maximum(db, DB_FLOOR)


def plot_file_comparison(name, sr, panels, out_path):
    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows))
    axes = np.array(axes).reshape(-1)
    im = None
    for ax, (nperseg, f, t, db) in zip(axes, panels):
        im = ax.pcolormesh(t, f, db, shading="gouraud", cmap="magma", vmin=DB_FLOOR, vmax=0)
        dt_ms = (nperseg * (1 - OVERLAP_RATIO)) / sr * 1000.0
        win_ms = nperseg / sr * 1000.0
        df_hz = sr / nperseg
        ax.set_title(
            f"nperseg={nperseg}  win={win_ms:.1f} ms  df={df_hz:.2f} Hz  hop={dt_ms:.1f} ms",
            fontsize=10,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"STFT window-size comparison — {name}  (Hann, {int(OVERLAP_RATIO*100)}% overlap)",
                 fontsize=13)
    fig.colorbar(im, ax=axes.tolist(), label="dB (rel. max)", shrink=0.8)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_notes(sr):
    notes_path = OUT_DIR / "README.txt"
    lines = [
        "STFT window-size comparison",
        "===========================",
        "",
        f"Source files: jazz_01.wav, metal_01.wav, pop_01.wav (sr = {sr} Hz, 30 s, mono).",
        "Fixed parameters: Hann window, 75% overlap ratio, magnitude in dB rel. per-plot max, floor -100 dB.",
        "Variable: nperseg (window length in samples).",
        "",
        "Settings tested (at 22050 Hz):",
    ]
    for n in NPERSEGS:
        win_ms = n / sr * 1000.0
        df = sr / n
        hop_ms = n * (1 - OVERLAP_RATIO) / sr * 1000.0
        tag = " (DEFAULT in plot_stft.py)" if n == 2048 else ""
        lines.append(
            f"  nperseg = {n:>5}  ->  window = {win_ms:6.1f} ms  |  df = {df:7.2f} Hz/bin  |  hop = {hop_ms:5.1f} ms{tag}"
        )
    lines += [
        "",
        "How to read the results",
        "-----------------------",
        "STFT imposes a time-frequency trade-off (Heisenberg-Gabor): df * dt = constant.",
        "Doubling the window halves df (finer frequency) and doubles the frame length",
        "(coarser time localization of transients).",
        "",
        "nperseg = 256  (~11.6 ms, df ~86 Hz)",
        "  + Sharp percussion attacks (drum hits, plucks) appear as thin vertical lines.",
        "  + Best time resolution: good for onset detection.",
        "  - Bass lines and low harmonics are smeared together (df > a semitone below ~1.5 kHz).",
        "  - Very poor frequency separation in the low end.",
        "",
        "nperseg = 1024 (~46 ms, df ~22 Hz)",
        "  + Better frequency detail: melodic lines start becoming visible as horizontal bands.",
        "  + Still catches most percussive transients.",
        "  - Low-frequency harmonics (< 100 Hz) still blurred.",
        "",
        "nperseg = 2048 (~93 ms, df ~11 Hz)  [DEFAULT, librosa's default]",
        "  + Balanced: harmonics and transients both reasonably visible.",
        "  + This is the MIR canonical setting for 22 kHz audio.",
        "  * Recommended starting point for general analysis and MFCC/mel-spec features.",
        "",
        "nperseg = 4096 (~186 ms, df ~5.4 Hz)",
        "  + Harmonic stacks (vocals, sustained notes) become very sharp.",
        "  + Good for pitch analysis, tonal content, sustained instruments.",
        "  - Transient smearing: kick/snare become horizontal streaks, losing temporal precision.",
        "",
        "nperseg = 8192 (~372 ms, df ~2.7 Hz)",
        "  + Semitone-level frequency resolution (1 semitone ~ 6% of freq -> < 3 Hz only below ~45 Hz).",
        "  + Very clean harmonic series visible for sustained tones.",
        "  - Poor time localization: a 93 ms drum transient is spread over 3-4 frames.",
        "",
        "nperseg = 16384 (~743 ms, df ~1.35 Hz)",
        "  + Extreme frequency resolution; useful for analyzing stationary tones.",
        "  - Destroys time structure: groove/rhythm unreadable.",
        "  - Overkill for genre features; only useful for long sustained content.",
        "",
        "Genre-specific implications",
        "---------------------------",
        "jazz   : sustained brass/piano harmonics benefit from larger windows (2048-4096);",
        "         ride-cymbal texture needs smaller (~1024) to stay crisp.",
        "metal  : fast double-kick and palm-muted chugs demand smaller windows (1024-2048)",
        "         to keep transients resolved. Distortion harmonics are broadband anyway.",
        "pop    : vocal pitch and kick transients both matter -> default 2048 is the safest.",
        "",
        "Recommendation",
        "--------------",
        "Keep nperseg = 2048 as the default for subsequent denoising/filtering work.",
        "If the next step is pitch-oriented (tonal separation), bump to 4096.",
        "If transient preservation matters (e.g. transient-protecting noise gate), drop to 1024.",
    ]
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {notes_path.name}")


def main():
    sr_seen = None
    for fname in FILES:
        path = IN_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        x, sr = load_mono(path)
        sr_seen = sr
        panels = []
        for nperseg in NPERSEGS:
            f, t, db = compute_stft_db(x, sr, nperseg)
            panels.append((nperseg, f, t, db))
        out_png = OUT_DIR / f"compare_{path.stem}.png"
        plot_file_comparison(path.stem, sr, panels, out_png)
        print(f"wrote {out_png.name}")
    if sr_seen is not None:
        write_notes(sr_seen)


if __name__ == "__main__":
    main()
