"""Plot STFT of 9 noisy WAVs (3 per genre) into Out/."""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib.pyplot as plt

IN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENRES = ["jazz", "metal", "pop"]
IDS = ["01", "02", "03"]
FILES = [f"{g}_{i}.wav" for g in GENRES for i in IDS]

NPERSEG = 2048
NOVERLAP = 1536
DB_FLOOR = -100.0


def load_mono(path: Path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def compute_stft_db(x, sr):
    f, t, Z = stft(x, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP, window="hann")
    mag = np.abs(Z)
    ref = mag.max() if mag.max() > 0 else 1.0
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / ref)
    return f, t, np.maximum(db, DB_FLOOR)


def plot_one(name, f, t, db, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.pcolormesh(t, f, db, shading="gouraud", cmap="magma", vmin=DB_FLOOR, vmax=0)
    ax.set_title(f"STFT (dB) — {name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax, label="dB (rel. max)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_grid(results, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=True)
    for ax, (name, f, t, db) in zip(axes.flat, results):
        im = ax.pcolormesh(t, f, db, shading="gouraud", cmap="magma", vmin=DB_FLOOR, vmax=0)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("f [Hz]")
    fig.suptitle("STFT (dB) — 3 files per genre (noisy)", fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="dB (rel. max)", shrink=0.8)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    results = []
    for fname in FILES:
        path = IN_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        x, sr = load_mono(path)
        f, t, db = compute_stft_db(x, sr)
        out_png = OUT_DIR / f"stft_{path.stem}.png"
        plot_one(path.stem, f, t, db, out_png)
        print(f"wrote {out_png.name}  (sr={sr}, dur={len(x)/sr:.2f}s, shape={db.shape})")
        results.append((path.stem, f, t, db))
    if results:
        grid_path = OUT_DIR / "stft_grid_3x3.png"
        plot_grid(results, grid_path)
        print(f"wrote {grid_path.name}")


if __name__ == "__main__":
    main()
