"""
Standalone STFT figure for the DSS 2026 talk.

Renders only the two spectrogram panels (clean vs noisy) from the music
case in Signal_generation/signal_contamination.py, using identical STFT
parameters and the identical (seed=42) noise realisation -- so this
figure is a 1:1 crop of the bottom row of out/music_plots.png.

Output: presentation_stft.png (next to this script).
"""

import os
import sys

import numpy as np
import scipy.signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIGGEN = os.path.join(HERE, "Signal_generation")
sys.path.insert(0, SIGGEN)

# Re-use the EXACT loader + noise model used to produce music_plots.png
from signal_contamination import load_music, make_noise  # noqa: E402


def main():
    clean, fs, source, _units = load_music()
    _emc, _meas, total_noise = make_noise(clean, fs)
    noisy = clean + total_noise

    # Same ADC clip the original pipeline applies before plotting.
    full_scale_adc = 3.0 * (np.max(np.abs(clean)) + 1e-12)
    noisy = np.clip(noisy, -full_scale_adc, full_scale_adc)

    # Spectrogram parameters: identical to signal_contamination.py:204-211
    sp_nperseg = max(64, min(1024, clean.size // 32))
    f_c, t_c, Sxx_c = sps.spectrogram(clean, fs=fs, nperseg=sp_nperseg)
    f_n, t_n, Sxx_n = sps.spectrogram(noisy, fs=fs, nperseg=sp_nperseg)

    Sc_db = 10 * np.log10(Sxx_c + 1e-12)
    Sn_db = 10 * np.log10(Sxx_n + 1e-12)
    vmin = float(min(Sc_db.min(), Sn_db.min()))
    vmax = float(max(Sc_db.max(), Sn_db.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    pcm = axes[0].pcolormesh(t_c, f_c, Sc_db, shading="auto",
                             vmin=vmin, vmax=vmax)
    axes[0].set_title("Clean-signal spectrogram [dB]")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("frequency [Hz]")

    axes[1].pcolormesh(t_n, f_n, Sn_db, shading="auto",
                       vmin=vmin, vmax=vmax)
    axes[1].set_title("Noisy-signal spectrogram [dB]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("frequency [Hz]")

    fig.colorbar(pcm, ax=axes, location="right", shrink=0.85, label="dB")
    fig.suptitle(f"{source} — STFT (nperseg={sp_nperseg}, fs={fs} Hz)",
                 fontsize=11)

    out_path = os.path.join(HERE, "presentation_stft.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
