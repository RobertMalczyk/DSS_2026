"""
make_music_contamination_figure.py

Horizontal slide-page version of
Signal_generation/out/music_plots.png.

Trimmed to six panels in a 3 x 2 grid (no metadata text panel):

  Row 1 (time-domain): Clean signal | Total noise | Noisy signal
  Row 2 (analysis):    Welch PSD    | Clean spec  | Noisy spec

Output: presentation_music_contamination.png at the project root.
"""

import os
import sys

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIGGEN = os.path.join(HERE, "Signal_generation")
sys.path.insert(0, SIGGEN)

from signal_contamination import (        # noqa: E402
    load_music, make_noise, db_power, SNR_TARGET_DB,
)


def main():
    clean, fs, source, units = load_music()
    _emc, _meas, total_noise = make_noise(clean, fs)
    noisy_signal = clean + total_noise

    # ADC saturation, identical to the production pipeline
    full_scale_adc = 3.0 * (np.max(np.abs(clean)) + 1e-12)
    noisy_signal   = np.clip(noisy_signal, -full_scale_adc, full_scale_adc)

    snr_db = db_power(clean) - db_power(total_noise)

    # ---------- 3 x 2 grid ----------
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(
        nrows=2, ncols=3,
        hspace=0.42, wspace=0.25,
        top=0.91, bottom=0.07, left=0.05, right=0.985,
    )

    ax_clean  = fig.add_subplot(gs[0, 0])
    ax_total  = fig.add_subplot(gs[0, 1])
    ax_noisy  = fig.add_subplot(gs[0, 2])
    ax_psd    = fig.add_subplot(gs[1, 0])
    ax_specC  = fig.add_subplot(gs[1, 1])
    ax_specN  = fig.add_subplot(gs[1, 2])

    # ---------- top row: time-domain panels ----------
    t = np.arange(clean.size) / fs
    panels = [
        (ax_clean, "Clean signal",                 clean,        "#1f77b4"),
        (ax_total, "Total noise",                  total_noise,  "#c0504d"),
        (ax_noisy, "Noisy signal (post ADC clip)", noisy_signal, "#0d1f3c"),
    ]
    for ax, title, y, color in panels:
        ax.plot(t, y, lw=0.6, color=color)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("time [s]", fontsize=10)
        ax.set_ylabel(units, fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)

    # share y-limits between Total noise and Noisy so amplitudes compare honestly
    noise_ymax = max(np.max(np.abs(total_noise)),
                     np.max(np.abs(noisy_signal)))
    for ax in (ax_total, ax_noisy):
        ax.set_ylim(-noise_ymax * 1.05, noise_ymax * 1.05)

    # ---------- Welch PSD ----------
    nperseg = max(64, min(2048, clean.size // 4))
    f1, P1 = sps.welch(clean,        fs=fs, nperseg=nperseg)
    f2, P2 = sps.welch(noisy_signal, fs=fs, nperseg=nperseg)
    ax_psd.semilogy(f1, P1, label="clean", lw=1.2, color="#1f77b4")
    ax_psd.semilogy(f2, P2, label="noisy", lw=1.2, color="#c0504d", alpha=0.85)
    ax_psd.set_title("Welch PSD  -  clean vs noisy",
                     fontsize=13, fontweight="bold")
    ax_psd.set_xlabel("frequency [Hz]", fontsize=10)
    ax_psd.set_ylabel("PSD",            fontsize=10)
    ax_psd.legend(fontsize=10, loc="upper right")
    ax_psd.grid(alpha=0.3, which="both")
    ax_psd.tick_params(labelsize=9)

    # ---------- spectrograms ----------
    sp_nperseg = max(64, min(1024, clean.size // 32))
    f_c, t_c, Sxx_c = sps.spectrogram(clean,        fs=fs, nperseg=sp_nperseg)
    f_n, t_n, Sxx_n = sps.spectrogram(noisy_signal, fs=fs, nperseg=sp_nperseg)
    Sc_db = 10 * np.log10(Sxx_c + 1e-12)
    Sn_db = 10 * np.log10(Sxx_n + 1e-12)
    vmin = float(min(Sc_db.min(), Sn_db.min()))
    vmax = float(max(Sc_db.max(), Sn_db.max()))

    pcm = None
    for ax, ttl, t_, f_, S_ in [
        (ax_specC, "Clean-signal spectrogram [dB]",   t_c, f_c, Sc_db),
        (ax_specN, "Noisy-signal spectrogram [dB]",   t_n, f_n, Sn_db),
    ]:
        pcm = ax.pcolormesh(t_, f_, S_, shading="auto",
                            vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(ttl, fontsize=13, fontweight="bold")
        ax.set_xlabel("time [s]", fontsize=10)
        ax.set_ylabel("frequency [Hz]", fontsize=10)
        ax.tick_params(labelsize=9)

    # shared colorbar for the two spectrograms
    cb_ax = fig.add_axes([0.992, 0.09, 0.008, 0.36])
    cbar = fig.colorbar(pcm, cax=cb_ax)
    cbar.set_label("[dB]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ---------- title ----------
    fig.suptitle(
        f"Music  -  clean vs EMC + measurement contamination "
        f"(target SNR = {SNR_TARGET_DB:+.1f} dB, achieved {snr_db:+.2f} dB)",
        fontsize=16, fontweight="bold", y=0.965,
    )

    out_path = os.path.join(HERE, "presentation_music_contamination.png")
    fig.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
