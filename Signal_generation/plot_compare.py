"""Comparison plots across ECG / Vibration / Music.

Produces two figures:
  * compare_clean_noisy_noise.png  -- 3x3 time-domain (clean, noisy, noise)
  * compare_spectrograms.png       -- 3x2 spectrograms (clean vs noisy)
"""
import os
import numpy as np
import scipy.signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signal_contamination import (
    load_ecg, load_vibration, load_music, make_noise, OUT, SNR_TARGET_DB,
)

loaders = [
    ("ECG",       load_ecg,       "mV"),
    ("Vibration", load_vibration, "g"),
    ("Music",     load_music,     "amplitude"),
]

prepared = []
for name, fn, units in loaders:
    clean, fs, source, _units = fn()
    _, _, total = make_noise(clean, fs)
    noisy = clean + total
    prepared.append((name, units, fs, clean, noisy, total))

# ---- Figure 1: time-domain 3x3 compare -------------------------------- #
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
col_titles = ["Clean", "Noisy (clean + total noise)", "Noise only"]
for row, (name, units, fs, clean, noisy, total) in enumerate(prepared):
    t = np.arange(clean.size) / fs
    series = [clean, noisy, total]
    for col, y in enumerate(series):
        ax = axes[row, col]
        ax.plot(t, y, lw=0.5)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=12)
        if col == 0:
            ax.set_ylabel(f"{name}\n[{units}]", fontsize=11)
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
    ymax = max(np.max(np.abs(s)) for s in series) * 1.05
    for col in range(3):
        axes[row, col].set_ylim(-ymax, ymax)
fig.suptitle(
    f"ECG · Vibration · Music — clean vs noisy vs noise only "
    f"(target SNR = {SNR_TARGET_DB:+.1f} dB)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.join(OUT, "compare_clean_noisy_noise.png")
fig.savefig(out1, dpi=120)
plt.close(fig)
print("Saved ->", out1)

# ---- Figure 2: spectrograms 3x2 (clean vs noisy) ---------------------- #
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for row, (name, units, fs, clean, noisy, total) in enumerate(prepared):
    nperseg = max(64, min(1024, clean.size // 32))
    f_c, t_c, Sxx_c = sps.spectrogram(clean, fs=fs, nperseg=nperseg)
    f_n, t_n, Sxx_n = sps.spectrogram(noisy, fs=fs, nperseg=nperseg)
    Sxx_c_db = 10 * np.log10(Sxx_c + 1e-12)
    Sxx_n_db = 10 * np.log10(Sxx_n + 1e-12)
    vmin = min(Sxx_c_db.min(), Sxx_n_db.min())
    vmax = max(Sxx_c_db.max(), Sxx_n_db.max())
    for col, (data, ttl_suffix) in enumerate([(Sxx_c_db, "clean"),
                                              (Sxx_n_db, "noisy")]):
        ax = axes[row, col]
        t_axis = t_c if col == 0 else t_n
        f_axis = f_c if col == 0 else f_n
        pc = ax.pcolormesh(t_axis, f_axis, data, shading="auto",
                           vmin=vmin, vmax=vmax)
        if row == 0:
            ax.set_title(f"Spectrogram — {ttl_suffix} [dB]", fontsize=12)
        if col == 0:
            ax.set_ylabel(f"{name}\nfreq [Hz]", fontsize=11)
        ax.set_xlabel("time [s]")
    fig.colorbar(pc, ax=axes[row, :], label="dB", shrink=0.85)

fig.suptitle(
    f"ECG · Vibration · Music — clean vs noisy spectrograms "
    f"(target SNR = {SNR_TARGET_DB:+.1f} dB)", fontsize=14)
out2 = os.path.join(OUT, "compare_spectrograms.png")
fig.savefig(out2, dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved ->", out2)
