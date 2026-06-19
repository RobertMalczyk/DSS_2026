"""
make_denoising_effect_figure.py

Generates presentation_denoising_effect.png for the DSS 2026 talk
(slide 16: effect of denoising).

Layout (16:10 canvas):

  Row 1 -- three side-by-side log-magnitude spectrograms on the same
           dB scale: clean / noisy / Stage-7-filtered, of the same
           music track over the same time window.

  Row 2 -- PSD overlay (Welch) of all three, with the peaks Stage 7
           discovered from the noisy audio annotated:
             50 / 100 / 150 / 200 Hz  -- mains comb
             ~4410 Hz                 -- switching tone

Audio sources (read from disk -- the noisy / filtered / clean WAVs
share the same int16 gain normalisation written by build_music_dataset
so amplitude comparisons between them are internally consistent):

  Signal_generation/out/Model/clean/<track>.wav
  Signal_generation/out/Model/noisy/<track>.wav
  Signal_generation/out/Model/Filtered ComponentMatched 7th stage/
      filtered_componentmatched_7th_stage_<track>.wav

Per the COMPONENT_MATCHED_v1 README the 9-track starter set carries
a mean +8.50 dB SNR gain (noisy -6.25 dB -> filtered +2.24 dB).
"""
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- track + window selection -----------------------------------
TRACK = "jazz_01"
T_START_S = 0.0
T_DUR_S   = 8.0      # ~8 s window keeps the spectrogram readable

ROOT = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model")
CLEAN_WAV    = ROOT / "clean" / f"{TRACK}.wav"
NOISY_WAV    = ROOT / "noisy" / f"{TRACK}.wav"
FILTERED_WAV = (ROOT / "Filtered ComponentMatched 7th stage"
                / f"filtered_componentmatched_7th_stage_{TRACK}.wav")

clean,    fs   = sf.read(CLEAN_WAV)
noisy,    fs_n = sf.read(NOISY_WAV)
filtered, fs_f = sf.read(FILTERED_WAV)
assert fs == fs_n == fs_f, f"sample-rate mismatch: {fs}, {fs_n}, {fs_f}"

# Trim to the same window
i0 = int(T_START_S * fs)
i1 = i0 + int(T_DUR_S * fs)
clean    = clean[i0:i1]
noisy    = noisy[i0:i1]
filtered = filtered[i0:i1]

# ---- spectrograms (shared dB scale) -----------------------------
NPERSEG  = 1024
NOVERLAP = 768

def spec_db(x):
    f, t, S = sps.spectrogram(x, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP,
                              window="hann", scaling="density")
    return f, t, 10.0 * np.log10(S + 1e-12)

f_c, t_c, S_c = spec_db(clean)
f_n, t_n, S_n = spec_db(noisy)
f_f, t_f, S_f = spec_db(filtered)

# Use noisy / filtered range as the dB scale (clean has a much larger
# dynamic range and would crush the others if we used a global vmin).
vmax = float(max(S_n.max(), S_f.max(), S_c.max()))
vmin = vmax - 70.0

# ---- Welch PSD overlay -----------------------------------------
def welch(x):
    f, P = sps.welch(x, fs=fs, nperseg=4096, noverlap=2048, window="hann")
    return f, 10.0 * np.log10(P + 1e-18)

fw_c, Pw_c = welch(clean)
fw_n, Pw_n = welch(noisy)
fw_f, Pw_f = welch(filtered)

# Discovered peaks (re-stated from analyze_noise_pattern -- these are
# the components Stage 7 notched, all derived from the noisy data only)
MAINS_HZ      = [51.1, 99.6, 150.7, 199.2]
SWITCHING_HZ  = 4410.0   # mean of the two FM sidebands ~4403/4417

# ---- figure -----------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"

NAVY    = "#0d1f3c"
NAVY_BAR= "#1f497d"
ACCENT  = "#c0504d"
GREY    = "#6b7280"
GREY_D  = "#374151"

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(
    nrows=2, ncols=3,
    height_ratios=[1.0, 1.0],
    hspace=0.40, wspace=0.10,
    left=0.06, right=0.93, top=0.89, bottom=0.07,
)

ax_clean = fig.add_subplot(gs[0, 0])
ax_noisy = fig.add_subplot(gs[0, 1], sharey=ax_clean, sharex=ax_clean)
ax_filt  = fig.add_subplot(gs[0, 2], sharey=ax_clean, sharex=ax_clean)
ax_psd   = fig.add_subplot(gs[1, :])

# --- spectrograms ---
def draw_spec(ax, f, t, S, title, badge_text=None, badge_color=NAVY_BAR):
    pcm = ax.pcolormesh(t, f, S, shading="auto", vmin=vmin, vmax=vmax,
                        cmap="magma")
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=6)
    ax.set_xlabel("time [s]", fontsize=11)
    ax.set_ylim(0, fs/2)
    if badge_text is not None:
        ax.text(0.97, 0.96, badge_text,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_color,
                          edgecolor="none", alpha=0.85))
    return pcm

pcm_c = draw_spec(ax_clean, f_c, t_c, S_c, "Clean (reference)",
                  badge_text="reference",
                  badge_color="#2e7d32")
ax_clean.set_ylabel("frequency [Hz]", fontsize=11)

pcm_n = draw_spec(ax_noisy, f_n, t_n, S_n, "Noisy  (SNR = -10 dB target)",
                  badge_text="before",
                  badge_color=ACCENT)
pcm_f = draw_spec(ax_filt, f_f, t_f, S_f,
                  "Filtered  (Stage 7 ComponentMatched)",
                  badge_text="after  +8.5 dB",
                  badge_color=NAVY_BAR)
plt.setp(ax_noisy.get_yticklabels(), visible=False)
plt.setp(ax_filt.get_yticklabels(), visible=False)

# Annotate the noisy panel with brackets pointing at structures that
# Stage 7 will remove
def annotate_arrow(ax, x_data, y_data, label, x_text=None, y_text=None,
                   color="white"):
    if x_text is None: x_text = x_data + 1.2
    if y_text is None: y_text = y_data
    ax.annotate(label, xy=(x_data, y_data), xytext=(x_text, y_text),
                fontsize=10, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

# Mains-comb annotation on noisy
annotate_arrow(ax_noisy, t_n[len(t_n)//6], 130,
               "mains comb 50/100/150/200 Hz",
               x_text=t_n[len(t_n)//6] + 0.4, y_text=1500,
               color="white")
# Switching-tone annotation on noisy
annotate_arrow(ax_noisy, t_n[len(t_n)//2], 4410,
               "switching tone ~4.4 kHz",
               x_text=t_n[len(t_n)//2] - 1.4, y_text=6300,
               color="white")
# Broadband EMI annotation on noisy
annotate_arrow(ax_noisy, t_n[int(len(t_n)*0.85)], 8500,
               "broadband EMI",
               x_text=t_n[int(len(t_n)*0.85)] - 2.6, y_text=9700,
               color="white")

# Shared colorbar for the three spectrograms (aligned with row 1)
cb_ax = fig.add_axes([0.945, 0.535, 0.011, 0.355])
cbar = fig.colorbar(pcm_n, cax=cb_ax)
cbar.set_label("[dB]", fontsize=10, color=NAVY)
cbar.ax.tick_params(labelsize=9)

# --- PSD overlay ---
# Shaded bands for the discovered noise regions (sit BEHIND the curves
# so they don't obscure the data)
ax_psd.axvspan(45, 220, color=ACCENT, alpha=0.07, zorder=0)
ax_psd.axvspan(4200, 4620, color=ACCENT, alpha=0.07, zorder=0)

ax_psd.semilogx(fw_c, Pw_c, color="#2e7d32", lw=1.4, label="clean",
                alpha=0.85, zorder=2)
ax_psd.semilogx(fw_n, Pw_n, color=ACCENT,    lw=1.4, label="noisy",
                alpha=0.95, zorder=3)
ax_psd.semilogx(fw_f, Pw_f, color=NAVY_BAR,  lw=1.7,
                label="filtered (Stage 7)", alpha=0.95, zorder=4)

# Vertical markers at exactly the detected frequencies
for fz in MAINS_HZ:
    ax_psd.axvline(fz, color=ACCENT, linestyle=":", alpha=0.55, lw=1.0,
                   zorder=1)
ax_psd.axvline(SWITCHING_HZ, color=ACCENT, linestyle=":", alpha=0.55,
               lw=1.0, zorder=1)

ax_psd.set_xlim(20, fs/2)
y_lo = float(np.percentile(Pw_c, 5)) - 3
y_hi = float(max(Pw_n.max(), Pw_c.max())) + 8
ax_psd.set_ylim(y_lo, y_hi)
ax_psd.set_xlabel("frequency [Hz]", fontsize=11)
ax_psd.set_ylabel("PSD [dB/Hz]", fontsize=11)
ax_psd.grid(True, which="both", alpha=0.3, linewidth=0.5)
ax_psd.legend(loc="lower left", fontsize=10, framealpha=0.95,
              ncol=3, columnspacing=1.5)

# Top-of-axes labels for the shaded bands
y_lab = y_hi - 2.5
ax_psd.text(np.sqrt(MAINS_HZ[0]*MAINS_HZ[-1]), y_lab,
            "mains comb (detected)",
            ha="center", va="top", fontsize=10.5, color=ACCENT,
            fontweight="bold")
ax_psd.text(SWITCHING_HZ, y_lab,
            "switching tone (detected)",
            ha="center", va="top", fontsize=10.5, color=ACCENT,
            fontweight="bold")

ax_psd.set_title("Power spectral density: noise components discovered "
                 "from the noisy audio alone, then notched by Stage 7",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=8)

# --- super title and subtitle ---
fig.text(0.5, 0.965,
         "Effect of denoising  --  Stage 7 ComponentMatched filter",
         ha="center", va="center",
         fontsize=20, fontweight="bold", color=NAVY)
fig.text(0.5, 0.932,
         f"track: {TRACK.replace('_', ' ')}   .   "
         f"{T_DUR_S:.0f}-second window   .   "
         "average gain on the 9-track starter set: +8.5 dB SNR",
         ha="center", va="center",
         fontsize=12, color=GREY, style="italic")

OUT = Path(r"C:\Robak\DSS2026\Claude\presentation_denoising_effect.png")
fig.savefig(OUT, dpi=160, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}")
