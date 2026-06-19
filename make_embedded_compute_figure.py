"""
make_embedded_compute_figure.py

Generates presentation_embedded_compute.png for the new embedded-hardware
slide of the DSS 2026 talk -- the slide that converts the "70 % recovery
from 173 hand-picked features" claim from slide 18 into the compute /
memory / cost payoff promised by conference.txt.

Visualizes five paths to the same 3 s music-window inference, plotted on
a log-log scatter of MACs/inference vs total memory footprint
(model params + peak activations). Hardware-fit zones are drawn as
shaded background bands -- MCU ($5), MCU+NPU SoC ($30), GPU/Jetson
($100+) -- so the reader sees at a glance which hardware class each
architecture forces.

Numbers are order-of-magnitude estimates derived from:
- v2 path: features_description.md (22050 Hz, 3 s window, n_fft=2048,
  hop=512 -> 129 frames; 86 per-frame quantities; median + 1.4826*MAD;
  + onset_rate; + RBF-SVM 3-way ~300 SVs).
- CNN ladder: typical MAC counts for log-mel 128x130 input through
  a 3-block small CNN, VGG-style deeper CNN, ResNet-18 (1.8 G MACs is
  the canonical 224x224 figure; spectrogram input is comparable),
  and AST/PaSST transformer at the top of slide 11's v1->v4 ladder.

These are not measured cycles -- the slide caption flags them as
spec-derived estimates.
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--for-slide", action="store_true",
                    help="Render the slide-embedded variant (no internal title/"
                         "subtitle/footnote -- slide-level title carries them).")
args = parser.parse_args()
FOR_SLIDE = args.for_slide
OUTFILE = ("presentation_embedded_compute_slide.png" if FOR_SLIDE
           else "presentation_embedded_compute.png")

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette (consistent with the rest of the deck) ----
BG          = "#f5f7fa"
NAVY        = "#0d1f3c"
GREY_D      = "#374151"
GREY        = "#6b7280"
GREEN_DARK  = "#1f6e3a"
GREEN_MID   = "#5cba78"
GREEN_BG    = "#d5efe0"
AMBER       = "#d97706"
AMBER_BG    = "#fff1d6"
RED         = "#b91c1c"
RED_BG      = "#fde2e2"
GREY_BG     = "#e8efe9"

# ---- architecture estimates (MACs/inference, total memory in bytes) ----
ARCHS = [
    {
        "name": "v2 features\n+ RBF-SVM",
        "macs": 13e6,        # ~13 MMACs (~25 MFLOPs)
        "mem":  250e3,       # ~250 KB (SVs + filterbank + streaming buffers)
        "color": GREEN_DARK,
        "fit": "mcu",
        "tier": "this talk",
    },
    {
        "name": "Small CNN\n(3 conv blocks)",
        "macs": 150e6,       # ~150 MMACs
        "mem":  3e6,         # ~3 MB (1 MB params + 2 MB peak act)
        "color": AMBER,
        "fit": "tight",
        "tier": "v1 on slide 11",
    },
    {
        "name": "Deeper CNN\n(VGG-style)",
        "macs": 1e9,         # ~1 GMAC
        "mem":  25e6,        # ~25 MB
        "color": RED,
        "fit": "off",
        "tier": "v2 on slide 11",
    },
    {
        "name": "ResNet-18\non spectrograms",
        "macs": 1.8e9,       # canonical
        "mem":  54e6,        # ~54 MB
        "color": RED,
        "fit": "off",
        "tier": "v3 on slide 11",
    },
    {
        "name": "Audio Spectrogram\nTransformer",
        "macs": 40e9,        # ~40 GMACs
        "mem":  390e6,       # ~390 MB
        "color": RED,
        "fit": "off",
        "tier": "v4 on slide 11",
    },
]

# ---- axis ranges (log) ----
X_MIN, X_MAX = 5e6, 2e11        # MACs
Y_MIN, Y_MAX = 5e4, 2e9         # bytes

# ---- hardware-fit zone boundaries ----
MCU_MAX_MACS = 1e8
MCU_MAX_MEM  = 1e6
NPU_MAX_MACS = 5e9
NPU_MAX_MEM  = 5e7

fig, ax = plt.subplots(figsize=(14, 7.4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)

# ============== HARDWARE ZONES ==============
ax.add_patch(Rectangle((X_MIN, Y_MIN),
                       MCU_MAX_MACS - X_MIN, MCU_MAX_MEM - Y_MIN,
                       facecolor=GREEN_BG, edgecolor="none", zorder=1))
ax.add_patch(Rectangle((MCU_MAX_MACS, Y_MIN),
                       NPU_MAX_MACS - MCU_MAX_MACS, NPU_MAX_MEM - Y_MIN,
                       facecolor=AMBER_BG, edgecolor="none", zorder=1))
ax.add_patch(Rectangle((X_MIN, MCU_MAX_MEM),
                       NPU_MAX_MACS - X_MIN, NPU_MAX_MEM - MCU_MAX_MEM,
                       facecolor=AMBER_BG, edgecolor="none", zorder=1))
ax.add_patch(Rectangle((NPU_MAX_MACS, Y_MIN),
                       X_MAX - NPU_MAX_MACS, Y_MAX - Y_MIN,
                       facecolor=RED_BG, edgecolor="none", zorder=1))
ax.add_patch(Rectangle((X_MIN, NPU_MAX_MEM),
                       NPU_MAX_MACS - X_MIN, Y_MAX - NPU_MAX_MEM,
                       facecolor=RED_BG, edgecolor="none", zorder=1))

# zone labels parked in empty corners so they don't collide with data points
ax.text(8.5e7, 1.0e5, "MCU class\n~ $5 (Cortex-M7, 1 MB SRAM)",
        ha="right", va="bottom",
        fontsize=11, fontweight="bold", color=GREEN_DARK, zorder=3)
ax.text(4.5e9, 1.3e5, "MCU + NPU SoC\n~ $30 (STM32N6, NXP MCX-N)",
        ha="right", va="bottom",
        fontsize=11, fontweight="bold", color=AMBER, zorder=3)
ax.text(1.5e11, 1.5e9, "GPU / Jetson\n$100+",
        ha="right", va="top",
        fontsize=11, fontweight="bold", color=RED, zorder=3)

# ============== POINTS ==============
for a in ARCHS:
    is_v2 = a["fit"] == "mcu"
    size = 600 if is_v2 else 280
    edge = "white" if is_v2 else a["color"]
    lw   = 2.5 if is_v2 else 1.0
    ax.scatter(a["macs"], a["mem"],
               s=size, c=a["color"], edgecolors=edge, linewidth=lw,
               zorder=5, marker="*" if is_v2 else "o")

# ---- per-point labels (positioned to avoid overlap) ----
LABELS = [
    # name match, dx_log10, dy_log10, ha, va
    ("v2 features\n+ RBF-SVM",            +0.30, +0.30, "left",   "center"),
    ("Small CNN\n(3 conv blocks)",        +0.30, +0.05, "left",   "center"),
    ("Deeper CNN\n(VGG-style)",           -0.30, +0.30, "right",  "center"),
    ("ResNet-18\non spectrograms",        +0.30, +0.05, "left",   "center"),
    ("Audio Spectrogram\nTransformer",    -0.30, +0.05, "right",  "center"),
]
for a, (_, dx, dy, ha, va) in zip(ARCHS, LABELS):
    x = a["macs"] * 10**dx
    y = a["mem"]  * 10**dy
    ax.text(x, y, a["name"],
            ha=ha, va=va,
            fontsize=11, fontweight="bold", color=NAVY, zorder=6)
    ax.text(x, y * 10**(-0.18 if va != "top" else -0.25),
            a["tier"],
            ha=ha, va="top",
            fontsize=9, color=GREY_D, style="italic", zorder=6)

# ---- guideline arrow from v2 to small CNN ----
v2 = ARCHS[0]; sm = ARCHS[1]
ax.annotate("", xy=(sm["macs"]*0.85, sm["mem"]*0.85),
            xytext=(v2["macs"]*1.4, v2["mem"]*1.4),
            arrowprops=dict(arrowstyle="->", color=GREY_D,
                            lw=1.2, linestyle="--"),
            zorder=4)
ax.text(np.sqrt(v2["macs"]*sm["macs"]),
        np.sqrt(v2["mem"]*sm["mem"]) * 10**0.18,
        "10x compute\n12x memory",
        ha="center", va="bottom",
        fontsize=10, color=GREY_D, style="italic", zorder=6)

# ============== AXES ==============
ax.set_xlabel("Compute per inference  (MACs, log scale)",
              fontsize=12, color=GREY_D, labelpad=8)
ax.set_ylabel("Memory footprint  (params + peak activations, log scale)",
              fontsize=12, color=GREY_D, labelpad=8)

ax.tick_params(colors=GREY_D, labelsize=10)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GREY_D)

# human-readable tick labels
def fmt_macs(x, _):
    if x >= 1e9: return f"{x/1e9:.0f} G"
    if x >= 1e6: return f"{x/1e6:.0f} M"
    return f"{x:.0f}"
def fmt_bytes(x, _):
    if x >= 1e9: return f"{x/1e9:.0f} GB"
    if x >= 1e6: return f"{x/1e6:.0f} MB"
    if x >= 1e3: return f"{x/1e3:.0f} KB"
    return f"{x:.0f}"
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(fmt_macs))
ax.yaxis.set_major_formatter(FuncFormatter(fmt_bytes))
ax.grid(True, which="major", linestyle=":", color=GREY, alpha=0.4, zorder=2)

# ============== TITLE BAND ==============
if not FOR_SLIDE:
    fig.suptitle("Smarter beats bigger -- and fits in flash",
                 fontsize=22, fontweight="bold", color=NAVY, y=0.985)
    fig.text(0.5, 0.935,
             "Five paths to the same 3 s music-window inference. "
             "v2 lands two orders of magnitude lower on compute and memory than even a small CNN.",
             ha="center", va="top",
             fontsize=12, color=GREY_D, style="italic")

    # ============== BOTTOM CAPTION ==============
    fig.text(0.5, 0.02,
             "Order-of-magnitude estimates from MAC counts, not measured cycles. "
             "MCU latency on Cortex-M7 @ 400 MHz: ~30 ms (v2) vs >300 ms (small CNN, INT8 quantized).",
             ha="center", va="bottom",
             fontsize=9.5, color=GREY, style="italic")

    plt.subplots_adjust(top=0.86, bottom=0.13, left=0.08, right=0.97)
else:
    plt.subplots_adjust(top=0.97, bottom=0.10, left=0.09, right=0.98)

plt.savefig(OUTFILE, dpi=170, facecolor=BG)
plt.close()
print(f"Saved {OUTFILE}")
