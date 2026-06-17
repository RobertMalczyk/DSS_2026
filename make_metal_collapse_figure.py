"""
make_metal_collapse_figure.py

Generates presentation_metal_collapse.png for the DSS 2026 talk.
A polished per-genre accuracy table showing that metal hit 0.00 in
four of eight test-time conditions, until the v2 173-dim feature
stack (H_v2) brought all three genres above 0.75.

Source: experiments_summary.txt (per-genre accuracies for A-H).
H_v2 per-genre numbers from presentation_story_full.txt -- only the
documented bound "all three above 0.75" is used.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette ----
NAVY     = "#0d1f3c"
GREY_D   = "#374151"
GREY_M   = "#6b7280"
GREY_L   = "#cbd5e1"
PAGE_BG  = "#f6f7fb"
CARD_BG  = "#ffffff"
ROW_ALT  = "#f8fafc"
HEADER_BG = "#eef2f7"

DEAD     = "#7f1d1d"
RED      = "#dc2626"
ORANGE   = "#fb923c"
YELLOW   = "#fde68a"
GREEN    = "#86c190"
GREEN_D  = "#15803d"


def cell_style(v):
    """(fill, text-color, label) for one accuracy cell."""
    if v is None:
        return GREEN, NAVY, "≥ 0.75"
    if v < 0.05:
        return DEAD,   "white", "0.00"
    if v < 0.30:
        return RED,    "white", f"{v:.2f}"
    if v < 0.55:
        return ORANGE, "white", f"{v:.2f}"
    if v < 0.80:
        return YELLOW, NAVY,    f"{v:.2f}"
    return GREEN,      NAVY,    f"{v:.2f}"


rows = [
    ("A",    "clean → clean",                      [0.99, 0.99, 0.99]),
    ("B",    "noisy → noisy",                      [0.90, 0.95, 0.88]),
    ("C",    "clean → noisy",                      [0.23, 0.00, 0.81]),
    ("D",    "clean → AI denoiser",                [0.99, 0.00, 0.22]),
    ("E",    "clean → 1st stage",                  [0.20, 0.81, 0.01]),
    ("F",    "clean → 2nd stage (classical+AI)",   [0.57, 0.33, 0.41]),
    ("G",    "clean → 6th stage (AdaptiveMS)",     [0.52, 0.00, 0.95]),
    ("H",    "clean → 7th stage (Component)",      [0.90, 0.00, 0.77]),
    ("H_v2", "+ 173-dim v2 features",                   [None, None, None]),
]

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor(PAGE_BG)
ax.set_facecolor(PAGE_BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

# ============ TITLE + BADGE ============
ax.text(7, 8.45, "Metal kept disappearing",
        ha="center", va="center",
        fontsize=26, fontweight="bold", color=NAVY)

# small red badge under the title
ax.add_patch(FancyBboxPatch((5.05, 7.78), 3.9, 0.38,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor=DEAD, edgecolor="none"))
ax.text(7, 7.97, "0.00 in 4 of 8 test-time filters",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="white")

# ============ CARD CONTAINER ============
CARD_X, CARD_Y = 0.4, 0.95
CARD_W, CARD_H = 13.2, 6.55

# subtle shadow
ax.add_patch(FancyBboxPatch((CARD_X + 0.06, CARD_Y - 0.06),
                            CARD_W, CARD_H,
                            boxstyle="round,pad=0.02,rounding_size=0.18",
                            facecolor="#dde2e8", edgecolor="none", alpha=0.6,
                            zorder=0))
# card
ax.add_patch(FancyBboxPatch((CARD_X, CARD_Y), CARD_W, CARD_H,
                            boxstyle="round,pad=0.02,rounding_size=0.18",
                            facecolor=CARD_BG, edgecolor=GREY_L,
                            linewidth=1.2, zorder=1))

# ============ COLUMN POSITIONS ============
COL_TAG    = 0.95
COL_COND   = 1.55
COL_JAZZ   = 8.50
COL_METAL  = 10.10
COL_POP    = 11.70

# ============ HEADER BAR ============
HEADER_Y_BOT = CARD_Y + CARD_H - 0.85
HEADER_H = 0.55
ax.add_patch(FancyBboxPatch((CARD_X + 0.25, HEADER_Y_BOT),
                            CARD_W - 0.50, HEADER_H,
                            boxstyle="round,pad=0.0,rounding_size=0.10",
                            facecolor=HEADER_BG, edgecolor="none", zorder=2))

H_CY = HEADER_Y_BOT + HEADER_H/2
ax.text(COL_TAG,   H_CY, "EXP",       ha="left",   va="center",
        fontsize=10.5, fontweight="bold", color=GREY_D, zorder=3)
ax.text(COL_COND,  H_CY, "CONDITION", ha="left",   va="center",
        fontsize=10.5, fontweight="bold", color=GREY_D, zorder=3)
ax.text(COL_JAZZ,  H_CY, "jazz",      ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=GREY_D, zorder=3)
ax.text(COL_METAL, H_CY, "metal",     ha="center", va="center",
        fontsize=13,   fontweight="bold", color=DEAD,   zorder=3)
ax.text(COL_POP,   H_CY, "pop",       ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=GREY_D, zorder=3)

# ============ DATA ROWS ============
ROW_H = 0.50
GAP_BEFORE_HV2 = 0.25

y = HEADER_Y_BOT - 0.45

for i, (tag, cond, accs) in enumerate(rows):
    if tag == "H_v2":
        y -= GAP_BEFORE_HV2
        # dashed separator above H_v2
        ax.plot([CARD_X + 0.4, CARD_X + CARD_W - 0.4],
                [y + 0.34, y + 0.34],
                color=GREEN_D, lw=1.0, linestyle=(0, (4, 3)), alpha=0.55,
                zorder=2)

    # alternating row stripe (skip H_v2 -- it gets its own treatment)
    if i % 2 == 1 and tag != "H_v2":
        ax.add_patch(Rectangle((CARD_X + 0.25, y - 0.24),
                               CARD_W - 0.50, 0.48,
                               facecolor=ROW_ALT, edgecolor="none", zorder=2))
    if tag == "H_v2":
        ax.add_patch(FancyBboxPatch((CARD_X + 0.25, y - 0.24),
                                    CARD_W - 0.50, 0.48,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor="#f0fdf4", edgecolor="none",
                                    zorder=2))

    color_tag = GREEN_D if tag == "H_v2" else NAVY
    weight    = "bold"  if tag == "H_v2" else "normal"

    ax.text(COL_TAG,  y, tag,  ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=color_tag, zorder=3)
    ax.text(COL_COND, y, cond, ha="left", va="center",
            fontsize=11, color=color_tag, fontweight=weight, zorder=3)

    for col_x, acc in zip([COL_JAZZ, COL_METAL, COL_POP], accs):
        fill, txtc, lbl = cell_style(acc)
        ax.add_patch(FancyBboxPatch((col_x - 0.62, y - 0.20),
                                    1.24, 0.40,
                                    boxstyle="round,pad=0.0,rounding_size=0.10",
                                    facecolor=fill, edgecolor="none",
                                    zorder=3))
        ax.text(col_x, y, lbl,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=txtc, zorder=4)

    y -= ROW_H

# ============ BOTTOM CALLOUT ============
ax.text(7, 0.45,
        "v2 173-dim features  —  metal stays alive end-to-end for the first time.",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=GREEN_D)

plt.savefig("presentation_metal_collapse.png", dpi=180,
            bbox_inches="tight", facecolor=PAGE_BG)
plt.close()
print("Saved presentation_metal_collapse.png")
