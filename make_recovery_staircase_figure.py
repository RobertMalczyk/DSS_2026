"""
make_recovery_staircase_figure.py

Generates presentation_recovery_staircase.png for slide 17 of the
DSS 2026 talk -- a graphic to replace the text-only green callout
box on the right of "Rebuild the features around what's left".

Visualizes the 34 % -> 49 % -> 70 % cliff-recovery progression as
three horizontal bars on a shared 0-100 % scale, where 0 % is C
(clean -> noisy collapse, accuracy 0.344) and 100 % is A
(clean -> clean ceiling, accuracy 0.988). Each bar maps an
intervention to its cumulative recovery.

Source numbers: experiments_summary.txt (A-H table) and
presentation_story_full.txt Chapter 3.4.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette tuned to the slide's green callout box ----
GREEN_BG    = "#d5efe0"   # the green box fill on slide 17
GREEN_LIGHT = "#a5d6b4"
GREEN_MID   = "#5cba78"
GREEN_DARK  = "#1f6e3a"
GREY_TRACK  = "#e8efe9"
NAVY        = "#0d1f3c"
GREY_D      = "#374151"
GREY        = "#6b7280"

fig, ax = plt.subplots(figsize=(14, 6.5))
fig.patch.set_facecolor(GREEN_BG)
ax.set_facecolor(GREEN_BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.5)
ax.axis("off")

# ============== TITLE ==============
ax.text(7, 6.05, "Cliff -> ceiling: 70 % closed",
        ha="center", va="center",
        fontsize=24, fontweight="bold", color=NAVY)
ax.text(7, 5.55,
        "clean_v2 -> 7th_v2 (H_v2): accuracy 0.796, all three genres above 0.75",
        ha="center", va="center",
        fontsize=12, color=GREY_D, style="italic")

# ============== BARS ==============
BAR_X0 = 4.7      # left edge of bars (room for technique labels on the left)
BAR_X1 = 11.7     # right edge (room for % labels on the right)
BAR_W  = BAR_X1 - BAR_X0
BAR_H  = 0.55

bars = [
    {"y": 4.20, "step": "1",
     "label": "Stage 7 filter alone",  "tag": "H",
     "pct": 34, "color": GREEN_LIGHT},
    {"y": 3.30, "step": "2",
     "label": "+ matched prefilter (LP @ 4 kHz)", "tag": "H_bf4000",
     "pct": 49, "color": GREEN_MID},
    {"y": 2.40, "step": "3",
     "label": "+ v2 173-dim feature stack", "tag": "H_v2",
     "pct": 70, "color": GREEN_DARK},
]

for b in bars:
    # Track (full 0-100 %)
    ax.add_patch(Rectangle((BAR_X0, b["y"]), BAR_W, BAR_H,
                           facecolor=GREY_TRACK, edgecolor="none"))
    # Filled portion (recovery so far)
    filled_w = BAR_W * b["pct"] / 100
    ax.add_patch(Rectangle((BAR_X0, b["y"]), filled_w, BAR_H,
                           facecolor=b["color"], edgecolor="none"))

    # Step number circle on the left
    cx = BAR_X0 - 3.65
    ax.add_patch(plt.Circle((cx, b["y"] + BAR_H/2), 0.18,
                            facecolor=GREEN_DARK, edgecolor="none"))
    ax.text(cx, b["y"] + BAR_H/2, b["step"],
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Technique label
    ax.text(cx + 0.30, b["y"] + BAR_H/2 + 0.12, b["label"],
            ha="left", va="center",
            fontsize=12, fontweight="bold", color=NAVY)
    ax.text(cx + 0.30, b["y"] + BAR_H/2 - 0.20, b["tag"],
            ha="left", va="center",
            fontsize=10, color=GREY_D, style="italic")

    # Percentage on the right
    ax.text(BAR_X1 + 0.15, b["y"] + BAR_H/2, f"{b['pct']} %",
            ha="left", va="center",
            fontsize=18, fontweight="bold", color=NAVY)

# ============== CLIFF / CEILING ANCHORS ==============
# vertical reference lines at 0 % and 100 %
ax.plot([BAR_X0, BAR_X0], [2.30, 4.85],
        color=GREY_D, lw=1.0, linestyle=":")
ax.plot([BAR_X1, BAR_X1], [2.30, 4.85],
        color=GREY_D, lw=1.0, linestyle=":")

# anchor labels under the bars
ax.text(BAR_X0, 2.10, "C  (cliff)\nacc 0.344",
        ha="center", va="top", fontsize=10, color=GREY_D)
ax.text(BAR_X1, 2.10, "A  (ceiling)\nacc 0.988",
        ha="center", va="top", fontsize=10, color=GREY_D)

# ============== BOTTOM MESSAGE ==============
ax.text(7, 1.10, "The single-genre collapse is gone.",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=GREEN_DARK)
ax.text(7, 0.65,
        "Each step is a domain-knowledge decision -- not a grid search.",
        ha="center", va="center",
        fontsize=11, color=GREY_D, style="italic")

plt.savefig("presentation_recovery_staircase.png", dpi=170,
            bbox_inches="tight", facecolor=GREEN_BG)
plt.close()
print("Saved presentation_recovery_staircase.png")
