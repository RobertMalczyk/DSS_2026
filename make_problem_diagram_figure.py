"""
make_problem_diagram_figure.py

Generates presentation_problem_diagram.png for the DSS 2026 talk.

English remake of Input/llm_fail.png. Preserves the original
slide's intent -- show the simplified LLM pipeline (1..5 + response,
with the transformer expanded into its three internal stages) and
point at the architectural reason the empirical-iterative failure
mode (Type 2 in the entry thesis) surfaces.

Verified literature anchors:
  Architecture & training
    - Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
    - Radford et al., GPT-2 / GPT-3 (2019, 2020)
    - Ouyang et al., InstructGPT-RLHF (NeurIPS 2022)
    - Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)
    - Bai et al., "Constitutional AI" (Anthropic, 2022)

  Planning / failure mode
    - Valmeekam et al., "On the Planning Abilities of LLMs" (NeurIPS 2023)
    - Dziri et al., "Faith and Fate" (NeurIPS 2023)
    - Yao et al., "Tree of Thoughts" (NeurIPS 2023)
    - Hao et al., RAP (EMNLP 2023)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette (close to the original PPT template) ----
NAVY      = "#0d1f3c"
NAVY_BAR  = "#1f497d"
LIGHT_B   = "#e8f0fa"
ACCENT    = "#c0504d"
ACCENT_L  = "#fdeae8"
GREY_D    = "#374151"
GREY      = "#6b7280"
WHITE     = "#ffffff"
TAKE_BG   = "#dbeafe"
ARCH_BG   = "#f8fafc"

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# ============== TITLE ==============
ax.text(8, 9.55,
        "Simplified LLM pipeline -- where the problem surfaces",
        ha="center", va="center",
        fontsize=24, fontweight="bold", color=NAVY)
ax.text(8, 9.05,
        "what the architecture does, and what it leaves unsolved",
        ha="center", va="center",
        fontsize=14, color=GREY, style="italic")

# ============== PIPELINE STAGES ==============
STAGE_Y = 6.65
STAGE_H = 1.80

stages = [
    {"cx": 1.30,  "w": 2.00, "num": "1",
     "title": "Training data",
     "lines": ["text, code,", "dialogues"],
     "fill": LIGHT_B,  "border": NAVY_BAR},

    {"cx": 3.85,  "w": 2.00, "num": "2",
     "title": "Pretraining",
     "lines": ["next-token", "prediction"],
     "fill": ACCENT_L, "border": ACCENT},

    {"cx": 6.95,  "w": 3.20, "num": "3",
     "title": "Transformer model",
     "lines": [],   # sub-boxes drawn separately
     "fill": LIGHT_B,  "border": NAVY_BAR},

    {"cx": 10.05, "w": 2.00, "num": "4",
     "title": "Post-training",
     "lines": ["instruction tuning",
               "RLHF / DPO",
               "(Constitutional AI)"],
     "fill": LIGHT_B,  "border": NAVY_BAR},

    {"cx": 12.55, "w": 2.00, "num": "5",
     "title": "Inference",
     "lines": ["user prompt,",
               "token-by-token"],
     "fill": ACCENT_L, "border": ACCENT},
]
RESP_CX, RESP_W = 14.95, 1.50

# Draw the five stage boxes
for s in stages:
    box = FancyBboxPatch((s["cx"] - s["w"]/2, STAGE_Y),
                         s["w"], STAGE_H,
                         boxstyle="round,pad=0.04",
                         facecolor=s["fill"], edgecolor=s["border"],
                         linewidth=1.8)
    ax.add_patch(box)
    # number circle above the box
    cy = STAGE_Y + STAGE_H + 0.28
    ax.add_patch(Circle((s["cx"], cy), 0.27,
                        facecolor=s["border"], edgecolor="none"))
    ax.text(s["cx"], cy, s["num"],
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    # title
    ax.text(s["cx"], STAGE_Y + STAGE_H - 0.32, s["title"],
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY)
    # subtitle lines
    line_y0 = STAGE_Y + STAGE_H - 0.78
    for j, line in enumerate(s["lines"]):
        ax.text(s["cx"], line_y0 - 0.30*j, line,
                ha="center", va="center",
                fontsize=10.5, color=s["border"])

# Sub-boxes inside the Transformer (stage 3)
S3 = stages[2]
sub_centers = [S3["cx"] - 1.00, S3["cx"], S3["cx"] + 1.00]
sub_labels  = ["Tokenization\n+ embeddings",
               "Attention\n+ MLP",
               "Contextual\nrepresentations"]
for sx, lbl in zip(sub_centers, sub_labels):
    sb = FancyBboxPatch((sx - 0.45, STAGE_Y + 0.15),
                        0.90, 1.05,
                        boxstyle="round,pad=0.02",
                        facecolor=WHITE, edgecolor=NAVY_BAR,
                        linewidth=1.0)
    ax.add_patch(sb)
    ax.text(sx, STAGE_Y + 0.68, lbl,
            ha="center", va="center", fontsize=9.5, color=NAVY)

# Response box at the right end
rbox = FancyBboxPatch((RESP_CX - RESP_W/2, STAGE_Y + 0.40),
                      RESP_W, STAGE_H - 0.80,
                      boxstyle="round,pad=0.04",
                      facecolor=WHITE, edgecolor=NAVY,
                      linewidth=1.8)
ax.add_patch(rbox)
ax.text(RESP_CX, STAGE_Y + STAGE_H/2, "Response",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=NAVY)

# Horizontal arrows linking the stages
arrow_y = STAGE_Y + STAGE_H/2
all_centers = [s["cx"] for s in stages] + [RESP_CX]
all_widths  = [s["w"]  for s in stages] + [RESP_W]
for i in range(len(all_centers) - 1):
    x1 = all_centers[i]   + all_widths[i]/2   + 0.06
    x2 = all_centers[i+1] - all_widths[i+1]/2 - 0.06
    ax.add_patch(FancyArrowPatch((x1, arrow_y), (x2, arrow_y),
                                 arrowstyle="->", color=GREY_D,
                                 mutation_scale=18, linewidth=1.6))

# ============== ARCHITECTURE CALLOUT ==============
# vertical connector from stage 3 into the callout box
ax.add_patch(FancyArrowPatch((S3["cx"], STAGE_Y - 0.03),
                             (S3["cx"], 6.40),
                             arrowstyle="-", color=GREY, linewidth=1.2))

ax.add_patch(FancyBboxPatch((1.0, 5.05), 14.0, 1.35,
                            boxstyle="round,pad=0.04",
                            facecolor=ARCH_BG, edgecolor=NAVY_BAR,
                            linewidth=1.2))
ax.text(8, 6.10,
        "The transformer decoder is autoregressive: "
        "each output token is sampled from  p(x_t | x_<t).",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=NAVY)
ax.text(8, 5.65,
        "There is no explicit goal / planner module in the architecture --",
        ha="center", va="center", fontsize=12, color=NAVY)
ax.text(8, 5.30,
        "the model continues text fluently, but lacks a separate "
        "mechanism for strategic goal management.",
        ha="center", va="center", fontsize=12, color=NAVY)

# ============== RED BANNER ==============
BANNER_Y = 4.20
ax.add_patch(FancyBboxPatch((1.0, BANNER_Y), 14.0, 0.65,
                            boxstyle="round,pad=0.03",
                            facecolor=ACCENT, edgecolor="none"))
ax.text(8, BANNER_Y + 0.32,
        "⚠   This is where the problem typically surfaces",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="white")

# Bullets with sub-explanations
bullets = [
    ("local optimization of the next step",
     "the autoregressive sampler maximizes p(x_t | x_<t), "
     "not a global objective"),
    ("decomposition is easy",
     "splitting a task into stages plays to the model's "
     "text-completion strength"),
    ("holding one overarching thesis is harder",
     "no explicit working memory of the goal -- long arcs "
     "drift toward local context"),
]
b_y0 = 3.65
b_spacing = 0.55
for i, (head, tail) in enumerate(bullets):
    y = b_y0 - i*b_spacing
    ax.text(1.50, y, "●",
            ha="left", va="center", fontsize=13, color=ACCENT)
    ax.text(1.95, y, head,
            ha="left", va="center",
            fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(1.95, y - 0.26, tail,
            ha="left", va="center",
            fontsize=10.5, color=GREY_D, style="italic")

# ============== TAKEAWAY ==============
TAKE_Y = 1.20
ax.add_patch(FancyBboxPatch((0.5, TAKE_Y), 15.0, 0.70,
                            boxstyle="round,pad=0.02",
                            facecolor=TAKE_BG, edgecolor=NAVY_BAR,
                            linewidth=1.2))
ax.text(0.95, TAKE_Y + 0.36, "Takeaway:",
        ha="left", va="center",
        fontsize=14, fontweight="bold", color=NAVY_BAR)
ax.text(2.85, TAKE_Y + 0.36,
        "LLMs decompose tasks well, but without external structure "
        "they drift from the guiding thesis and fail to converge.",
        ha="left", va="center", fontsize=12, color=NAVY)

# ============== LITERATURE ==============
ax.text(8, 0.72,
        "Architecture & training:  Vaswani et al. NeurIPS 2017 "
        "(Attention Is All You Need)   "
        "Ouyang et al. NeurIPS 2022 (InstructGPT/RLHF)   "
        "Rafailov et al. NeurIPS 2023 (DPO)   "
        "Bai et al. 2022 (Constitutional AI)",
        ha="center", va="center", fontsize=9.5, color=GREY)
ax.text(8, 0.38,
        "Failure mode:   Valmeekam et al. NeurIPS 2023 (Planning)   "
        "Dziri et al. NeurIPS 2023 (Faith and Fate)   "
        "Yao et al. NeurIPS 2023 (Tree of Thoughts)   "
        "Hao et al. EMNLP 2023 (RAP)",
        ha="center", va="center", fontsize=9.5, color=GREY)

plt.tight_layout()
plt.savefig("presentation_problem_diagram.png", dpi=170,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved presentation_problem_diagram.png")
