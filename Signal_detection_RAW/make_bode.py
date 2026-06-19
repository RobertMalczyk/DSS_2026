"""Bode plots (magnitude + phase) of the three filter chains defined in
denoise.get_chain(). Writes out/bode.png.

Note: the chains are applied with `sosfiltfilt` (forward + backward), so the
effective magnitude is |H|^2 (i.e. dB doubled) and the effective phase is
zero. The plots below show the single-pass design response — the standard
Bode characteristic of the filter itself — with a note on zero-phase usage.
"""
from pathlib import Path
import numpy as np
from scipy import signal as sps
import matplotlib.pyplot as plt

from denoise import get_chain

ROOT = Path(__file__).parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

SIGNALS = [
    ("ecg",       360),
    ("vibration", 12000),
    ("music",     22050),
]

N_POINTS = 4096


def cascade_response(stages, fs):
    """Return (w_hz, H_total, per_stage_H). H's are complex."""
    # log-spaced freq grid from 0.1 Hz up to Nyquist * 0.999
    w = np.logspace(-1, np.log10(fs / 2 * 0.999), N_POINTS)
    H_total = np.ones_like(w, dtype=np.complex128)
    per_stage = []
    for label, sos in stages:
        _, H = sps.sosfreqz(sos, worN=w, fs=fs)
        per_stage.append((label, H))
        H_total = H_total * H
    return w, H_total, per_stage


def plot_bode():
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    for row, (name, fs) in enumerate(SIGNALS):
        stages = get_chain(fs)
        w, H, per_stage = cascade_response(stages, fs)

        mag_db = 20 * np.log10(np.abs(H) + 1e-30)
        phase_deg = np.unwrap(np.angle(H)) * 180 / np.pi

        ax_m = axes[row, 0]
        ax_p = axes[row, 1]

        # individual stages in light grey
        for label, Hk in per_stage:
            mk = 20 * np.log10(np.abs(Hk) + 1e-30)
            pk = np.unwrap(np.angle(Hk)) * 180 / np.pi
            ax_m.semilogx(w, mk, lw=0.8, alpha=0.55, label=label)
            ax_p.semilogx(w, pk, lw=0.8, alpha=0.55, label=label)

        # cascade in thick green
        ax_m.semilogx(w, mag_db,   lw=2.0, color="#1a7", label="CASCADE")
        ax_p.semilogx(w, phase_deg, lw=2.0, color="#1a7", label="CASCADE")

        ax_m.set_title(f"{name}  magnitude  (fs = {fs} Hz)")
        ax_m.set_xlabel("Hz")
        ax_m.set_ylabel("|H|  (dB)")
        ax_m.set_ylim(-120, 5)
        ax_m.grid(which="both", alpha=0.3)
        ax_m.legend(fontsize=7, loc="lower left")

        ax_p.set_title(f"{name}  phase")
        ax_p.set_xlabel("Hz")
        ax_p.set_ylabel("∠H  (deg)")
        ax_p.grid(which="both", alpha=0.3)
        ax_p.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Bode diagrams of denoise() chains — single-pass design response.\n"
        "Actual pipeline uses sosfiltfilt ⇒ effective magnitude = |H|² (dB ×2), phase = 0.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = OUT / "bode.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plot_bode()
