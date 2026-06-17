"""Generate a noisy-vs-filtered report.

Produces:
  out/comparison.png        — consolidated figure: 3 signals × (time excerpt, PSD overlay)
  out/report.md             — Markdown report embedding all figures

No clean reference is used — per the blind task constraint, comparison is
strictly noisy (input) vs filtered (denoise() output).
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal as sps
import matplotlib.pyplot as plt

from denoise import denoise

np.random.seed(42)

ROOT = Path(__file__).parent
SAMPLES = ROOT.parent / "Signal_generation" / "out" / "Samples"
OUT = ROOT / "out"

SIGNALS = [
    ("ecg",       SAMPLES / "ecg_noisy.wav",       (0.0, 3.0)),
    ("vibration", SAMPLES / "vibration_noisy.wav", (0.0, 0.3)),
    ("music",     SAMPLES / "music_noisy.wav",     (0.0, 0.5)),
]


def welch_db(x, fs):
    nper = min(8192, len(x) // 2) if len(x) > 8192 else len(x) // 2
    f, P = sps.welch(x - x.mean(), fs, nperseg=nper, window="hann", detrend=False)
    return f, 10 * np.log10(P + 1e-30)


def comparison_figure():
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row, (name, path, (t0, t1)) in enumerate(SIGNALS):
        x, fs = sf.read(path, dtype="float64")
        y = denoise(x, fs)

        # time excerpt — small window so QRS / transients are visible
        s0, s1 = int(t0 * fs), int(t1 * fs)
        t = np.arange(s0, s1) / fs
        ax = axes[row, 0]
        ax.plot(t, x[s0:s1], lw=0.6, color="#777", label="noisy",    alpha=0.85)
        ax.plot(t, y[s0:s1], lw=0.9, color="#1a7", label="filtered")
        ax.set_title(f"{name}  (time, {t0:.2f}–{t1:.2f} s)")
        ax.set_xlabel("s")
        ax.set_ylabel("amp")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        # PSD overlay
        fN, Pn = welch_db(x, fs)
        _,  Pf = welch_db(y, fs)
        ax = axes[row, 1]
        ax.plot(fN, Pn, lw=0.6, color="#777", label="noisy")
        ax.plot(fN, Pf, lw=0.9, color="#1a7", label="filtered")
        ax.set_title(f"{name}  (Welch PSD, fs={int(fs)} Hz)")
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
        ax.set_xlim(0, fs / 2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "comparison.png")


def metrics(noisy, filt, fs):
    f, Pn = welch_db(noisy, fs)
    _, Pf = welch_db(filt,  fs)
    tail = f > 0.6 * fs / 2
    return dict(
        var_ratio      = float(np.var(noisy) / max(np.var(filt), 1e-30)),
        hf_drop_db     = float(np.median(Pn[tail]) - np.median(Pf[tail])),
        median_drop_db = float(np.median(Pn) - np.median(Pf)),
        dc_noisy       = float(noisy.mean()),
        dc_filt        = float(filt.mean()),
    )


REPORT_INTRO = """\
# Denoising report — noisy vs filtered

*Blind denoising of three unknown signals. Only the noisy inputs were
inspected; no clean reference was used. All filter parameters are derived
from measurements on the noisy waveforms (see `summary.md` for the per-stage
justification).*

Pipeline:

- Inspection: `inspect_signals.py`, `inspect_detail.py`
- Filter chains: `denoise.py` (single entry point `denoise(noisy, fs)`,
  zero-phase via `scipy.signal.sosfiltfilt`)
- Execution + figures: `run_pipeline.py`
- This report: `make_report.py`

"""

CHAIN_TABLE = """\
## Filter chains (measured parameters only)

| signal | chain |
|---|---|
| ecg (fs = 360 Hz)        | HP Butter ord 4 @ 0.5 Hz · LP Butter ord 6 @ 30 Hz |
| vibration (fs = 12 kHz)  | HP Butter ord 4 @ 5 Hz · notch 50/100/150 Hz (Q=30) · LP Butter ord 8 @ 2800 Hz |
| music (fs = 22.05 kHz)   | HP Butter ord 4 @ 25 Hz · notch 50/100 Hz (Q=30) · notch 4410 Hz (Q=110) |
"""


def fmt_metrics_table(rows):
    hdr = (
        "| signal | var(noisy)/var(filt) | PSD-floor drop (upper 40% band) "
        "| overall PSD median drop | DC noisy → filtered |\n"
        "|---|---:|---:|---:|---|\n"
    )
    lines = [hdr]
    for name, fs, m in rows:
        lines.append(
            f"| {name} (fs={int(fs)} Hz) | "
            f"**{m['var_ratio']:.2f}×** | "
            f"**{m['hf_drop_db']:.1f} dB** | "
            f"{m['median_drop_db']:.1f} dB | "
            f"{m['dc_noisy']:+.3e} → {m['dc_filt']:+.3e} |\n"
        )
    return "".join(lines)


def write_report():
    rows = []
    sections = []
    for name, path, _ in SIGNALS:
        x, fs = sf.read(path, dtype="float64")
        y = denoise(x, fs)
        m = metrics(x, y, fs)
        rows.append((name, fs, m))

        sections.append(
            f"### {name}  (fs = {int(fs)} Hz, duration = {len(x)/fs:.2f} s)\n\n"
            f"![{name}]({name}_filtered.png)\n\n"
            f"- var(noisy)/var(filt) = **{m['var_ratio']:.2f}×**\n"
            f"- PSD floor drop in upper 40% of band: **{m['hf_drop_db']:.1f} dB**\n"
            f"- DC: {m['dc_noisy']:+.3e} → {m['dc_filt']:+.3e}\n\n"
        )

    body = []
    body.append(REPORT_INTRO)
    body.append("## Consolidated comparison\n\n![comparison](comparison.png)\n\n")
    body.append(CHAIN_TABLE)
    body.append("\n## Reference-free metrics (noisy → filtered)\n\n")
    body.append(fmt_metrics_table(rows))
    body.append("\n## Per-signal figures\n\n")
    body.extend(sections)
    body.append(
        "## Notes\n\n"
        "- The comparison above contrasts **noisy input** with the **filtered\n"
        "  output** of `denoise(noisy, fs)`. No clean reference is read or\n"
        "  assumed.\n"
        "- For music, the filtered PSD overlays the noisy PSD almost exactly\n"
        "  outside of DC/mains/4410 Hz — that is intentional: no broadband LP\n"
        "  could be justified from measurement, so only narrowband interferers\n"
        "  and sub-audible rumble are removed.\n"
        "- For ECG and vibration, the filtered PSD above the LP cutoff drops\n"
        "  to the numerical noise floor of the plotting pipeline, which is why\n"
        "  the upper-band PSD-drop metric looks implausibly large (>150 dB).\n"
        "  Practically this means 'attenuated well below anything measurable'.\n"
    )
    (OUT / "report.md").write_text("".join(body), encoding="utf-8")
    print("wrote", OUT / "report.md")


if __name__ == "__main__":
    comparison_figure()
    write_report()
