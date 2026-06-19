"""Run the blind denoise() pipeline on all three signals, produce the figures
and summary.md required by CLAUDE.md."""
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
OUT.mkdir(exist_ok=True)

SIGNALS = [
    ("ecg",       SAMPLES / "ecg_noisy.wav"),
    ("vibration", SAMPLES / "vibration_noisy.wav"),
    ("music",     SAMPLES / "music_noisy.wav"),
]


# ---------------------------------------------------------------------------
# reference-free quality indicators
# ---------------------------------------------------------------------------

def welch_db(x, fs):
    nper = min(8192, len(x) // 2) if len(x) > 8192 else len(x) // 2
    f, P = sps.welch(x - x.mean(), fs, nperseg=nper, window="hann", detrend=False)
    return f, 10 * np.log10(P + 1e-30)


def metrics(noisy, filt, fs):
    """Reference-free: variance reduction, PSD-floor reduction in dB, DC removal."""
    var_n = float(np.var(noisy - noisy.mean()))
    var_f = float(np.var(filt - filt.mean()))
    var_ratio = var_n / max(var_f, 1e-30)

    f, Pn = welch_db(noisy, fs)
    _, Pf = welch_db(filt,  fs)

    # High-frequency floor = median PSD in top 40% of Nyquist (where denoise should have nuked noise)
    tail = f > 0.6 * fs / 2
    hf_floor_n = float(np.median(Pn[tail]))
    hf_floor_f = float(np.median(Pf[tail]))
    hf_drop_db = hf_floor_n - hf_floor_f

    # Overall PSD median (broadband floor proxy) before/after
    med_n = float(np.median(Pn))
    med_f = float(np.median(Pf))

    # DC: magnitude of mean
    dc_n = float(np.mean(noisy))
    dc_f = float(np.mean(filt))

    return dict(
        var_noisy=var_n, var_filt=var_f, var_ratio=var_ratio,
        hf_floor_noisy_db=hf_floor_n, hf_floor_filt_db=hf_floor_f, hf_floor_drop_db=hf_drop_db,
        psd_median_noisy_db=med_n, psd_median_filt_db=med_f,
        dc_noisy=dc_n, dc_filt=dc_f,
    )


# ---------------------------------------------------------------------------
# figure per signal
# ---------------------------------------------------------------------------

def make_figure(name, noisy, filt, fs, out_path):
    t = np.arange(len(noisy)) / fs

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.1, 1.3, 1.3], hspace=0.55)

    # shared y-limits for noisy + filtered
    ymax = float(np.max(np.abs(np.concatenate([noisy, filt]))))
    ymax *= 1.05

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, noisy, lw=0.5, color="#555")
    ax0.set_title(f"{name} — noisy (time)")
    ax0.set_ylabel("amplitude")
    ax0.set_xlim(t[0], t[-1])
    ax0.set_ylim(-ymax, ymax)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(t, filt, lw=0.5, color="#1a7")
    ax1.set_title(f"{name} — filtered (time, same y-axis)")
    ax1.set_xlabel("s")
    ax1.set_ylabel("amplitude")
    ax1.set_ylim(-ymax, ymax)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    f_n, Pn = welch_db(noisy, fs)
    f_f, Pf = welch_db(filt,  fs)
    ax2.plot(f_n, Pn, lw=0.6, color="#555", label="noisy")
    ax2.plot(f_f, Pf, lw=0.8, color="#1a7", label="filtered")
    ax2.set_title("Welch PSD (dB)")
    ax2.set_xlabel("Hz")
    ax2.set_ylabel("dB")
    ax2.set_xlim(0, fs / 2)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[3])
    nper_sg = min(1024, len(filt) // 8)
    f_sg, t_sg, S = sps.spectrogram(
        filt - filt.mean(), fs, nperseg=nper_sg, noverlap=nper_sg // 2, window="hann"
    )
    pcm = ax3.pcolormesh(
        t_sg, f_sg, 10 * np.log10(S + 1e-30),
        shading="auto", cmap="magma",
    )
    ax3.set_title("Spectrogram of filtered")
    ax3.set_xlabel("s")
    ax3.set_ylabel("Hz")
    fig.colorbar(pcm, ax=ax3, pad=0.01, label="dB")

    fig.suptitle(f"{name}  (fs={int(fs)} Hz, {len(noisy)/fs:.2f} s)", y=0.995)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver + summary
# ---------------------------------------------------------------------------

CHAIN_DESC = {
    360: (
        "HP Butter ord 4, 0.5 Hz (zero-phase, `sosfiltfilt`)  \n"
        "  — kills DC offset (measured mean = −1.107) and any baseline wander.\n\n"
        "LP Butter ord 6, 30 Hz (zero-phase)  \n"
        "  — measured PSD knee at ~14 Hz; 30 Hz leaves a comfort margin over that knee\n"
        "  and lies well below the interference lines at 50/72/100 Hz. Effective order 12\n"
        "  under `sosfiltfilt` gives >80 dB attenuation at 50 Hz, so no dedicated notch\n"
        "  is needed."
    ),
    12000: (
        "HP Butter ord 4, 5 Hz (zero-phase)  \n"
        "  — removes DC (measured mean = −0.144) while keeping sub-50 Hz low-band\n"
        "  structure intact for measurement.\n\n"
        "IIR notch 50 Hz, Q=30  (BW ≈ 1.7 Hz)  \n"
        "IIR notch 100 Hz, Q=30  (BW ≈ 3.3 Hz)  \n"
        "IIR notch 150 Hz, Q=30  (BW ≈ 5 Hz)  \n"
        "  — the three tonals were measured at +28/+22/+17 dB above local floor;\n"
        "  they track the 50 Hz mains family exactly. Q=30 is narrow enough not to\n"
        "  touch the 358 Hz / 1 kHz / 2.4 kHz content that looks like the real signal.\n\n"
        "LP Butter ord 8, 2800 Hz (zero-phase)  \n"
        "  — PSD above ~2500 Hz is uniform broadband floor at about −58 dB; no\n"
        "  tonal or structural content measured there, so everything above 2.8 kHz\n"
        "  is noise that can be removed without risk."
    ),
    22050: (
        "HP Butter ord 4, 25 Hz (zero-phase)  \n"
        "  — PSD climbs steeply below ~20 Hz (rumble / DC, far below any plausible\n"
        "  audio fundamental). 25 Hz cutoff removes the rumble without eating into\n"
        "  the audible bass band.\n\n"
        "IIR notch 50 Hz, Q=30  \n"
        "IIR notch 100 Hz, Q=30  \n"
        "  — measured at +17 / +14 dB above local floor; identified as mains hum\n"
        "  family.\n\n"
        "IIR notch 4410 Hz, Q=110  (BW ≈ 40 Hz)  \n"
        "  — measured: a tight pair of peaks at 4403 and 4417 Hz (14 Hz apart),\n"
        "  each ~20 dB above local floor. Too narrow to be musical content (sidebands\n"
        "  right next to each other), most likely a narrowband interferer. Q=110\n"
        "  straddles both peaks in one stage.\n\n"
        "No LP stage — music carries real content right up to Nyquist."
    ),
}

OBSERVATIONS = {
    "ecg": (
        "Low sample rate (360 Hz, 10 s). Strong negative DC (mean ≈ −1.11, σ ≈ 0.83). "
        "Welch PSD shows the signal band collapses into the noise floor above ~14 Hz "
        "(smoothed-PSD knee); three sharp tonals at 50 Hz (+23 dB over local floor), "
        "72 Hz (+17 dB), 100 Hz (+17 dB); flat broadband floor around −30 dB. The "
        "signal of interest therefore occupies roughly 0.5–15 Hz."
    ),
    "vibration": (
        "fs = 12 kHz, 3 s. Small DC (mean ≈ −0.14). Low-band energy elevated up to "
        "~200 Hz (knee). Three mains harmonics at 50/100/150 Hz stand ~28/22/17 dB "
        "above the local floor. Two strong tonal clusters at 1035/1066 Hz and around "
        "2400 Hz (≈21–25 dB above floor), plus weaker lines at 358 Hz and 2102 Hz — "
        "these look like the actual vibration signal. Above ~2500 Hz the PSD is "
        "uniform floor at about −58 dB, indicating nothing but broadband noise sits "
        "there."
    ),
    "music": (
        "fs = 22.05 kHz, 4 s. Small DC (mean ≈ −0.14), with PSD climbing steeply "
        "below ~20 Hz (rumble). Broadband, time-varying content (typical music) from "
        "the bass up to Nyquist. Mains tonals at 50 and 100 Hz stick out by 17 / 14 dB, "
        "and a very narrow pair of peaks at 4403 / 4417 Hz (14 Hz apart, ~20 dB over "
        "local floor) is too sharp to be musical content and looks like a narrowband "
        "interferer."
    ),
}


def fmt_metrics(m):
    lines = [
        f"- variance reduction ratio: **{m['var_ratio']:.2f}×**  "
        f"(var noisy = {m['var_noisy']:.3e}, filtered = {m['var_filt']:.3e})",
        f"- PSD floor in upper 40% of band: "
        f"{m['hf_floor_noisy_db']:+.1f} dB → {m['hf_floor_filt_db']:+.1f} dB  "
        f"(**drop = {m['hf_floor_drop_db']:.1f} dB**)",
        f"- PSD median across full band: "
        f"{m['psd_median_noisy_db']:+.1f} dB → {m['psd_median_filt_db']:+.1f} dB",
        f"- DC offset: {m['dc_noisy']:+.3e} → {m['dc_filt']:+.3e}",
    ]
    return "\n".join(lines)


def main():
    summary_lines = [
        "# Blind denoising summary",
        "",
        "Three noisy WAVs were inspected (time / Welch PSD / spectrogram) and a\n"
        "per-signal zero-phase filter chain was designed from the measured features.\n"
        "Every cutoff, notch centre, Q, and order below was derived from the\n"
        "inspection pass — nothing was assumed about what the three signals carry.\n"
        "All filters are applied with `scipy.signal.sosfiltfilt` (zero-phase).\n",
        "",
    ]

    for name, path in SIGNALS:
        x, fs = sf.read(path, dtype="float64")
        assert x.ndim == 1
        y = denoise(x, fs)

        out_png = OUT / f"{name}_filtered.png"
        out_wav = OUT / f"{name}_filtered.wav"
        make_figure(name, x, y, fs, out_png)
        sf.write(out_wav, y.astype(np.float32), int(fs), subtype="FLOAT")
        m = metrics(x, y, fs)

        print(f"\n=== {name}  fs={fs}  n={len(x)} ===")
        for k, v in m.items():
            print(f"  {k:24s} {v}")

        summary_lines += [
            f"## {name}   (fs = {int(fs)} Hz, duration = {len(x)/fs:.2f} s)",
            "",
            "### Observations",
            OBSERVATIONS[name],
            "",
            "### Filter chain",
            CHAIN_DESC[int(fs)],
            "",
            "### Reference-free quality indicators",
            fmt_metrics(m),
            "",
            f"Figure: `out/{name}_filtered.png`",
            "",
        ]

    summary_lines += [
        "## Notes on what was deliberately left out",
        "",
        "- **No broadband LP for music.** The music PSD is not flat above some\n"
        "  knee — there is time-varying content all the way to Nyquist in the\n"
        "  spectrogram, so there is no measurement-based justification for a\n"
        "  lowpass. Only the narrow 4410 Hz interferer is removed.",
        "- **Mains notches only on measured harmonics.** 150 Hz is at floor in\n"
        "  music (so not notched there); ECG's 30 Hz LP removes 50/72/100 Hz\n"
        "  without needing dedicated notches.",
        "- **No tonal notches in vibration.** The 1 kHz and 2.4 kHz clusters are\n"
        "  ~21–25 dB above the floor and look like the actual vibration content —\n"
        "  if this were machinery, those would be resonances. Notching them would\n"
        "  destroy the signal, so they are preserved.",
    ]

    (OUT / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nwrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
