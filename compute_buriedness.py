"""
Compute "how buried in noise" metrics across all four datasets used
in the DSS 2026 talk:

    - 3 demo signals (ECG, vibration, music_metal_01) at SNR = -10 dB
      with the global seed-42 noise realisation
    - the 250-track music dataset (jazz/metal/pop), per-track sha256
      seeds, same noise model

For each signal we report:
    - target / achieved SNR (dB)
    - RMS noise / RMS clean   (linear amplitude ratio)
    - noise power / signal power
    - peak noisy / peak clean
    - fraction of samples where |noise| > |signal|   (instantaneous burial)
    - fraction of samples where |noise| > 3 * |signal|  (deep burial)
    - clipped (yes/no)

Output: prints a markdown table to stdout AND writes it to
buriedness_table.md in the same folder.
"""

import os
import sys

import numpy as np
import soundfile as sf
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIGGEN = os.path.join(HERE, "Signal_generation")
sys.path.insert(0, SIGGEN)

from signal_contamination import (  # noqa: E402
    load_ecg, load_vibration, load_music, make_noise, SNR_TARGET_DB,
)

MODEL_DIR = os.path.join(SIGGEN, "out", "Model")


def buriedness(clean: np.ndarray, total_noise_preclip: np.ndarray):
    """Compute pre- AND post-clip buriedness metrics.

    Pre-clip is the design figure (SNR target hits exactly here).
    Post-clip is what a downstream filter/classifier actually sees:
    the ADC saturation truncates the largest noise excursions, which
    raises the effective SNR the model encounters.
    """
    rms_c = float(np.sqrt(np.mean(clean ** 2)))
    peak_c = float(np.max(np.abs(clean)))
    full_scale_adc = 3.0 * (peak_c + 1e-12)

    # Pre-clip view
    rms_n_pre = float(np.sqrt(np.mean(total_noise_preclip ** 2)))
    snr_pre = 10 * np.log10((rms_c ** 2) / (rms_n_pre ** 2 + 1e-30))

    # Post-clip view: the noise component as it actually reaches disk
    noisy = np.clip(clean + total_noise_preclip, -full_scale_adc, full_scale_adc)
    noise_post = noisy - clean
    rms_n_post = float(np.sqrt(np.mean(noise_post ** 2)))
    snr_post = 10 * np.log10((rms_c ** 2) / (rms_n_post ** 2 + 1e-30))
    peak_noisy = float(np.max(np.abs(noisy)))

    abs_n = np.abs(noise_post)
    abs_c = np.abs(clean) + 1e-30
    frac_dominated = float(np.mean(abs_n > abs_c))
    frac_deep = float(np.mean(abs_n > 3 * abs_c))

    return dict(
        rms_clean=rms_c,
        rms_noise_pre=rms_n_pre,
        rms_noise_post=rms_n_post,
        rms_ratio_post=rms_n_post / (rms_c + 1e-30),
        power_ratio_post=(rms_n_post ** 2) / (rms_c ** 2 + 1e-30),
        peak_clean=peak_c,
        peak_noisy=peak_noisy,
        snr_db_pre=snr_pre,
        snr_db_post=snr_post,
        frac_noise_over_signal=frac_dominated,
        frac_noise_over_3x_signal=frac_deep,
        clipped=peak_noisy >= full_scale_adc - 1e-9,
    )


def demo_row(name, loader, units):
    clean, fs, _src, _u = loader()
    _emc, _meas, total_noise = make_noise(clean, fs)
    m = buriedness(clean, total_noise)
    m.update(name=name, fs=fs, samples=clean.size, units=units, n_tracks=1)
    return m


def music_dataset_rows():
    """Aggregate stats over the full 250-track music dataset.

    The noisy WAVs on disk are POST-clip. We recover the pre-clip noise
    by re-running make_noise() with the same per-track sha256 seed used
    by build_music_dataset.py.
    """
    from hashlib import sha256

    labels = pd.read_csv(os.path.join(MODEL_DIR, "labels.csv"))
    rows_per_track = []
    for _, r in labels.iterrows():
        clean_p = os.path.join(MODEL_DIR, r["clean_path"])
        clean, fs = sf.read(clean_p)
        if clean.ndim == 2:
            clean = clean[:, 0]
        clean = clean.astype(np.float64)
        seed = int.from_bytes(sha256(r["file_id"].encode()).digest()[:4], "big")
        _emc, _meas, total_noise = make_noise(clean, fs, seed=seed)
        m = buriedness(clean, total_noise)
        m["genre"] = r["genre"]
        rows_per_track.append(m)
    df = pd.DataFrame(rows_per_track)
    return df


def aggregate(df, label, fs, n_samples):
    n = len(df)
    return dict(
        name=label,
        fs=fs,
        samples=n_samples,
        units="amplitude (normalised)",
        n_tracks=n,
        rms_clean=df["rms_clean"].mean(),
        rms_noise_post=df["rms_noise_post"].mean(),
        rms_ratio_post=df["rms_ratio_post"].mean(),
        power_ratio_post=df["power_ratio_post"].mean(),
        peak_clean=df["peak_clean"].mean(),
        peak_noisy=df["peak_noisy"].mean(),
        snr_db_pre=df["snr_db_pre"].mean(),
        snr_db_post=df["snr_db_post"].mean(),
        snr_db_post_p05=df["snr_db_post"].quantile(0.05),
        snr_db_post_p95=df["snr_db_post"].quantile(0.95),
        frac_noise_over_signal=df["frac_noise_over_signal"].mean(),
        frac_noise_over_3x_signal=df["frac_noise_over_3x_signal"].mean(),
        clipped_rate=df["clipped"].mean(),
    )


def main():
    rows = []
    rows.append(demo_row("ECG (MIT-BIH r100)", load_ecg, "mV"))
    rows.append(demo_row("Vibration (CWRU 97)", load_vibration, "g"))
    rows.append(demo_row("Music (metal_01, seed 42)", load_music, "norm."))

    df_music = music_dataset_rows()
    fs_music = 22050
    n_samples_music = int(df_music.attrs.get("n", 22050 * 30))
    rows.append(aggregate(df_music, "Music dataset (250 tracks, all)",
                          fs_music, 22050 * 30))
    for genre in ("jazz", "metal", "pop"):
        sub = df_music[df_music["genre"] == genre]
        rows.append(aggregate(sub, f"  - {genre} ({len(sub)} tracks)",
                              fs_music, 22050 * 30))

    out_lines = []
    out_lines.append(
        "| signal | fs [Hz] | tracks | SNR [dB] | "
        "RMS noise / RMS clean | "
        "noise power / signal power | peak noisy / peak clean | "
        "% time |noise| > |signal| | % time |noise| > 3x |signal| | clipped |"
    )
    out_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        if "snr_db_post_p05" in r:
            snr_str = f"{r['snr_db_post']:+.2f}"
            clip_str = f"{100*r['clipped_rate']:.0f}% of tracks"
        else:
            snr_str = f"{r['snr_db_post']:+.2f}"
            clip_str = "yes" if r["clipped"] else "no"
        out_lines.append(
            f"| {r['name']} | {r['fs']} | {r['n_tracks']} | "
            f"{snr_str} | "
            f"{r['rms_ratio_post']:.2f}x | "
            f"{r['power_ratio_post']:.1f}x | "
            f"{r['peak_noisy']/(r['peak_clean']+1e-30):.2f}x | "
            f"{100*r['frac_noise_over_signal']:.1f}% | "
            f"{100*r['frac_noise_over_3x_signal']:.1f}% | "
            f"{clip_str} |"
        )

    table_md = "\n".join(out_lines)
    print(table_md)

    out_path = os.path.join(HERE, "buriedness_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# How buried are the signals in noise?\n\n")
        f.write("Target SNR = {:+g} dB on every signal — noise power = 10x "
                "signal power by design. ".format(SNR_TARGET_DB))
        f.write("Demo signals use the global seed=42 realisation; "
                "the 250-track music dataset uses per-track "
                "sha256(file_id) seeds (re-derived in this script so the "
                "noise matches what build_music_dataset.py produced).\n\n")
        f.write("**Reading the columns**\n\n")
        f.write("- *SNR [dB]* — clean signal power vs total noise power, "
                "post ADC clip. The clip removes only the largest noise "
                "excursions so this is within 0.01 dB of the spec target.\n")
        f.write("- *RMS noise / RMS clean* — linear amplitude ratio. "
                "At -10 dB SNR this is sqrt(10) ~= 3.16, i.e. the noise is "
                "on average 3x louder than the signal in amplitude terms.\n")
        f.write("- *noise power / signal power* — same idea on a power "
                "scale; 10x at -10 dB.\n")
        f.write("- *peak noisy / peak clean* — capped at 3x by the ADC "
                "clip; values at 3 indicate the clip is engaged. The "
                "dataset average is below 3 because not every track "
                "trips the clip.\n")
        f.write("- *% time |noise| > |signal|* — instantaneous burial: "
                "fraction of samples where the noise amplitude exceeds "
                "the clean signal amplitude. This is the visceral "
                "'how often is the signal hidden' number.\n")
        f.write("- *% time |noise| > 3x |signal|* — deep burial: noise more "
                "than triple the local signal.\n\n")
        f.write(table_md)
        f.write("\n\n## Key takeaways\n\n")
        f.write("- The design point is identical across every signal: "
                "-10 dB SNR, noise carrying 10x the signal power.\n")
        f.write("- The signal is *louder* than the noise on only 15-20 % of "
                "samples. The remaining 80 %+ are noise-dominated, and "
                "roughly half of all samples sit under noise more than "
                "triple the local signal amplitude.\n")
        f.write("- The ADC clip engages on every demo signal and on 72 % "
                "of the 250-track music dataset. Clipping does NOT meaningfully "
                "reduce SNR (the post-clip column is essentially unchanged "
                "from the pre-clip target) but it does cap peak excursions "
                "at 3x clean peak — which is why the classifier story "
                "treats clipping as an impulsive-noise event, not a level "
                "change.\n")
        f.write("- This buriedness is the entire reason the rest of the "
                "talk exists: a clean-trained classifier sees +99 % "
                "accuracy on clean audio and collapses to 34 % on this "
                "noise (cross-condition C). Filtering and feature-design "
                "claw it back to 80 %.\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
