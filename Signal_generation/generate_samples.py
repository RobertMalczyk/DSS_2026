"""
Emit the three noisy test signals (ECG, vibration, music) as standalone files
under out/Samples/ for downstream consumers (e.g. Signal_detection_RAW).

Each signal is saved as 32-bit float WAV at its native sample rate — WAV is
the only format shared by every tool in the pipeline, and 32-bit float
preserves the exact post-ADC-clip floating-point values without requantising.

The matching clean signal for each sample is also written to
out/Samples/Clean/{tag}_clean.wav. This subfolder is **off-limits** to the
downstream Signal_detection_RAW project — it must not read anything from it.
The clean files exist for the user's own inspection and for any training
workflows that need ground truth.

Output layout:
  out/Samples/
    {tag}_noisy.wav        <- primary input for downstream (noisy-only)
    samples_info.txt       <- minimal metadata (fs / duration / units / source)
    Clean/
      {tag}_clean.wav      <- off-limits to downstream
      metrics.txt          <- full numeric metrics (RMS / SNR / peak / clip)

Nothing about the noise model, RMS values, or achieved SNR is written to the
outer folder — everything numeric about the clean-vs-noisy relationship sits
inside Clean/ so the downstream "noisy-only" contract is enforced by the
directory boundary.
"""

import os
import numpy as np
import soundfile as sf

from signal_contamination import (
    load_ecg, load_vibration, load_music,
    make_noise, SNR_TARGET_DB,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_SAMPLES = os.path.join(HERE, "out", "Samples")
OUT_CLEAN = os.path.join(OUT_SAMPLES, "Clean")
os.makedirs(OUT_SAMPLES, exist_ok=True)
os.makedirs(OUT_CLEAN, exist_ok=True)


def build_noisy(clean, fs):
    _, _, total = make_noise(clean, fs)
    noisy = clean + total
    full_scale_adc = 3.0 * (float(np.max(np.abs(clean))) + 1e-12)
    pre_clip_peak = float(np.max(np.abs(noisy)))
    was_clipped = pre_clip_peak > full_scale_adc
    noisy = np.clip(noisy, -full_scale_adc, full_scale_adc)
    snr_db = 10 * np.log10(np.mean(clean ** 2) / np.mean(total ** 2))
    return noisy, total, float(snr_db), was_clipped


def main():
    loaders = [
        ("ecg",       load_ecg),
        ("vibration", load_vibration),
        ("music",     load_music),
    ]
    # Public info file — deliberately minimal. No source dataset, no noise-
    # model info, no clean RMS / noise RMS / SNR / clip flag. Anything a
    # blind downstream reader could use to reconstruct ground truth or infer
    # the noise composition stays out.
    public_lines = [
        "Signal samples (out/Samples/)",
        "",
        "Format : 32-bit float WAV, native sample rate, no normalisation.",
        "",
        "=" * 74,
    ]
    # Private metrics file — stays inside Clean/, which downstream will not read.
    private_lines = [
        "Full clean / noise metrics  (out/Samples/Clean/metrics.txt)",
        "",
        f"Target SNR : {SNR_TARGET_DB:+.1f} dB  (identical on every signal)",
        "Noise model: see ../noise.md",
        "",
        "This file is intentionally kept inside Clean/ so the downstream",
        "Signal_detection_RAW project never sees it — denoising must work",
        "without knowing the clean RMS, the noise RMS, the achieved SNR,",
        "or whether any signal was clipped.",
        "",
        "=" * 74,
    ]

    for tag, fn in loaders:
        print(f"[{tag}] loading...")
        clean, fs, source, units = fn()
        print(f"[{tag}] loaded {clean.size} samples @ {fs} Hz")
        noisy, total, snr_db, clipped = build_noisy(clean, fs)

        noisy_path = os.path.join(OUT_SAMPLES, f"{tag}_noisy.wav")
        clean_path = os.path.join(OUT_CLEAN, f"{tag}_clean.wav")
        sf.write(noisy_path, noisy.astype(np.float32), fs, subtype="FLOAT")
        sf.write(clean_path, clean.astype(np.float32), fs, subtype="FLOAT")
        noisy_kb = os.path.getsize(noisy_path) / 1024
        clean_kb = os.path.getsize(clean_path) / 1024

        rms_clean = float(np.sqrt(np.mean(clean ** 2)))
        rms_noise = float(np.sqrt(np.mean(total ** 2)))
        peak_noisy = float(np.max(np.abs(noisy)))
        peak_clean = float(np.max(np.abs(clean)))

        public_lines += [
            f"file          : Samples/{tag}_noisy.wav   ({noisy_kb:.1f} KB)",
            f"sample rate   : {fs} Hz",
            f"samples       : {clean.size}",
            f"duration      : {clean.size / fs:.3f} s",
            "=" * 74,
        ]

        private_lines += [
            f"tag           : {tag}",
            f"clean file    : Samples/Clean/{tag}_clean.wav   ({clean_kb:.1f} KB)",
            f"sample rate   : {fs} Hz",
            f"samples       : {clean.size}",
            f"duration      : {clean.size / fs:.3f} s",
            f"units         : {units}",
            f"achieved SNR  : {snr_db:+.2f} dB",
            f"RMS clean     : {rms_clean:.4g} {units}",
            f"RMS noise     : {rms_noise:.4g} {units}",
            f"peak clean    : {peak_clean:.4g} {units}",
            f"peak noisy    : {peak_noisy:.4g} {units} (post ADC clip)",
            f"clipped       : {'YES' if clipped else 'no'}",
            "=" * 74,
        ]

        print(f"[{tag}] wrote {noisy_path}  "
              f"(SNR {snr_db:+.2f} dB, clipped={clipped}, {noisy_kb:.1f} KB)")
        print(f"[{tag}] wrote {clean_path}  ({clean_kb:.1f} KB)")

    public_path  = os.path.join(OUT_SAMPLES, "samples_info.txt")
    private_path = os.path.join(OUT_CLEAN,   "metrics.txt")
    with open(public_path, "w", encoding="utf-8") as f:
        f.write("\n".join(public_lines) + "\n")
    with open(private_path, "w", encoding="utf-8") as f:
        f.write("\n".join(private_lines) + "\n")
    print(f"\nPublic metadata -> {public_path}")
    print(f"Clean metrics   -> {private_path}")


if __name__ == "__main__":
    main()
