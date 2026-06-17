"""Stage 3 variant A+C: magnitude-oracle reconstruction + mel-band frequency weighting.

Option A (magnitude oracle):
    C_filt = |C_clean| * exp(j * angle(C_noisy))
  -> magnitude-based features (MFCC, centroid, rolloff, bandwidth) match clean
     because |output| ~ |clean|. Phase is borrowed from noisy (irrelevant to those features).

Option C (mel-band weighting):
    g(f) cosine-tapered bandpass shaping on top, emphasizing the model-relevant band.
      - stop below  40 Hz  (sub-rumble)
      - taper 40 -> 80 Hz
      - flat  80 -> 6000 Hz   (MFCC / centroid / rolloff / bandwidth core)
      - taper 6000 -> 9000 Hz
      - stop above 9000 Hz  (mostly ZCR-only content; cleaning it harder avoids
                             noise leakage through phase even though |C_clean| is
                             already clean in magnitude)

Also computes librosa features for {clean, noisy, current-Wiener stage3, A+C}
and reports feature-vector L2 distance to clean.
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import librosa
from ssqueezepy import cwt, icwt, Wavelet

CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")
NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
WIENER_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered assist and WAVELET 3rd stage")
AC_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\stage3_AC_test")
AC_DIR.mkdir(parents=True, exist_ok=True)

MU = 6
NV = 32
F_LO_STOP = 40.0
F_LO_PASS = 80.0
F_HI_PASS = 6000.0
F_HI_STOP = 9000.0
PREFIX = "filtered_assist_wavelet_3rd_stage_AC_"


def freq_weight_hz(freqs):
    w = np.zeros_like(freqs, dtype=np.float64)
    for i, f in enumerate(freqs):
        if f <= F_LO_STOP or f >= F_HI_STOP:
            w[i] = 0.0
        elif F_LO_STOP < f < F_LO_PASS:
            w[i] = 0.5 * (1 - np.cos(np.pi * (f - F_LO_STOP) / (F_LO_PASS - F_LO_STOP)))
        elif F_LO_PASS <= f <= F_HI_PASS:
            w[i] = 1.0
        else:
            w[i] = 0.5 * (1 + np.cos(np.pi * (f - F_HI_PASS) / (F_HI_STOP - F_HI_PASS)))
    return w


def scale_to_freq_hz(scales, sr, mu):
    # Morlet peak frequency in Hz: fc * fs / scale, with fc = mu / (2*pi).
    return (mu / (2.0 * np.pi)) * sr / scales


def apply_AC(clean, noisy, sr, wav):
    C_clean, scales = cwt(clean, wavelet=wav, fs=sr, nv=NV)
    C_noisy, _ = cwt(noisy, wavelet=wav, fs=sr, nv=NV)
    freqs = scale_to_freq_hz(scales, sr, MU)
    g = freq_weight_hz(freqs)[:, None]                   # (S, 1)
    C_filt = (g * np.abs(C_clean)) * np.exp(1j * np.angle(C_noisy))
    y = icwt(C_filt, wavelet=wav, scales=scales, nv=NV)
    return np.real(y).astype(np.float32)[: len(clean)], freqs, g.ravel()


def match_rms_no_clip(y, rms_target, peak_cap=0.999):
    rms_y = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak_y = float(np.max(np.abs(y)))
    g_rms = rms_target / (rms_y + 1e-20)
    g_clip = peak_cap / (peak_y + 1e-20)
    g = min(g_rms, g_clip)
    return (y * g).astype(np.float32), g, g < g_rms


def extract_features(x, sr):
    # Match the model spec: MFCC 20 coef (mean + std), centroid, rolloff (85%),
    # bandwidth, zero-crossing rate. Return one flat feature vector.
    x = x.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
    cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
    bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=x)
    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),          # 40
        [cent.mean(), cent.std()],                    # 2
        [roll.mean(), roll.std()],                    # 2
        [bw.mean(), bw.std()],                        # 2
        [zcr.mean(), zcr.std()],                      # 2
    ])
    return feats


def normalized_l2(feat, ref, scale):
    return float(np.linalg.norm((feat - ref) / (scale + 1e-12)))


def main():
    subset = [f"{g}_{i:02d}.wav" for g in ("jazz", "metal", "pop") for i in (1, 2, 3)]
    wav = Wavelet(("morlet", {"mu": MU}))

    # Freq-weighting preview (one sample rate, 22050)
    dummy_scales = np.linspace(1, 1000, 128)
    prev_freqs = scale_to_freq_hz(dummy_scales, 22050, MU)
    prev_w = freq_weight_hz(prev_freqs)
    print(f"freq-weighting preview: passband {F_LO_PASS:.0f}-{F_HI_PASS:.0f} Hz, "
          f"taper {F_LO_STOP:.0f}-{F_LO_PASS:.0f}, {F_HI_PASS:.0f}-{F_HI_STOP:.0f}; "
          f"max g={prev_w.max():.2f}")
    print(f"Output -> {AC_DIR}\n")

    # First pass: run A+C and collect features for all 4 variants
    feats_clean = []
    feats_noisy = []
    feats_wiener = []
    feats_ac = []
    t0 = time.time()
    for i, name in enumerate(subset, 1):
        clean, sr = sf.read(str(CLEAN_DIR / name), always_2d=False)
        noisy, _ = sf.read(str(NOISY_DIR / name), always_2d=False)
        clean = clean.astype(np.float32)
        noisy = noisy.astype(np.float32)
        wiener, _ = sf.read(str(WIENER_DIR / f"filtered_assist_wavelet_3rd_stage_{name}"), always_2d=False)
        wiener = wiener.astype(np.float32)

        y_ac, _, _ = apply_AC(clean, noisy, sr, wav)
        # RMS-match to clean, cap to avoid clipping
        rms_c = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
        y_ac, gain, capped = match_rms_no_clip(y_ac, rms_c)
        info = sf.info(str(NOISY_DIR / name))
        out = AC_DIR / f"{PREFIX}{name}"
        sf.write(str(out), y_ac, sr, subtype=info.subtype)

        fc = extract_features(clean, sr)
        fn = extract_features(noisy, sr)
        fw = extract_features(wiener, sr)
        fa = extract_features(y_ac, sr)
        feats_clean.append(fc); feats_noisy.append(fn); feats_wiener.append(fw); feats_ac.append(fa)
        cap = " (cap)" if capped else ""
        print(f"  [{i}/{len(subset)}] {name:<13s}  A+C gain x{gain:.3f}{cap}  t={time.time()-t0:.1f}s")

    feats_clean = np.array(feats_clean)
    feats_noisy = np.array(feats_noisy)
    feats_wiener = np.array(feats_wiener)
    feats_ac = np.array(feats_ac)

    # Feature-wise scale for fair comparison (use std over clean set).
    scale = feats_clean.std(axis=0) + 1e-6

    # Per-file normalized L2 distance to clean.
    print(f"\n{'file':<13s} {'noisy':>10s} {'Wiener':>10s} {'A+C':>10s}")
    for i, name in enumerate(subset):
        dn = normalized_l2(feats_noisy[i], feats_clean[i], scale)
        dw = normalized_l2(feats_wiener[i], feats_clean[i], scale)
        da = normalized_l2(feats_ac[i], feats_clean[i], scale)
        print(f"{name:<13s} {dn:10.2f} {dw:10.2f} {da:10.2f}")
    print(f"{'mean':<13s} "
          f"{np.mean([normalized_l2(feats_noisy[i], feats_clean[i], scale) for i in range(len(subset))]):10.2f} "
          f"{np.mean([normalized_l2(feats_wiener[i], feats_clean[i], scale) for i in range(len(subset))]):10.2f} "
          f"{np.mean([normalized_l2(feats_ac[i], feats_clean[i], scale) for i in range(len(subset))]):10.2f}")

    # Sub-group distances: which feature block each method fixes / hurts
    blocks = {
        "MFCC mean (20)": slice(0, 20),
        "MFCC std  (20)": slice(20, 40),
        "centroid (2)":   slice(40, 42),
        "rolloff  (2)":   slice(42, 44),
        "bandwidth(2)":   slice(44, 46),
        "ZCR      (2)":   slice(46, 48),
    }
    print(f"\nMean normalized L2 per feature block:")
    print(f"{'block':<18s} {'noisy':>8s} {'Wiener':>8s} {'A+C':>8s}")
    for name, sl in blocks.items():
        bs = scale[sl]
        dn = np.mean([np.linalg.norm((feats_noisy[i][sl] - feats_clean[i][sl]) / bs) for i in range(len(subset))])
        dw = np.mean([np.linalg.norm((feats_wiener[i][sl] - feats_clean[i][sl]) / bs) for i in range(len(subset))])
        da = np.mean([np.linalg.norm((feats_ac[i][sl] - feats_clean[i][sl]) / bs) for i in range(len(subset))])
        print(f"{name:<18s} {dn:8.2f} {dw:8.2f} {da:8.2f}")

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
