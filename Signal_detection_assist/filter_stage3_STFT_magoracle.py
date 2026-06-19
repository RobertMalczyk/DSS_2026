"""Stage 3 Path-1 test: STFT-domain magnitude oracle.

    S_clean = STFT(clean)
    S_noisy = STFT(noisy)
    Y       = |S_clean| * exp(j * angle(S_noisy))
    y       = iSTFT(Y)

Because STFT+iSTFT with overlap-add (Hann, 75% overlap) reconstructs exactly,
|STFT(y)| == |S_clean| up to numerical error -> magnitude-based features
(MFCC, centroid, rolloff, bandwidth) are essentially the clean feature vector.
Phase is borrowed from the noisy signal (model ignores phase).

Compares against: clean, noisy, stage-3 Wiener output, and previous A+C attempt.
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import librosa

CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")
NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
WIENER_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered assist and WAVELET 3rd stage")
AC_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\stage3_AC_test")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\stage3_STFT_magoracle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FFT = 2048
HOP = 512       # 75% overlap, matches librosa defaults and our earlier STFT plots
WINDOW = "hann"
PREFIX = "filtered_assist_stft_magoracle_"


def stft(x):
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)


def istft(S, length):
    return librosa.istft(S, hop_length=HOP, win_length=N_FFT, window=WINDOW, length=length, center=True)


def magnitude_oracle(clean, noisy):
    n = min(len(clean), len(noisy))
    clean = clean[:n]; noisy = noisy[:n]
    Sc = stft(clean)
    Sn = stft(noisy)
    Y = np.abs(Sc) * np.exp(1j * np.angle(Sn))
    y = istft(Y, length=n)
    return y.astype(np.float32)


def match_rms_no_clip(y, rms_target, peak_cap=0.999):
    rms_y = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak_y = float(np.max(np.abs(y)))
    g = min(rms_target / (rms_y + 1e-20), peak_cap / (peak_y + 1e-20))
    return (y * g).astype(np.float32), g, g * rms_y < rms_target * 0.999


def extract_features(x, sr):
    x = x.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
    cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
    bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=x)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [cent.mean(), cent.std()],
        [roll.mean(), roll.std()],
        [bw.mean(), bw.std()],
        [zcr.mean(), zcr.std()],
    ])


def main():
    subset = [f"{g}_{i:02d}.wav" for g in ("jazz", "metal", "pop") for i in (1, 2, 3)]
    feats_clean, feats_noisy, feats_wiener, feats_ac, feats_mag = [], [], [], [], []
    t0 = time.time()
    for i, name in enumerate(subset, 1):
        clean, sr = sf.read(str(CLEAN_DIR / name), always_2d=False)
        noisy, _ = sf.read(str(NOISY_DIR / name), always_2d=False)
        clean = clean.astype(np.float32); noisy = noisy.astype(np.float32)
        wiener, _ = sf.read(str(WIENER_DIR / f"filtered_assist_wavelet_3rd_stage_{name}"), always_2d=False)
        wiener = wiener.astype(np.float32)
        ac, _ = sf.read(str(AC_DIR / f"filtered_assist_wavelet_3rd_stage_AC_{name}"), always_2d=False)
        ac = ac.astype(np.float32)

        y = magnitude_oracle(clean, noisy)
        rms_c = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
        y, gain, capped = match_rms_no_clip(y, rms_c)
        info = sf.info(str(NOISY_DIR / name))
        sf.write(str(OUT_DIR / f"{PREFIX}{name}"), y, sr, subtype=info.subtype)

        feats_clean.append(extract_features(clean, sr))
        feats_noisy.append(extract_features(noisy, sr))
        feats_wiener.append(extract_features(wiener, sr))
        feats_ac.append(extract_features(ac, sr))
        feats_mag.append(extract_features(y, sr))
        cap = " (cap)" if capped else ""
        print(f"  [{i}/{len(subset)}] {name:<13s}  mag-oracle gain x{gain:.3f}{cap}  t={time.time()-t0:.1f}s")

    A = np.array(feats_clean); N = np.array(feats_noisy); W = np.array(feats_wiener)
    C = np.array(feats_ac); M = np.array(feats_mag)
    scale = A.std(axis=0) + 1e-6

    def dist(f):
        return np.array([np.linalg.norm((f[i] - A[i]) / scale) for i in range(len(subset))])

    dn, dw, dc, dm = dist(N), dist(W), dist(C), dist(M)
    print(f"\nPer-file normalized feature-vector L2 distance to CLEAN:")
    print(f"{'file':<13s} {'noisy':>10s} {'Wiener':>10s} {'A+C':>10s} {'mag-oracle':>12s}")
    for i, name in enumerate(subset):
        print(f"{name:<13s} {dn[i]:10.2f} {dw[i]:10.2f} {dc[i]:10.2f} {dm[i]:12.2f}")
    print(f"{'mean':<13s} {dn.mean():10.2f} {dw.mean():10.2f} {dc.mean():10.2f} {dm.mean():12.2f}")

    blocks = {
        "MFCC mean (20)": slice(0, 20),
        "MFCC std  (20)": slice(20, 40),
        "centroid (2)":   slice(40, 42),
        "rolloff  (2)":   slice(42, 44),
        "bandwidth(2)":   slice(44, 46),
        "ZCR      (2)":   slice(46, 48),
    }
    print(f"\nPer-block mean normalized L2 to clean:")
    print(f"{'block':<18s} {'noisy':>8s} {'Wiener':>8s} {'A+C':>8s} {'mag-oracle':>12s}")
    for bname, sl in blocks.items():
        bs = scale[sl]
        def bd(F):
            return np.mean([np.linalg.norm((F[i][sl] - A[i][sl]) / bs) for i in range(len(subset))])
        print(f"{bname:<18s} {bd(N):8.2f} {bd(W):8.2f} {bd(C):8.2f} {bd(M):12.2f}")

    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
