"""Stage 5 filter: spectral subtraction with multi-file noise profile (non-oracle).

Phase 1 - build noise profile (once):
  Sample K noisy files. For each, compute time-averaged |STFT|^2 (per-file PSD),
  normalize each to unit sum, then take the cross-file MEDIAN.
  This captures the shared stationary noise shape; music content (different per
  file) averages out. Never touches the clean folder.

Phase 2 - per file:
  Per-file noise level L_i = median(p20_per_bin / N_shape); N_i(f) = L_i * N_shape.
  Classic spectral subtraction:
      |Y(f,t)|^2 = max(|X(f,t)|^2 - alpha * N_i(f), gamma * |X(f,t)|^2)
      Y = |Y| * exp(j * angle(X))
      y = iSTFT(Y)
  Clean signal is used ONLY for evaluation (SNR, feature distance).
"""
from pathlib import Path
import random
import time
import numpy as np
import soundfile as sf
import librosa

NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")    # EVAL-ONLY
OUT_DIR   = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered SpecSub 5th stage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "filtered_specsub_5th_stage_"

N_FFT = 2048
HOP = 512
WINDOW = "hann"

PROFILE_N = 50          # files pooled for the noise profile
PROFILE_SEED = 42
LOW_PERCENTILE = 20.0   # per-bin, over time
ALPHA_OVER = 1.5        # over-subtraction factor
SPEC_FLOOR = 0.05       # keep at least 5% of |X|^2 at each bin (prevents musical noise holes)


def stft(x):
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)

def istft(S, length):
    return librosa.istft(S, hop_length=HOP, win_length=N_FFT, window=WINDOW, length=length, center=True)


def load_mono(path: Path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def build_noise_profile(files):
    profiles = []
    for p in files:
        x, _ = load_mono(p)
        X = stft(x)
        P = (np.abs(X) ** 2).mean(axis=1)   # time-averaged PSD
        P = P / (P.sum() + 1e-20)           # unit-sum normalization
        profiles.append(P)
    return np.median(np.stack(profiles, axis=0), axis=0)


def apply_specsub(x, N_shape):
    X = stft(x)
    PX = np.abs(X) ** 2
    low = np.percentile(PX, LOW_PERCENTILE, axis=1)
    mask = N_shape > 1e-20
    L = float(np.median(low[mask] / N_shape[mask]))
    N_i = L * N_shape
    PY = np.maximum(PX - ALPHA_OVER * N_i[:, None], SPEC_FLOOR * PX)
    Y = np.sqrt(PY) * np.exp(1j * np.angle(X))
    y = istft(Y, length=len(x))
    return y.astype(np.float32), L


def snr_db(ref, est):
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64); est = est[:n].astype(np.float64)
    return 10.0 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-20))


def extract_features(x, sr):
    x = x.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
    cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)
    bw   = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    zcr  = librosa.feature.zero_crossing_rate(y=x)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [cent.mean(), cent.std()],
        [roll.mean(), roll.std()],
        [bw.mean(), bw.std()],
        [zcr.mean(), zcr.std()],
    ])


def main():
    all_noisy = sorted(NOISY_DIR.glob("*.wav"))
    rng = random.Random(PROFILE_SEED)
    prof_files = rng.sample(all_noisy, k=min(PROFILE_N, len(all_noisy)))
    print(f"Noise profile: pooling {len(prof_files)} noisy files (seed={PROFILE_SEED}).")
    t0 = time.time()
    N_shape = build_noise_profile(prof_files)
    print(f"  profile built in {time.time()-t0:.1f}s  (shape len {len(N_shape)})")

    # Save the profile so it can be reused for the full 250-file batch.
    prof_path = OUT_DIR / "noise_profile.npy"
    np.save(prof_path, N_shape)
    print(f"  saved -> {prof_path.name}")

    subset = [f"{g}_{i:02d}.wav" for g in ("jazz", "metal", "pop") for i in (1, 2, 3)]

    feats_clean = []; feats_noisy = []; feats_filt = []
    snr_in = []; snr_out = []
    t0 = time.time()
    for i, name in enumerate(subset, 1):
        noisy, sr = load_mono(NOISY_DIR / name)
        y, L = apply_specsub(noisy, N_shape)
        peak = float(np.max(np.abs(y)))
        if peak > 0.999:
            y = (y * (0.999 / peak)).astype(np.float32)
        info = sf.info(str(NOISY_DIR / name))
        out = OUT_DIR / f"{PREFIX}{name}"
        sf.write(str(out), y, sr, subtype=info.subtype)

        # EVAL ONLY - clean used for metrics, not for filtering.
        clean, _ = load_mono(CLEAN_DIR / name)
        snr_in.append(snr_db(clean, noisy))
        snr_out.append(snr_db(clean, y))
        feats_clean.append(extract_features(clean, sr))
        feats_noisy.append(extract_features(noisy, sr))
        feats_filt.append(extract_features(y, sr))

        print(f"  [{i}/{len(subset)}] {out.name:<45s} L={L:.3e}  "
              f"SNR {snr_in[-1]:+6.2f} -> {snr_out[-1]:+6.2f} dB")

    A = np.array(feats_clean); N = np.array(feats_noisy); F = np.array(feats_filt)
    scale = A.std(axis=0) + 1e-6
    def dist(f):
        return np.array([np.linalg.norm((f[i] - A[i]) / scale) for i in range(len(subset))])
    dn = dist(N); df = dist(F)
    print(f"\nFeature-vector L2 distance to clean (eval only):")
    print(f"{'file':<13s} {'noisy':>10s} {'specsub':>10s}")
    for i, name in enumerate(subset):
        print(f"{name:<13s} {dn[i]:10.2f} {df[i]:10.2f}")
    print(f"{'mean':<13s} {dn.mean():10.2f} {df.mean():10.2f}")

    blocks = {
        "MFCC mean (20)": slice(0, 20),
        "MFCC std  (20)": slice(20, 40),
        "centroid (2)":   slice(40, 42),
        "rolloff  (2)":   slice(42, 44),
        "bandwidth(2)":   slice(44, 46),
        "ZCR      (2)":   slice(46, 48),
    }
    print(f"\nPer-block mean normalized L2 to clean:")
    print(f"{'block':<18s} {'noisy':>8s} {'specsub':>8s}")
    for bname, sl in blocks.items():
        bs = scale[sl]
        def bd(X):
            return np.mean([np.linalg.norm((X[i][sl] - A[i][sl]) / bs) for i in range(len(subset))])
        print(f"{bname:<18s} {bd(N):8.2f} {bd(F):8.2f}")

    print(f"\nSNR summary over {len(snr_in)} files:")
    print(f"  mean SNR (noisy)    = {np.mean(snr_in):+6.2f} dB")
    print(f"  mean SNR (specsub)  = {np.mean(snr_out):+6.2f} dB")
    print(f"  mean improvement    = {np.mean(np.array(snr_out)-np.array(snr_in)):+6.2f} dB")
    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
