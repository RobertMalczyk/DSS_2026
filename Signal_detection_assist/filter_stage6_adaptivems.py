"""Stage 6 filter: adaptive Wiener with Minimum-Statistics noise tracking
    + decision-directed a-priori SNR smoothing.

Non-oracle: uses only the noisy input and the cross-file noise profile
(bootstrap init) saved in stage 5.

Per-frame pipeline:
  1. STFT -> X(f,t);  P(f,t) = |X|^2
  2. Smoothed power P_sm(f,t) = alpha_p * P_sm(f,t-1) + (1-alpha_p) * P(f,t)
  3. Minimum-statistics noise PSD:
         lambda_d(f,t) = B_min * min over last D frames of P_sm
     (Martin 2001 simplified: sliding window + bias compensation)
  4. A-posteriori SNR gamma = P / lambda_d
  5. A-priori SNR (decision-directed, Ephraim-Malah):
         xi = alpha_dd * |S_prev|^2 / lambda_d_prev + (1-alpha_dd) * max(gamma-1, 0)
  6. Wiener gain G = xi / (1 + xi), floored
  7. Y(f,t) = G(f,t) * X(f,t);  iSTFT
Clean signal is EVAL-ONLY (SNR, feature distance).
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import librosa

NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")      # EVAL-ONLY
BOOTSTRAP_PROFILE = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered SpecSub 5th stage\noise_profile.npy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered AdaptiveMS 6th stage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "filtered_adaptivems_6th_stage_"

N_FFT = 2048
HOP = 512
WINDOW = "hann"

ALPHA_P = 0.85        # power smoothing
ALPHA_DD = 0.98       # decision-directed SNR smoothing (Ephraim-Malah)
D_FRAMES = 96         # min-statistics sliding window (~2.2 s @ hop=512, sr=22050)
B_MIN = 1.3           # bias compensation for minimum (retuned from 1.5)
XI_MIN = 1e-2         # a-priori SNR floor (~-20 dB, retuned from 1e-3 => min gain ~0.1)
GAIN_MIN = np.sqrt(XI_MIN / (1 + XI_MIN))  # equivalent floor on |Y|/|X|


def stft(x):
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window=WINDOW, center=True)

def istft(S, length):
    return librosa.istft(S, hop_length=HOP, win_length=N_FFT, window=WINDOW, length=length, center=True)


def load_mono(path: Path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def bootstrap_noise_psd(P: np.ndarray, N_shape: np.ndarray) -> np.ndarray:
    # Initial lambda_d estimate: per-file scale of the cross-file noise shape,
    # matched to the 20th percentile of P per freq bin.
    low = np.percentile(P, 20, axis=1)
    mask = N_shape > 1e-20
    L = float(np.median(low[mask] / N_shape[mask]))
    return L * N_shape


def adaptive_wiener(x: np.ndarray, N_shape: np.ndarray):
    X = stft(x)
    P = (np.abs(X) ** 2).astype(np.float64)
    F, T = P.shape

    lam0 = bootstrap_noise_psd(P, N_shape).astype(np.float64)  # (F,)
    # Pre-fill the min-statistics circular buffer with lam0 / B_min so that
    # min * B_min = lam0 until enough real frames have been observed.
    mins_buf = np.full((F, D_FRAMES), lam0[:, None] / B_MIN)
    P_sm = P[:, 0].copy()

    lam = np.empty_like(P)
    lam[:, 0] = lam0
    G = np.empty_like(P)
    S_sq_prev = np.maximum(P[:, 0] - lam0, 0.0)
    lam_prev = lam0.copy()

    # First frame gain
    xi0 = np.maximum(S_sq_prev / (lam_prev + 1e-20), XI_MIN)
    G[:, 0] = xi0 / (1 + xi0)

    for t in range(1, T):
        P_sm = ALPHA_P * P_sm + (1 - ALPHA_P) * P[:, t]
        mins_buf[:, t % D_FRAMES] = P_sm
        P_min = mins_buf.min(axis=1)
        lam_t = B_MIN * P_min
        lam[:, t] = lam_t

        gamma_t = P[:, t] / (lam_t + 1e-20)
        xi = ALPHA_DD * (S_sq_prev / (lam_prev + 1e-20)) + (1 - ALPHA_DD) * np.maximum(gamma_t - 1.0, 0.0)
        xi = np.maximum(xi, XI_MIN)
        Gt = xi / (1 + xi)
        G[:, t] = Gt

        # Update for next DD step
        S_sq_prev = (Gt ** 2) * P[:, t]
        lam_prev = lam_t

    G = np.maximum(G, GAIN_MIN)
    Y = G.astype(np.complex64) * X
    y = istft(Y, length=len(x))
    return y.astype(np.float32), G.astype(np.float32), lam.astype(np.float32)


def snr_db(ref, est):
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64); est = est[:n].astype(np.float64)
    return 10.0 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-20))


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
    assert BOOTSTRAP_PROFILE.exists(), f"bootstrap profile missing: {BOOTSTRAP_PROFILE}"
    N_shape = np.load(BOOTSTRAP_PROFILE)
    print(f"Loaded bootstrap noise profile: {BOOTSTRAP_PROFILE.name} (len {len(N_shape)})")
    print(f"Params: alpha_p={ALPHA_P}, alpha_dd={ALPHA_DD}, D={D_FRAMES} frames "
          f"(~{D_FRAMES*HOP/22050:.2f}s), B_min={B_MIN}, xi_min={XI_MIN}")

    subset = [f"{g}_{i:02d}.wav" for g in ("jazz", "metal", "pop") for i in (1, 2, 3)]

    feats_clean = []; feats_noisy = []; feats_filt = []
    snr_in = []; snr_out = []
    t0 = time.time()
    for i, name in enumerate(subset, 1):
        noisy, sr = load_mono(NOISY_DIR / name)
        y, G, lam = adaptive_wiener(noisy, N_shape)
        peak = float(np.max(np.abs(y)))
        if peak > 0.999:
            y = (y * (0.999 / peak)).astype(np.float32)

        info = sf.info(str(NOISY_DIR / name))
        out = OUT_DIR / f"{PREFIX}{name}"
        sf.write(str(out), y, sr, subtype=info.subtype)

        # EVAL ONLY (clean for metrics, not for filtering)
        clean, _ = load_mono(CLEAN_DIR / name)
        snr_in.append(snr_db(clean, noisy))
        snr_out.append(snr_db(clean, y))
        feats_clean.append(extract_features(clean, sr))
        feats_noisy.append(extract_features(noisy, sr))
        feats_filt.append(extract_features(y, sr))

        g_stats = f"G median={np.median(G):.3f}, mean={G.mean():.3f}"
        print(f"  [{i}/{len(subset)}] {out.name:<50s}  "
              f"SNR {snr_in[-1]:+6.2f} -> {snr_out[-1]:+6.2f} dB  {g_stats}")

    A = np.array(feats_clean); N = np.array(feats_noisy); F = np.array(feats_filt)
    scale = A.std(axis=0) + 1e-6
    def dist(f):
        return np.array([np.linalg.norm((f[i] - A[i]) / scale) for i in range(len(subset))])
    dn = dist(N); df = dist(F)
    print(f"\nFeature distance to clean (eval only):")
    print(f"{'file':<13s} {'noisy':>10s} {'adaptiveMS':>12s}")
    for i, name in enumerate(subset):
        print(f"{name:<13s} {dn[i]:10.2f} {df[i]:12.2f}")
    print(f"{'mean':<13s} {dn.mean():10.2f} {df.mean():12.2f}")

    blocks = {
        "MFCC mean (20)": slice(0, 20),
        "MFCC std  (20)": slice(20, 40),
        "centroid (2)":   slice(40, 42),
        "rolloff  (2)":   slice(42, 44),
        "bandwidth(2)":   slice(44, 46),
        "ZCR      (2)":   slice(46, 48),
    }
    print(f"\nPer-block mean normalized L2 to clean:")
    print(f"{'block':<18s} {'noisy':>8s} {'adaptiveMS':>12s}")
    for bname, sl in blocks.items():
        bs = scale[sl]
        def bd(X):
            return np.mean([np.linalg.norm((X[i][sl] - A[i][sl]) / bs) for i in range(len(subset))])
        print(f"{bname:<18s} {bd(N):8.2f} {bd(F):12.2f}")

    print(f"\nSNR summary over {len(snr_in)} files:")
    print(f"  mean SNR (noisy)       = {np.mean(snr_in):+6.2f} dB")
    print(f"  mean SNR (adaptiveMS)  = {np.mean(snr_out):+6.2f} dB")
    print(f"  mean improvement       = {np.mean(np.array(snr_out)-np.array(snr_in)):+6.2f} dB")
    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
