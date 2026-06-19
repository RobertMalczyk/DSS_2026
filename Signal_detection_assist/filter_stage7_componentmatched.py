"""Stage 7 filter: component-matched chain.

Non-oracle — uses only noisy-side information. The clean signal is read
ONLY for evaluation metrics.

Pipeline (time-domain cascade, then spectral Wiener):

  1. Zero-phase 2nd-order Butterworth high-pass at 25 Hz
       -> kills the 1/f^2 drift without touching musical content.
  2. Mains-harmonic comb: four IIR notches at the DATA-DETECTED
     harmonics of the 50 Hz mains fundamental (typically 51.1, 99.6,
     150.7, 199.2 Hz), Q = 35 (~1.4 Hz -3 dB width).
  3. Switching-tone notch centred on the mean of the two FM sidebands
     detected near 4410 Hz; Q chosen so the -3 dB width spans both.
  4. Adaptive Wiener (Minimum-Statistics + decision-directed) with the
     tuned parameters from feedback_adaptive_wiener_tuning.md:
       alpha_p=0.85, alpha_dd=0.98, D=96 frames, B_min=1.3, xi_min=1e-2.

Notch frequencies are RE-DISCOVERED inside this script from the noisy
dataset alone (low-energy-frame pooled PSD -> find_peaks), so no
spec constants drive the coefficients.
"""
from __future__ import annotations

import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, iirnotch, sosfiltfilt, tf2sos

NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")  # EVAL-ONLY
BOOTSTRAP_PROFILE = Path(
    r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered SpecSub 5th stage\noise_profile.npy"
)
OUT_DIR = Path(
    r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered ComponentMatched 7th stage"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path(
    r"C:\Robak\DSS2026\Claude\Signal_detection_assist\Out\COMPONENT_MATCHED_v1"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "filtered_componentmatched_7th_stage_"

FS_EXPECTED = 22050

# ----- stage A: high-pass -----
HP_CUTOFF_HZ = 25.0
HP_ORDER = 2

# ----- stage B/C: notch Q factors -----
Q_MAINS = 35.0           # narrow — mains lines are ~2 Hz wide
Q_SWITCHING = 150.0      # wider — span FM sidebands

# ----- stage D: Wiener (memory-validated sweet spot) -----
N_FFT = 2048
HOP = 512
ALPHA_P = 0.85
ALPHA_DD = 0.98
D_FRAMES = 96
B_MIN = 1.3
XI_MIN = 1e-2
GAIN_MIN = float(np.sqrt(XI_MIN / (1 + XI_MIN)))

# ----- dataset iteration -----
STARTER_FILES = [
    f"{g}_{i:02d}.wav" for g in ("jazz", "metal", "pop") for i in (1, 2, 3)
]


# =========================================================================== #
# 1. Blind detection of narrowband noise lines                                #
# =========================================================================== #
def low_energy_psd(x: np.ndarray, fs: int,
                   frame_len: int = 8192, hop: int = 4096,
                   pct: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean |STFT|^2 over frames with energy in the bottom `pct` percent."""
    n = len(x)
    n_frames = 1 + (n - frame_len) // hop
    win = np.hanning(frame_len)
    energies = np.empty(n_frames)
    mags = np.empty((n_frames, frame_len // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame_len] * win
        mags[i] = np.abs(np.fft.rfft(seg, n=frame_len)) ** 2
        energies[i] = seg @ seg
    sel = energies <= np.percentile(energies, pct)
    if sel.sum() < 5:
        sel = np.zeros_like(energies, dtype=bool)
        sel[np.argsort(energies)[:5]] = True
    psd = mags[sel].mean(axis=0)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / fs)
    return freqs, psd


def detect_noise_lines(fs: int, n_files_per_genre: int = 8) -> dict:
    """Pool low-energy PSDs across 8 files/genre, find prominent narrowband
    lines, return the mains-comb + switching-tone frequencies.

    Purely non-oracle: reads only noisy/*.wav.
    """
    psds = []
    for g in ("jazz", "metal", "pop"):
        for p in sorted(NOISY_DIR.glob(f"{g}_*.wav"))[:n_files_per_genre]:
            x, sr = sf.read(str(p))
            if x.ndim > 1:
                x = x.mean(axis=1)
            assert sr == fs
            _, psd = low_energy_psd(x.astype(np.float64), sr)
            psds.append(psd)
    stack = np.stack(psds)
    stack = stack / stack.sum(axis=1, keepdims=True)
    med = np.median(stack, axis=0)
    med_db = 10 * np.log10(med + 1e-18)
    freqs = np.fft.rfftfreq(8192, d=1.0 / fs)

    continuum = np.convolve(med_db, np.ones(31) / 31, mode="same")
    mask = freqs >= 25.0
    idx, _ = find_peaks(med_db[mask], prominence=3.0)
    all_peaks: list[tuple[float, float]] = []
    for i in idx:
        f = float(freqs[mask][i])
        i_full = int(np.argmin(np.abs(freqs - f)))
        excess = float(med_db[i_full] - continuum[i_full])
        if excess > 2.5:
            all_peaks.append((f, excess))
    all_peaks.sort(key=lambda t: -t[1])

    # Mains comb: pick the 4 strongest peaks below 300 Hz, sort by frequency
    mains = sorted([p for p in all_peaks if p[0] < 300.0],
                   key=lambda t: -t[1])[:4]
    mains_freqs = sorted(f for f, _ in mains)

    # Switching band: ~4 kHz to 6 kHz region; take the strongest peaks there
    sw_band = [p for p in all_peaks if 3500.0 < p[0] < 6000.0]
    sw_band.sort(key=lambda t: -t[1])
    sw_sidebands = sorted(f for f, _ in sw_band[:2])
    if len(sw_sidebands) == 2:
        sw_center = 0.5 * (sw_sidebands[0] + sw_sidebands[1])
        sw_width_hz = max(sw_sidebands[1] - sw_sidebands[0] + 4.0, 20.0)
    elif len(sw_sidebands) == 1:
        sw_center = sw_sidebands[0]
        sw_width_hz = 20.0
    else:
        sw_center = None
        sw_width_hz = None

    return {
        "mains_freqs": mains_freqs,
        "switching_center": sw_center,
        "switching_sidebands": sw_sidebands,
        "switching_width_hz": sw_width_hz,
        "all_peaks_top10": all_peaks[:10],
    }


# =========================================================================== #
# 2. Filter design                                                            #
# =========================================================================== #
def design_highpass(fs: int, cutoff: float, order: int) -> np.ndarray:
    return butter(order, cutoff, btype="highpass", fs=fs, output="sos")


def design_notch_sos(f0: float, Q: float, fs: int) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    return tf2sos(b, a)


def build_prefilter_sos(detect: dict, fs: int) -> np.ndarray:
    sections = [design_highpass(fs, HP_CUTOFF_HZ, HP_ORDER)]
    for f0 in detect["mains_freqs"]:
        sections.append(design_notch_sos(f0, Q_MAINS, fs))
    if detect["switching_center"] is not None:
        sc = detect["switching_center"]
        # Choose Q so that -3 dB width just spans the sidebands + margin
        w3db = max(detect["switching_width_hz"], sc / Q_SWITCHING)
        Q_eff = sc / w3db
        sections.append(design_notch_sos(sc, Q_eff, fs))
    return np.vstack(sections)


# =========================================================================== #
# 3. Adaptive Wiener (copied from stage 6, same tuned params)                 #
# =========================================================================== #
def stft(x):
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window="hann", center=True)


def istft(S, length):
    return librosa.istft(
        S, hop_length=HOP, win_length=N_FFT, window="hann",
        length=length, center=True,
    )


def bootstrap_noise_psd(P: np.ndarray, N_shape: np.ndarray) -> np.ndarray:
    low = np.percentile(P, 20, axis=1)
    mask = N_shape > 1e-20
    L = float(np.median(low[mask] / N_shape[mask]))
    return L * N_shape


def adaptive_wiener(x: np.ndarray, N_shape: np.ndarray) -> np.ndarray:
    X = stft(x)
    P = (np.abs(X) ** 2).astype(np.float64)
    F, T = P.shape
    lam0 = bootstrap_noise_psd(P, N_shape).astype(np.float64)
    mins_buf = np.full((F, D_FRAMES), lam0[:, None] / B_MIN)
    P_sm = P[:, 0].copy()

    G = np.empty_like(P)
    S_sq_prev = np.maximum(P[:, 0] - lam0, 0.0)
    lam_prev = lam0.copy()
    xi0 = np.maximum(S_sq_prev / (lam_prev + 1e-20), XI_MIN)
    G[:, 0] = xi0 / (1 + xi0)

    for t in range(1, T):
        P_sm = ALPHA_P * P_sm + (1 - ALPHA_P) * P[:, t]
        mins_buf[:, t % D_FRAMES] = P_sm
        P_min = mins_buf.min(axis=1)
        lam_t = B_MIN * P_min
        gamma_t = P[:, t] / (lam_t + 1e-20)
        xi = ALPHA_DD * (S_sq_prev / (lam_prev + 1e-20)) + \
             (1 - ALPHA_DD) * np.maximum(gamma_t - 1.0, 0.0)
        xi = np.maximum(xi, XI_MIN)
        Gt = xi / (1 + xi)
        G[:, t] = Gt
        S_sq_prev = (Gt ** 2) * P[:, t]
        lam_prev = lam_t

    G = np.maximum(G, GAIN_MIN)
    Y = G.astype(np.complex64) * X
    return istft(Y, length=len(x)).astype(np.float32)


# =========================================================================== #
# 4. Per-file pipeline + evaluation                                           #
# =========================================================================== #
def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


def filter_one(x: np.ndarray, sos: np.ndarray, N_shape: np.ndarray) -> np.ndarray:
    y = sosfiltfilt(sos, x).astype(np.float32)
    y = adaptive_wiener(y, N_shape)
    peak = float(np.max(np.abs(y)))
    if peak > 0.999:
        y = (y * (0.999 / peak)).astype(np.float32)
    return y


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64)
    est = est[:n].astype(np.float64)
    return 10.0 * np.log10(ref @ ref / (np.sum((ref - est) ** 2) + 1e-20))


def extract_features(x: np.ndarray, sr: int) -> np.ndarray:
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


# =========================================================================== #
# main                                                                        #
# =========================================================================== #
def main() -> None:
    assert BOOTSTRAP_PROFILE.exists(), f"missing bootstrap profile: {BOOTSTRAP_PROFILE}"
    N_shape = np.load(BOOTSTRAP_PROFILE)

    # ---- 1. Detect the narrowband noise lines from noisy audio alone
    print("Detecting noise lines blindly from noisy/*.wav ...")
    detect = detect_noise_lines(FS_EXPECTED, n_files_per_genre=8)
    mf = detect["mains_freqs"]
    sc = detect["switching_center"]
    sb = detect["switching_sidebands"]
    print(f"  mains comb (detected)      : {['%.1f' % f for f in mf]} Hz")
    print(f"  switching tone (detected)  : centre={sc:.1f} Hz, "
          f"sidebands={['%.1f' % f for f in sb]}")
    print(f"  top 10 peaks over continuum: "
          f"{['%.1f Hz/%.1f dB' % p for p in detect['all_peaks_top10']]}")

    # ---- 2. Build the fixed linear prefilter
    sos = build_prefilter_sos(detect, FS_EXPECTED)
    print(f"  prefilter SOS shape        : {sos.shape}")

    # ---- 3. Evaluate on the 9-file starter set
    print("\n--- STARTER SET (9 files) ---")
    feats_clean, feats_noisy, feats_filt = [], [], []
    snr_in, snr_out = [], []
    t0 = time.time()
    for i, name in enumerate(STARTER_FILES, 1):
        noisy, sr = load_mono(NOISY_DIR / name)
        y = filter_one(noisy, sos, N_shape)
        sf.write(str(OUT_DIR / f"{PREFIX}{name}"), y, sr, subtype="PCM_16")

        clean, _ = load_mono(CLEAN_DIR / name)  # EVAL ONLY
        snr_in.append(snr_db(clean, noisy))
        snr_out.append(snr_db(clean, y))
        feats_clean.append(extract_features(clean, sr))
        feats_noisy.append(extract_features(noisy, sr))
        feats_filt.append(extract_features(y, sr))
        print(f"  [{i}/9] {name:<14s}  SNR {snr_in[-1]:+6.2f} -> "
              f"{snr_out[-1]:+6.2f} dB  (+{snr_out[-1]-snr_in[-1]:+5.2f} dB)")

    A = np.array(feats_clean); N = np.array(feats_noisy); F = np.array(feats_filt)
    scale = A.std(axis=0) + 1e-6

    def dist(X):
        return np.array([np.linalg.norm((X[i] - A[i]) / scale)
                         for i in range(len(STARTER_FILES))])

    dn = dist(N); df = dist(F)
    print("\nPer-file feature distance to clean (eval only):")
    print(f"{'file':<14s} {'noisy':>10s} {'stage7':>10s}  {'delta':>8s}")
    for i, name in enumerate(STARTER_FILES):
        arrow = "OK  " if df[i] < dn[i] else "WARN"
        print(f"{name:<14s} {dn[i]:10.2f} {df[i]:10.2f}  "
              f"{df[i]-dn[i]:+8.2f}  {arrow}")
    print(f"{'mean':<14s} {dn.mean():10.2f} {df.mean():10.2f}  "
          f"{df.mean()-dn.mean():+8.2f}")

    blocks = {
        "MFCC mean (20)": slice(0, 20),
        "MFCC std  (20)": slice(20, 40),
        "centroid (2)":   slice(40, 42),
        "rolloff  (2)":   slice(42, 44),
        "bandwidth(2)":   slice(44, 46),
        "ZCR      (2)":   slice(46, 48),
    }
    print("\nPer-block mean normalised L2 to clean:")
    print(f"{'block':<18s} {'noisy':>8s} {'stage7':>10s}  {'delta':>8s}")
    for bname, sl in blocks.items():
        bs = scale[sl]

        def bd(X):
            return np.mean([np.linalg.norm((X[i][sl] - A[i][sl]) / bs)
                            for i in range(len(STARTER_FILES))])

        nv, fv = bd(N), bd(F)
        flag = "WARN" if fv > nv else "ok"
        print(f"{bname:<18s} {nv:8.2f} {fv:10.2f}  {fv-nv:+8.2f}  {flag}")

    print(f"\nSNR on starter set:")
    print(f"  noisy  : {np.mean(snr_in):+6.2f} dB")
    print(f"  stage7 : {np.mean(snr_out):+6.2f} dB  "
          f"(+{np.mean(snr_out)-np.mean(snr_in):+5.2f} dB)")

    # ---- 4. Process the full 250-file set so the classifier can be rerun
    print(f"\n--- FULL BATCH ---")
    all_files = sorted(NOISY_DIR.glob("*.wav"))
    done = 0
    t1 = time.time()
    for p in all_files:
        if (OUT_DIR / f"{PREFIX}{p.name}").exists() and p.name in STARTER_FILES:
            done += 1
            continue
        x, sr = load_mono(p)
        y = filter_one(x, sos, N_shape)
        sf.write(str(OUT_DIR / f"{PREFIX}{p.name}"), y, sr, subtype="PCM_16")
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(all_files)} done  "
                  f"({(time.time()-t1)/done:.2f}s/file)")
    print(f"  {done}/{len(all_files)} done  "
          f"total {(time.time()-t1):.1f}s")

    # ---- 5. Write README
    readme_path = REPORT_DIR / "README.txt"
    lines = [
        "COMPONENT-MATCHED FILTER v1 (stage 7)",
        "=" * 72,
        "",
        "Pipeline (all non-oracle — uses only noisy-side information):",
        f"  1. Butterworth HP, cutoff={HP_CUTOFF_HZ} Hz, order={HP_ORDER}, "
        f"zero-phase",
        f"  2. Mains comb — IIR notches at detected frequencies "
        f"{['%.1f' % f for f in mf]} Hz, Q={Q_MAINS}",
        f"  3. Switching notch — centred at "
        f"{sc:.1f} Hz (mean of sidebands "
        f"{['%.1f' % f for f in sb]}), "
        f"adaptive Q so width = max({sc/Q_SWITCHING:.1f} Hz, "
        f"{detect['switching_width_hz']:.1f} Hz)",
        f"  4. Adaptive Wiener (MS + DD), "
        f"alpha_p={ALPHA_P}, alpha_dd={ALPHA_DD}, "
        f"D={D_FRAMES} frames, B_min={B_MIN}, xi_min={XI_MIN}",
        "",
        "All notch frequencies are re-discovered from the noisy audio at the",
        "start of each run (low-energy-frame PSD + find_peaks). The spec",
        "(noise.md) is NOT consulted for any numeric coefficient.",
        "",
        "Output WAVs go to",
        f"  {OUT_DIR}",
        "so that the Training_model cross-condition experiments can use",
        "the stage-7 file set the same way they use stage-6.",
        "",
        f"Starter-set summary (9 files = 3 x jazz/metal/pop):",
        f"  mean noisy  SNR = {np.mean(snr_in):+6.2f} dB",
        f"  mean stage7 SNR = {np.mean(snr_out):+6.2f} dB "
        f"(delta {np.mean(snr_out)-np.mean(snr_in):+5.2f} dB)",
        f"  mean noisy  feat-distance = {dn.mean():.2f}",
        f"  mean stage7 feat-distance = {df.mean():.2f} "
        f"(delta {df.mean()-dn.mean():+.2f})",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nREADME: {readme_path}")
    print(f"WAVs  : {OUT_DIR}")
    print(f"Elapsed total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
