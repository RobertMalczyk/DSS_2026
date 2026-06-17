"""Stage 3 filter: oracle Wiener filter in complex-Morlet CWT domain.

For each matched pair (clean[i], noisy[i]):
  C_clean = CWT(clean)
  C_noisy = CWT(noisy)
  C_noise = C_noisy - C_clean                   # CWT is linear
  W       = |C_clean|^2 / (|C_clean|^2 + |C_noise|^2 + eps)
  C_filt  = W * C_noisy
  y       = iCWT(C_filt)

This is the classical time-frequency Wiener filter applied in the CWT plane
(constant-Q / log-scale tiling) instead of STFT. Using the clean reference makes
it an ORACLE filter: the upper bound on what wavelet denoising can achieve.

Wavelet: generalized Morlet, mu=6 (canonical choice - same carrier as the
'cmor*' family used in the earlier scalograms, reconstructs well with ssqueezepy).
"""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from ssqueezepy import cwt, icwt, Wavelet

CLEAN_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\clean")
NOISY_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\noisy")
OUT_DIR = Path(r"C:\Robak\DSS2026\Claude\Signal_generation\out\Model\Filtered assist and WAVELET 3rd stage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "filtered_assist_wavelet_3rd_stage_"
MU = 6                # Morlet carrier (canonical)
NV = 32               # voices per octave
EPS = 1e-10           # Wiener numerical floor


def snr_db(ref, est):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    n = min(len(ref), len(est))
    ref = ref[:n]; est = est[:n]
    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2) + 1e-20
    return 10.0 * np.log10(num / den)


def process_one(name: str, wav: Wavelet):
    clean_path = CLEAN_DIR / name
    noisy_path = NOISY_DIR / name

    clean, sr_c = sf.read(str(clean_path), always_2d=False)
    noisy, sr_n = sf.read(str(noisy_path), always_2d=False)
    assert sr_c == sr_n, f"sr mismatch for {name}: {sr_c} vs {sr_n}"
    sr = sr_c

    if clean.ndim == 2:
        clean = clean.mean(axis=1)
    if noisy.ndim == 2:
        noisy = noisy.mean(axis=1)

    n = min(len(clean), len(noisy))
    clean = clean[:n].astype(np.float32)
    noisy = noisy[:n].astype(np.float32)

    C_clean, scales = cwt(clean, wavelet=wav, fs=sr, nv=NV)
    C_noisy, _      = cwt(noisy, wavelet=wav, fs=sr, nv=NV)

    C_noise = C_noisy - C_clean
    P_c = np.abs(C_clean) ** 2
    P_n = np.abs(C_noise) ** 2
    W = P_c / (P_c + P_n + EPS)

    C_filt = W * C_noisy
    y = icwt(C_filt, wavelet=wav, scales=scales, nv=NV)
    y = np.real(y).astype(np.float32)[:n]

    # Amplitude match: Wiener mask (<=1) attenuates overall RMS.
    # Rescale to clean RMS; cap to avoid clipping past +/-0.999.
    rms_clean = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
    rms_filt = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak_filt = float(np.max(np.abs(y)))
    gain_rms = rms_clean / (rms_filt + 1e-20)
    gain_clip = 0.999 / (peak_filt + 1e-20)
    gain = min(gain_rms, gain_clip)
    y = (y * gain).astype(np.float32)

    info = sf.info(str(noisy_path))
    out_path = OUT_DIR / f"{PREFIX}{name}"
    sf.write(str(out_path), y, sr, subtype=info.subtype)

    return {
        "name": name,
        "sr": sr,
        "samples": n,
        "snr_noisy_db": snr_db(clean, noisy),
        "snr_filtered_db": snr_db(clean, y),
        "gain": gain,
        "clipped_rms_target": gain < gain_rms,
        "out": out_path,
    }


def main():
    # Process every file present in BOTH clean and noisy folders.
    clean_files = {p.name for p in CLEAN_DIR.glob("*.wav")}
    noisy_files = {p.name for p in NOISY_DIR.glob("*.wav")}
    shared = sorted(clean_files & noisy_files)
    only_clean = sorted(clean_files - noisy_files)
    only_noisy = sorted(noisy_files - clean_files)
    print(f"shared pairs: {len(shared)}  clean-only: {len(only_clean)}  noisy-only: {len(only_noisy)}")
    if only_clean:
        print("  skipping clean-only:", only_clean[:5], "...")
    if only_noisy:
        print("  skipping noisy-only:", only_noisy[:5], "...")

    wav = Wavelet(("morlet", {"mu": MU}))
    print(f"Wavelet: generalized Morlet, mu={MU}, nv={NV}, eps={EPS}")
    print(f"Output -> {OUT_DIR}")

    snr_in = []
    snr_out = []
    t0 = time.time()
    for i, name in enumerate(shared, 1):
        try:
            r = process_one(name, wav)
        except Exception as e:
            print(f"  [{i:3d}/{len(shared)}] FAILED {name}: {e!r}")
            continue
        snr_in.append(r["snr_noisy_db"])
        snr_out.append(r["snr_filtered_db"])
        if i % 10 == 0 or i == len(shared) or i <= 3:
            gain = r["snr_filtered_db"] - r["snr_noisy_db"]
            print(f"  [{i:3d}/{len(shared)}] {r['out'].name}  "
                  f"SNR {r['snr_noisy_db']:+6.2f} -> {r['snr_filtered_db']:+6.2f} dB "
                  f"(+{gain:5.2f}), elapsed {time.time()-t0:.1f}s")

    if snr_in:
        print(f"\nSummary over {len(snr_in)} files:")
        print(f"  mean SNR (noisy)    = {np.mean(snr_in):+6.2f} dB")
        print(f"  mean SNR (filtered) = {np.mean(snr_out):+6.2f} dB")
        print(f"  mean improvement    = {np.mean(np.array(snr_out)-np.array(snr_in)):+6.2f} dB")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
