"""
ECG, vibration, and audio — load, trim, contaminate with the SAME composite
EMC + measurement-noise model, plot, and report metrics.

Noise model (shared across the three signals, same relative composition and
same target SNR):
  EMC chain
    - narrowband mains hum at 50 Hz + harmonics (100, 150, 200 Hz)
    - high-frequency switching tone at 0.40 * Nyquist with small FM (DC/DC-like)
    - band-limited broadband EMI (0.05..0.95 * Nyquist, Gaussian)
    - impulsive bursts (3 / s, ~5 ms decay, random amplitude/sign)
  Measurement chain
    - white thermal noise
    - 1/f-like baseline drift (cumulative-sum integrator)
    - ADC quantization noise (uniform, 8-bit equivalent)
  Overall SNR is then scaled to a single fixed target (-3 dB) on every signal.
"""

import os
import numpy as np
import scipy.signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from scipy.io import wavfile, loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SEED = 42
SNR_TARGET_DB = -10.0         # heavily flooded (noise power 10x signal power)
MAINS_HZ = 50.0
SWITCHING_FRAC = 0.40         # fraction of Nyquist
N_BURSTS_PER_SEC = 3.0
BURST_DECAY_S = 0.005
ADC_BITS = 8


# ---------------------------- DATA LOADERS ---------------------------- #

def load_ecg():
    import wfdb
    fs = 360
    sig_arr, fields = wfdb.rdsamp("100", sampto=10 * fs, pn_dir="mitdb")
    sig = sig_arr[:, 0].astype(float)                             # MLII lead
    return (sig, fs,
            "MIT-BIH Arrhythmia DB · record 100 · lead MLII · first 10 s",
            "mV")


def load_vibration():
    url = "https://engineering.case.edu/sites/default/files/97.mat"
    fpath = os.path.join(OUT, "cwru_97.mat")
    if not os.path.exists(fpath):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(r.content)
    mat = loadmat(fpath)
    de_keys = [k for k in mat if k.endswith("_DE_time")]
    key = de_keys[0]
    sig = mat[key].squeeze().astype(float)
    fs = 12_000
    sig = sig[: 3 * fs]                                           # first 3 s
    return (sig, fs,
            f"CWRU Bearing Data Center · Normal baseline 97.mat · {key} · first 3 s",
            "g")


def load_music():
    """First clean track from the GTZAN-derived music dataset (metal_01 by default).

    Falls back to building the dataset if it isn't there yet.
    """
    track = os.path.join(OUT, "Model", "clean", "metal_01.wav")
    if not os.path.exists(track):
        raise RuntimeError(
            f"Music dataset not built yet — expected {track}. "
            "Run build_music_dataset.py first."
        )
    fs, data = wavfile.read(track)
    if data.ndim == 2:
        data = data[:, 0]
    sig = data.astype(float)
    if np.issubdtype(data.dtype, np.integer):
        sig /= np.iinfo(data.dtype).max
    sig = sig[: 4 * fs]                                           # first 4 s
    return (sig, fs,
            "GTZAN metal_01 (from out/Model/clean) · first 4 s",
            "amplitude (normalised)")


# ---------------------------- NOISE MODEL ---------------------------- #

def make_noise(clean, fs, seed=SEED):
    """Return (emc_noise, measurement_noise, total_noise) scaled to SNR_TARGET_DB."""
    rng = np.random.default_rng(seed)
    n = clean.size
    t = np.arange(n) / fs
    nyq = fs / 2

    # mains hum + harmonics, amplitudes roll off with harmonic order
    hum = np.zeros(n)
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
        f = MAINS_HZ * k
        if f >= 0.95 * nyq:
            break
        hum += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    # switching tone with small FM
    f_sw = SWITCHING_FRAC * nyq
    fm_dev = 0.002 * f_sw * np.sin(2 * np.pi * 0.7 * t)
    switching = 0.6 * np.sin(2 * np.pi * f_sw * t + 2 * np.pi * np.cumsum(fm_dev) / fs)

    # band-limited broadband EMI
    bb_raw = rng.standard_normal(n)
    sos = sps.butter(4, [0.05 * nyq, 0.95 * nyq], btype="band", output="sos", fs=fs)
    broadband = 0.8 * sps.sosfiltfilt(sos, bb_raw)

    # impulsive EMI bursts (decaying noise bursts at random times)
    bursts = np.zeros(n)
    n_bursts = max(1, int(round(N_BURSTS_PER_SEC * (n / fs))))
    burst_len = max(8, int(BURST_DECAY_S * fs * 4))
    decay = np.exp(-np.arange(burst_len) / max(1.0, BURST_DECAY_S * fs))
    for _ in range(n_bursts):
        idx = int(rng.integers(0, max(1, n - burst_len)))
        amp = rng.uniform(2.5, 5.0) * (1 if rng.random() < 0.5 else -1)
        bursts[idx:idx + burst_len] += amp * decay * rng.standard_normal(burst_len)

    emc = hum + switching + broadband + bursts

    # white thermal noise
    white = rng.standard_normal(n)
    # 1/f-like slow drift via integrated white noise
    drift = np.cumsum(rng.standard_normal(n))
    drift = drift / np.std(drift) if np.std(drift) > 0 else drift
    # quantization: uniform in [-q/2, q/2] based on clean full-scale
    full_scale = np.max(np.abs(clean)) + 1e-12
    q = 2 * full_scale / (2 ** ADC_BITS)
    quant = rng.uniform(-q / 2, q / 2, n)
    measurement = 0.6 * white + 1.5 * drift + 1.0 * quant

    # scale so the sum reaches the target SNR
    p_sig = float(np.mean(clean ** 2))
    raw_total = emc + measurement
    p_noise_raw = float(np.mean(raw_total ** 2))
    target_p_noise = p_sig / (10 ** (SNR_TARGET_DB / 10))
    scale = np.sqrt(target_p_noise / p_noise_raw) if p_noise_raw > 0 else 1.0
    emc *= scale
    measurement *= scale
    total = emc + measurement
    return emc, measurement, total


# ---------------------------- PIPELINE ---------------------------- #

def db_power(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-30)


def process(name, clean, fs, source, units):
    emc_noise, measurement_noise, total_noise = make_noise(clean, fs)
    noisy_signal = clean + total_noise

    # realistic ADC saturation at 3x peak of clean signal
    full_scale_adc = 3.0 * (np.max(np.abs(clean)) + 1e-12)
    pre_clip_peak = float(np.max(np.abs(noisy_signal)))
    was_clipped = pre_clip_peak > full_scale_adc
    noisy_signal = np.clip(noisy_signal, -full_scale_adc, full_scale_adc)

    snr_db = db_power(clean) - db_power(total_noise)
    rms_clean = float(np.sqrt(np.mean(clean ** 2)))
    rms_noise = float(np.sqrt(np.mean(total_noise ** 2)))

    fig, axes = plt.subplots(5, 2, figsize=(14, 15))
    t = np.arange(clean.size) / fs
    panels = [
        ("Clean signal",           clean),
        ("EMC noise alone",        emc_noise),
        ("Measurement noise alone", measurement_noise),
        ("Total noise",            total_noise),
        ("Noisy signal (post ADC clip)", noisy_signal),
    ]
    for ax, (ttl, y) in zip(axes.flat[:5], panels):
        ax.plot(t, y, lw=0.6)
        ax.set_title(ttl)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(units)
        ax.grid(alpha=0.3)

    nperseg = max(64, min(2048, clean.size // 4))
    f1, P1 = sps.welch(clean, fs=fs, nperseg=nperseg)
    f2, P2 = sps.welch(noisy_signal, fs=fs, nperseg=nperseg)
    ax = axes.flat[5]
    ax.semilogy(f1, P1, label="clean")
    ax.semilogy(f2, P2, label="noisy", alpha=0.7)
    ax.set_title("Welch PSD — clean vs noisy")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    sp_nperseg = max(64, min(1024, clean.size // 32))
    f_c, t_c, Sxx_c = sps.spectrogram(clean,        fs=fs, nperseg=sp_nperseg)
    f_n, t_n, Sxx_n = sps.spectrogram(noisy_signal, fs=fs, nperseg=sp_nperseg)
    # shared dB color scale for fair visual comparison
    vmin = float(np.min([10*np.log10(Sxx_c + 1e-12).min(),
                         10*np.log10(Sxx_n + 1e-12).min()]))
    vmax = float(np.max([10*np.log10(Sxx_c + 1e-12).max(),
                         10*np.log10(Sxx_n + 1e-12).max()]))

    ax = axes.flat[6]
    ax.pcolormesh(t_c, f_c, 10 * np.log10(Sxx_c + 1e-12),
                  shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Clean-signal spectrogram [dB]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("frequency [Hz]")

    ax = axes.flat[7]
    ax.pcolormesh(t_n, f_n, 10 * np.log10(Sxx_n + 1e-12),
                  shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Noisy-signal spectrogram [dB]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("frequency [Hz]")

    ax = axes.flat[8]
    ax.axis("off")
    axes.flat[9].axis("off")
    report = (
        f"SOURCE\n  {source}\n\n"
        f"fs            = {fs} Hz\n"
        f"duration      = {clean.size / fs:.3f} s  ({clean.size} samples)\n"
        f"target SNR    = {SNR_TARGET_DB:+.1f} dB\n"
        f"achieved SNR  = {snr_db:+.2f} dB\n"
        f"RMS clean     = {rms_clean:.4g} {units}\n"
        f"RMS noise     = {rms_noise:.4g} {units}\n"
        f"peak noise    = {float(np.max(np.abs(total_noise))):.4g}\n"
        f"pre-clip peak = {pre_clip_peak:.4g}\n"
        f"ADC full-scale= {full_scale_adc:.4g} (3x clean peak)\n"
        f"clipped       = {'YES' if was_clipped else 'no'}\n\n"
        "Shared noise components\n"
        f"  mains hum         : 50 Hz + harmonics (rolled off)\n"
        f"  switching tone    : {SWITCHING_FRAC:.2f}*Nyquist = {SWITCHING_FRAC*fs/2:.1f} Hz (FM'd)\n"
        f"  broadband EMI     : band [0.05..0.95]*Nyquist\n"
        f"  impulsive bursts  : {N_BURSTS_PER_SEC:.0f}/s, ~{BURST_DECAY_S*1000:.0f} ms decay\n"
        f"  thermal white     : gaussian\n"
        f"  1/f drift         : cumulative-sum integrator\n"
        f"  quantization      : {ADC_BITS}-bit uniform\n"
    )
    ax.text(0.0, 1.0, report, va="top", family="monospace", fontsize=9,
            transform=ax.transAxes)

    fig.suptitle(f"{name} — clean vs EMC + measurement contamination "
                 f"(target SNR = {SNR_TARGET_DB:+.1f} dB, achieved {snr_db:+.2f} dB)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUT, f"{name.lower()}_plots.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return {
        "name": name, "source": source, "fs": fs,
        "samples": int(clean.size),
        "duration_s": float(clean.size / fs),
        "target_snr_db": SNR_TARGET_DB,
        "achieved_snr_db": float(snr_db),
        "rms_clean": rms_clean, "rms_noise": rms_noise,
        "clipped": bool(was_clipped), "plot": out_path,
    }


def main():
    np.random.seed(SEED)
    loaders = [
        ("ECG",       load_ecg),
        ("Vibration", load_vibration),
        ("Music",     load_music),
    ]
    summaries = []
    for name, fn in loaders:
        try:
            print(f"[{name}] loading...")
            clean, fs, source, units = fn()
            print(f"[{name}] loaded: {clean.size} samples @ {fs} Hz "
                  f"({clean.size / fs:.2f} s)")
            r = process(name, clean, fs, source, units)
            summaries.append(r)
            print(f"[{name}] plots saved -> {r['plot']}  "
                  f"(achieved SNR = {r['achieved_snr_db']:+.2f} dB)")
        except Exception as e:
            print(f"[{name}] FAILED: {e!r}")
            summaries.append({"name": name, "error": repr(e)})

    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        for r in summaries:
            f.write("=" * 74 + "\n")
            for k, v in r.items():
                f.write(f"{k}: {v}\n")
        f.write("=" * 74 + "\n")

    print("\nDone. Artifacts in:", OUT)


if __name__ == "__main__":
    main()
