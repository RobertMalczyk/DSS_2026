"""Re-run the exact noise generator and dump the per-component RMS and scale factor."""
import numpy as np
import scipy.signal as sps
from signal_contamination import (
    load_ecg, load_vibration, load_audio,
    SNR_TARGET_DB, MAINS_HZ, SWITCHING_FRAC,
    N_BURSTS_PER_SEC, BURST_DECAY_S, ADC_BITS, SEED,
)

def gen(clean, fs, seed=SEED):
    rng = np.random.default_rng(seed)
    n = clean.size
    t = np.arange(n) / fs
    nyq = fs / 2
    hum = np.zeros(n)
    hum_lines = []
    for k, amp in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
        f = MAINS_HZ * k
        if f >= 0.95 * nyq:
            break
        hum += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        hum_lines.append((f, amp))
    f_sw = SWITCHING_FRAC * nyq
    fm_dev = 0.002 * f_sw * np.sin(2 * np.pi * 0.7 * t)
    switching = 0.6 * np.sin(2 * np.pi * f_sw * t + 2 * np.pi * np.cumsum(fm_dev) / fs)
    bb_raw = rng.standard_normal(n)
    sos = sps.butter(4, [0.05 * nyq, 0.95 * nyq], btype="band", output="sos", fs=fs)
    broadband = 0.8 * sps.sosfiltfilt(sos, bb_raw)
    bursts = np.zeros(n)
    n_bursts = max(1, int(round(N_BURSTS_PER_SEC * (n / fs))))
    burst_len = max(8, int(BURST_DECAY_S * fs * 4))
    decay = np.exp(-np.arange(burst_len) / max(1.0, BURST_DECAY_S * fs))
    for _ in range(n_bursts):
        idx = int(rng.integers(0, max(1, n - burst_len)))
        amp = rng.uniform(2.5, 5.0) * (1 if rng.random() < 0.5 else -1)
        bursts[idx:idx + burst_len] += amp * decay * rng.standard_normal(burst_len)
    emc = hum + switching + broadband + bursts

    white = rng.standard_normal(n)
    drift = np.cumsum(rng.standard_normal(n))
    drift = drift / np.std(drift)
    full_scale = np.max(np.abs(clean)) + 1e-12
    q = 2 * full_scale / (2 ** ADC_BITS)
    quant = rng.uniform(-q / 2, q / 2, n)
    meas = 0.6 * white + 1.5 * drift + 1.0 * quant

    raw_total = emc + meas
    p_sig = float(np.mean(clean ** 2))
    p_noise_raw = float(np.mean(raw_total ** 2))
    target_p_noise = p_sig / (10 ** (SNR_TARGET_DB / 10))
    scale = np.sqrt(target_p_noise / p_noise_raw)
    return dict(
        n=n, fs=fs, nyq=nyq, hum_lines=hum_lines, f_sw=f_sw,
        n_bursts=n_bursts, burst_len=burst_len, burst_len_s=burst_len / fs,
        rms={
            "hum":        float(np.std(hum)),
            "switching":  float(np.std(switching)),
            "broadband":  float(np.std(broadband)),
            "bursts":     float(np.std(bursts)),
            "white(0.6)": float(np.std(0.6 * white)),
            "drift(1.5)": float(np.std(1.5 * drift)),
            "quant(1.0)": float(np.std(quant)),
        },
        q=q, scale=float(scale),
        emc_rms=float(np.std(emc)), meas_rms=float(np.std(meas)),
        final_noise_rms=float(np.sqrt(np.mean((scale * raw_total) ** 2))),
        clean_rms=float(np.sqrt(p_sig)),
    )

for name, fn in [("ECG", load_ecg), ("Vibration", load_vibration), ("Audio", load_audio)]:
    clean, fs, *_ = fn()
    r = gen(clean, fs)
    print(f"--- {name} ---")
    for k, v in r.items():
        print(f"  {k}: {v}")
    print()
