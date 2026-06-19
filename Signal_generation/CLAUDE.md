# Task: Download, contaminate, describe, and plot a raw signal

You are given access to public datasets and standard Python scientific libraries. Your goal is to obtain a **raw signal** from an openly available dataset, trim it to a sensible working length, research realistic **EMC/EMI** and **measurement-noise** contamination, add those disturbances strongly enough to substantially degrade the signal, and then describe and plot the results for the user.

## Objectives

1. **Find and download one public raw signal** from an open dataset.
   - Prefer a dataset that provides real sampled waveform data (for example ECG, vibration, audio, radar, EEG, RF/IQ, seismic, or similar).
   - Prefer a source that is easy to access programmatically.
   - Briefly explain what the signal is, where it came from, its sample rate, units if known, and the file format.

2. **Trim the signal to a sensible length** for experimentation.
   - The segment should be long enough to show signal structure but short enough to plot and process comfortably.
   - As a rule of thumb, target something like a few seconds to a few tens of seconds, unless the domain clearly requires another duration.
   - State exactly how the segment was chosen.

3. **Research realistic disturbance models** before adding noise.
   - Do not invent arbitrary noise blindly.
   - First research how **EMC/EMI disturbances** and **measurement noise** typically appear for this type of signal and in general instrumentation.
   - Summarize the most relevant findings in clear technical language.

## Disturbance model requirements

Build a composite contamination model that can include, where appropriate:

- **Broadband EMI / wideband noise**
  - Often approximated as white or band-limited noise.
- **Narrowband interference**
  - Sinusoidal or multi-tone components, harmonics, switching frequencies, mains hum (50/60 Hz) and harmonics, or clock-like tones if appropriate.
- **Transient / impulsive EMI**
  - Bursts, spikes, short pulses, intermittent interference, or burst trains.
- **Low-frequency drift / 1/f-like noise**
  - Baseline wander, flicker-like components, or slow instrumentation drift where appropriate.
- **Measurement-chain noise**
  - Thermal/white noise, quantization-like noise, ADC-related effects, saturation/clipping if justified, and occasional outliers if realistic.

Use the research to decide which components are appropriate for the chosen signal. EMC disturbance signals are often described as **transient, narrowband, or broadband**, and measurement noise commonly includes **white noise**, **1/f noise**, and **quantization-related effects**. Use those ideas as a starting point, but adapt them to the selected signal and cite your reasoning.  
Source guidance: fast/intermittent EMI can be transient, narrowband, or broadband; white noise is spread across the Nyquist band; ADC/input noise may include white and 1/f components; ADC error behavior can also include Gaussian-like random noise and occasional anomalies. citeturn303866search2turn303866search1turn303866search7turn303866search16turn303866view0

## Noise strength requirement

The final contaminated signal should be **heavily flooded** by interference and measurement noise.

This means:
- The original signal should still be conceptually present, but substantially obscured.
- The contamination should be strong enough that the user can clearly see the degradation in both time-domain and frequency-domain views.
- Choose noise amplitudes deliberately rather than arbitrarily.
- Report the achieved contamination level using metrics such as:
  - SNR
  - signal power vs noise power
  - RMS values of the clean and noise components
  - peak amplitudes if relevant

## Implementation requirements

1. Use Python.
2. Download the chosen dataset programmatically when practical.
3. Load the raw signal into memory.
4. Trim it to the chosen analysis window.
5. Create the disturbance components separately.
6. Combine them into a total contamination signal.
7. Add the contamination to the clean signal.
8. Keep the following arrays available and clearly named:
   - `clean_signal`
   - `emc_noise`
   - `measurement_noise`
   - `total_noise`
   - `noisy_signal`

## Analysis and plotting requirements

At the end, present all of the following to the user:

1. **Short description of the source signal**
   - dataset name
   - record/file used
   - sample rate
   - selected duration
   - why this segment was chosen

2. **Short description of every added noise component**
   - what it represents physically
   - why it is plausible
   - its main characteristics (frequency content, impulsive nature, drift, broadband nature, etc.)
   - its amplitude or power settings

3. **Plots**
   - Clean signal in time domain
   - EMC noise alone in time domain
   - Measurement noise alone in time domain
   - Total noise in time domain
   - Noisy signal in time domain
   - Magnitude spectrum or PSD of the clean signal
   - Magnitude spectrum or PSD of the noisy signal
   - Spectrogram of the noisy signal if useful

4. **Comparison metrics**
   - SNR before/after if applicable
   - RMS of clean signal and total noise
   - any clipping/saturation notes

## Output style

- Be practical and concise.
- Explain assumptions explicitly.
- Do not hide implementation choices.
- If the first chosen dataset is inaccessible, switch to another public raw-signal dataset and continue.
- Prefer reproducible code with a fixed random seed.

## Deliverables

Produce:
1. a short research summary,
2. the Python code,
3. the generated plots,
4. a final explanation of the clean signal, the EMC noise, the measurement noise, and the final contaminated waveform.

---

# Current implementation (last updated 2026-04-22)

The task has been implemented for **three signals with a single shared noise model** (ECG + vibration + audio). The same `make_noise(clean, fs)` routine runs on each, with all components scaled by one factor α so the final SNR is identical on every signal.

## Files

| Path | Purpose |
|---|---|
| `signal_contamination.py` | Main pipeline: load all three signals, generate shared-pattern noise, save 5×2 detailed plots + `summary.txt` |
| `plot_compare.py` | Comparison figures: 3×3 time-domain (clean / noisy / noise) and 3×2 spectrograms (clean vs noisy) |
| `probe_noise.py` | Diagnostic: dumps per-component raw RMS, LSB, and the scale factor α for each signal |
| `generate_samples.py` | Standalone sample exporter: writes the three post-clip noisy signals as 32-bit float WAVs under `out/Samples/` for downstream consumers |
| `out/*.png` | All generated plots |
| `out/summary.txt` | Numeric metrics per signal |
| `out/Samples/*_noisy.wav`, `out/Samples/samples_info.txt` | Noisy WAV exports + leak-free metadata (fs / duration / samples only — no source, no units, no noise info). Sole input for `Signal_detection_RAW` |
| `out/Samples/Clean/*_clean.wav`, `out/Samples/Clean/metrics.txt` | Matched clean WAVs + full RMS/SNR/clip metrics. **Off-limits to `Signal_detection_RAW`** — kept here for user inspection and any training that needs ground truth |
| `out/cwru_97.mat`, `out/audio.wav` | Cached downloads |

## How to run

Python is **not on PATH** — use the full exe:

```bash
PY="/c/Users/robak/AppData/Local/Programs/Python/Python312/python.exe"
"$PY" signal_contamination.py     # regenerate detailed per-signal figures
"$PY" plot_compare.py             # regenerate compare figures (imports from above)
"$PY" generate_samples.py         # regenerate out/Samples/*.wav + samples_info.txt
```

## Datasets used

| Signal | Source | File | fs | Duration |
|---|---|---|---|---|
| ECG | PhysioNet MIT-BIH Arrhythmia DB, record 100, lead MLII (via `wfdb`) | downloaded at runtime | 360 Hz | first 10 s |
| Vibration | CWRU Bearing Data Center, Normal baseline | `97.mat` → `X097_DE_time` | 12 kHz | first 3 s |
| Audio | Kozco public test recording | `piano2.wav` | 48 kHz | first 4 s |

## Shared noise model (current tuning)

Top-of-file constants in `signal_contamination.py`:

| Constant | Value | Meaning |
|---|---|---|
| `SEED` | 42 | deterministic output |
| `SNR_TARGET_DB` | **−10.0** | noise power = 10 × signal power |
| `MAINS_HZ` | 50.0 | European grid fundamental |
| `SWITCHING_FRAC` | 0.40 | switching tone at 0.40 × Nyquist |
| `N_BURSTS_PER_SEC` | 3.0 | impulsive EMI rate |
| `BURST_DECAY_S` | 0.005 | 5 ms exponential decay of each burst |
| `ADC_BITS` | 8 | quantization LSB = 2·peak(|clean|)/256 |

Components generated, in order (EMC first, then measurement):

1. **Mains hum + harmonics** — sines at 50/100/150/200 Hz (truncated at 0.95·Nyq), amplitudes 1.0/0.5/0.3/0.15, random phases.
2. **Switching / clock tone** — sine at 0.40·Nyq, peak 0.6, small FM (±0.2 % deviation, 0.7 Hz modulator).
3. **Broadband EMI** — zero-phase 4th-order Butterworth bandpass [0.05·Nyq, 0.95·Nyq] on Gaussian white, scaled 0.8.
4. **Impulsive bursts** — Gaussian × exponential decay, 3 per second, τ = 5 ms, random ±(2.5…5).
5. **Thermal white** — `0.6 × N(0,1)`.
6. **1/f-like drift** — cumulative-sum integrator, normalised, weighted 1.5 (dominates low-frequency power).
7. **Quantization** — uniform on `[−q/2, q/2]`, added rather than re-quantised.
8. **ADC saturation** — hard clip at `±3 × peak(|clean|)` applied after summation.

All seven noise pieces are summed, then the sum is multiplied by a single α chosen so total-noise power equals `P_signal × 10^(+10 dB/10)`. That makes the mix **identical in relative composition** across ECG / vibration / audio — only amplitude differs.

## Required array names (as produced inside `process(...)`)

`clean_signal` (= `clean`), `emc_noise`, `measurement_noise`, `total_noise`, `noisy_signal` — all available in-scope and plotted per spec.

## Generated plots

Per-signal detailed figures (`ecg_plots.png`, `vibration_plots.png`, `audio_plots.png`) are 5 rows × 2 columns:

1. Clean signal (time)
2. EMC noise alone (time)
3. Measurement noise alone (time)
4. Total noise (time)
5. Noisy signal post ADC-clip (time)
6. Welch PSD — clean vs noisy (log scale)
7. Clean-signal spectrogram (dB)
8. Noisy-signal spectrogram (dB, shared color scale with clean)
9. Metrics text panel
10. (blank filler)

Compare figures:

- `compare_clean_noisy_noise.png` — 3×3 time-domain, rows = ECG/Vib/Audio, cols = clean / noisy / noise-only, y-limits shared per row.
- `compare_spectrograms.png` — 3×2, rows = signals, cols = clean vs noisy spectrogram, shared dB scale per row.

## Latest achieved metrics (SNR target = −10 dB)

| | RMS clean | RMS total noise | Achieved SNR | Clipped |
|---|---|---|---|---|
| ECG | 0.362 mV | 1.146 mV | −10.00 dB | yes |
| Vibration | 0.0735 g | 0.233 g | −10.00 dB | yes |
| Audio | 0.122 | 0.386 | −10.00 dB | yes |

## How to change the contamination level

Edit `SNR_TARGET_DB` in `signal_contamination.py`:

- `0.0` — noise power equals signal (noticeable but feature-preserving)
- `−3.0` — 2× noise power, signal still clearly visible in time plot
- `−10.0` (current) — 10× noise power, signal mostly buried, clipping kicks in
- `−20.0` — ~100× noise, signal basically invisible in time domain

No other knob needs to change — every component scales together.
