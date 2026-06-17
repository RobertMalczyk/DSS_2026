# Blind denoising summary

Three noisy WAVs were inspected (time / Welch PSD / spectrogram) and a
per-signal zero-phase filter chain was designed from the measured features.
Every cutoff, notch centre, Q, and order below was derived from the
inspection pass — nothing was assumed about what the three signals carry.
All filters are applied with `scipy.signal.sosfiltfilt` (zero-phase).


## ecg   (fs = 360 Hz, duration = 10.00 s)

### Observations
Low sample rate (360 Hz, 10 s). Strong negative DC (mean ≈ −1.11, σ ≈ 0.83). Welch PSD shows the signal band collapses into the noise floor above ~14 Hz (smoothed-PSD knee); three sharp tonals at 50 Hz (+23 dB over local floor), 72 Hz (+17 dB), 100 Hz (+17 dB); flat broadband floor around −30 dB. The signal of interest therefore occupies roughly 0.5–15 Hz.

### Filter chain
HP Butter ord 4, 0.5 Hz (zero-phase, `sosfiltfilt`)  
  — kills DC offset (measured mean = −1.107) and any baseline wander.

LP Butter ord 6, 30 Hz (zero-phase)  
  — measured PSD knee at ~14 Hz; 30 Hz leaves a comfort margin over that knee
  and lies well below the interference lines at 50/72/100 Hz. Effective order 12
  under `sosfiltfilt` gives >80 dB attenuation at 50 Hz, so no dedicated notch
  is needed.

### Reference-free quality indicators
- variance reduction ratio: **10.54×**  (var noisy = 6.870e-01, filtered = 6.518e-02)
- PSD floor in upper 40% of band: -31.1 dB → -189.0 dB  (**drop = 157.9 dB**)
- PSD median across full band: -30.0 dB → -166.4 dB
- DC offset: -1.107e+00 → +1.080e-02

Figure: `out/ecg_filtered.png`

## vibration   (fs = 12000 Hz, duration = 3.00 s)

### Observations
fs = 12 kHz, 3 s. Small DC (mean ≈ −0.14). Low-band energy elevated up to ~200 Hz (knee). Three mains harmonics at 50/100/150 Hz stand ~28/22/17 dB above the local floor. Two strong tonal clusters at 1035/1066 Hz and around 2400 Hz (≈21–25 dB above floor), plus weaker lines at 358 Hz and 2102 Hz — these look like the actual vibration signal. Above ~2500 Hz the PSD is uniform floor at about −58 dB, indicating nothing but broadband noise sits there.

### Filter chain
HP Butter ord 4, 5 Hz (zero-phase)  
  — removes DC (measured mean = −0.144) while keeping sub-50 Hz low-band
  structure intact for measurement.

IIR notch 50 Hz, Q=30  (BW ≈ 1.7 Hz)  
IIR notch 100 Hz, Q=30  (BW ≈ 3.3 Hz)  
IIR notch 150 Hz, Q=30  (BW ≈ 5 Hz)  
  — the three tonals were measured at +28/+22/+17 dB above local floor;
  they track the 50 Hz mains family exactly. Q=30 is narrow enough not to
  touch the 358 Hz / 1 kHz / 2.4 kHz content that looks like the real signal.

LP Butter ord 8, 2800 Hz (zero-phase)  
  — PSD above ~2500 Hz is uniform broadband floor at about −58 dB; no
  tonal or structural content measured there, so everything above 2.8 kHz
  is noise that can be removed without risk.

### Reference-free quality indicators
- variance reduction ratio: **3.54×**  (var noisy = 3.541e-02, filtered = 1.001e-02)
- PSD floor in upper 40% of band: -59.3 dB → -228.4 dB  (**drop = 169.1 dB**)
- PSD median across full band: -58.8 dB → -74.9 dB
- DC offset: -1.438e-01 → -1.590e-04

Figure: `out/vibration_filtered.png`

## music   (fs = 22050 Hz, duration = 4.00 s)

### Observations
fs = 22.05 kHz, 4 s. Small DC (mean ≈ −0.14), with PSD climbing steeply below ~20 Hz (rumble). Broadband, time-varying content (typical music) from the bass up to Nyquist. Mains tonals at 50 and 100 Hz stick out by 17 / 14 dB, and a very narrow pair of peaks at 4403 / 4417 Hz (14 Hz apart, ~20 dB over local floor) is too sharp to be musical content and looks like a narrowband interferer.

### Filter chain
HP Butter ord 4, 25 Hz (zero-phase)  
  — PSD climbs steeply below ~20 Hz (rumble / DC, far below any plausible
  audio fundamental). 25 Hz cutoff removes the rumble without eating into
  the audible bass band.

IIR notch 50 Hz, Q=30  
IIR notch 100 Hz, Q=30  
  — measured at +17 / +14 dB above local floor; identified as mains hum
  family.

IIR notch 4410 Hz, Q=110  (BW ≈ 40 Hz)  
  — measured: a tight pair of peaks at 4403 and 4417 Hz (14 Hz apart),
  each ~20 dB above local floor. Too narrow to be musical content (sidebands
  right next to each other), most likely a narrowband interferer. Q=110
  straddles both peaks in one stage.

No LP stage — music carries real content right up to Nyquist.

### Reference-free quality indicators
- variance reduction ratio: **2.96×**  (var noisy = 1.258e-01, filtered = 4.251e-02)
- PSD floor in upper 40% of band: -55.8 dB → -55.8 dB  (**drop = 0.0 dB**)
- PSD median across full band: -55.2 dB → -55.2 dB
- DC offset: -1.406e-01 → +6.983e-05

Figure: `out/music_filtered.png`

## Notes on what was deliberately left out

- **No broadband LP for music.** The music PSD is not flat above some
  knee — there is time-varying content all the way to Nyquist in the
  spectrogram, so there is no measurement-based justification for a
  lowpass. Only the narrow 4410 Hz interferer is removed.
- **Mains notches only on measured harmonics.** 150 Hz is at floor in
  music (so not notched there); ECG's 30 Hz LP removes 50/72/100 Hz
  without needing dedicated notches.
- **No tonal notches in vibration.** The 1 kHz and 2.4 kHz clusters are
  ~21–25 dB above the floor and look like the actual vibration content —
  if this were machinery, those would be resonances. Notching them would
  destroy the signal, so they are preserved.