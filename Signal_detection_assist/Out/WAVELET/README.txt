Continuous wavelet transform (CWT) scalograms
=============================================

Source files: jazz_01.wav, metal_01.wav, pop_01.wav (sr = 22050 Hz).
Segment analyzed: 10.0 s to 15.0 s (5 s slice).
Frequency grid: 96 log-spaced scales, 40 Hz to 8000 Hz.
Color: |CWT coefficient| in dB, relative to per-plot max, floor -80 dB.

Why only 5 seconds?
  CWT is O(N * num_scales). At 22 050 Hz, a 30 s clip with 96 scales takes
  significantly longer than STFT. A 5 s central slice is enough to judge
  time-frequency trade-offs and stylistic differences between wavelets.

CWT vs STFT - what is different
-------------------------------
STFT uses a fixed window -> constant df at all frequencies.
CWT scales the mother wavelet -> constant Q: resolution is FINE IN TIME at
high frequencies (short wavelet) and FINE IN FREQUENCY at low frequencies
(long wavelet). This matches how music is structured: bass notes hold,
cymbals and transients are fast. Low-frequency bass lines become much more
readable than in a linear-frequency STFT.


Figure 1: compare_wavelets_<file>.png  - 5 mother wavelets
----------------------------------------------------------

cmor1.5-1.0  (complex Morlet, bandwidth=1.5, center=1.0)
  The de-facto standard for music CWT. Complex-valued -> magnitude is
  phase-invariant and smooth. Good balance between time and freq resolution.
  Horizontal harmonic lines appear clean; transients stay compact.

morl  (real Morlet)
  Same carrier as cmor but real-valued. Magnitude shows interference
  stripes when harmonics beat -> looks noisier. Useful for edge/phase
  studies, worse for spectrogram reading.

mexh  (Mexican hat / Ricker)
  Second derivative of a Gaussian. No oscillation inside the envelope,
  so it is poor for resolving harmonics but excellent for detecting sharp
  events (onsets, glitches). Expect blurred harmonic bands and punchy
  transient columns.

gaus4  (4th derivative of Gaussian, real)
  A few oscillations in the envelope -> partway between mexh and morl.
  Real-valued, so it also shows interference patterns. Sharper than
  mexh at isolating harmonics but less clean than cmor.

cgau4  (complex 4th Gaussian derivative)
  Analytic (complex) counterpart of gaus4. Magnitude is smooth like cmor,
  so this is a good side-by-side to morl vs cmor: same trade-off but with
  Gaussian derivatives.


Figure 2: compare_cmor_<file>.png  - cmor bandwidth sweep
---------------------------------------------------------

The complex Morlet is parameterized cmorB-C:
  B = bandwidth (controls time support / frequency resolution)
  C = center frequency (normalized; kept at 1.0 throughout)

Effect of B (time-frequency Heisenberg trade-off):
  B = 0.5  -> short wavelet in time, broad in frequency.
             Transients sharp; harmonics bleed into wide bands.
  B = 1.5  -> balanced (standard setting).
  B = 3.0  -> longer wavelet; harmonic stacks become very tight,
             transient edges soften.
  B = 6.0  -> extreme frequency resolution; near pure-tone analysis,
             time structure is lost (single drum hit smears across
             many frames).

Genre reading tips
------------------
jazz   : cmor3.0-1.0 reveals brass/piano harmonic stacks nicely;
         cmor1.5 still shows cymbals crisply.
metal  : cmor1.5 preserves kick/snare attacks needed to resolve double-bass;
         mexh highlights every transient.
pop    : cmor1.5 is the safest default; low B (0.5) exposes sibilance
         and kick transients; high B (>=3) emphasizes sustained vocals.

Bottom line
-----------
For the subsequent genre-preserving denoising, cmor1.5-1.0 is the
recommended scalogram. If you want to emphasize tonal content, use
cmor3.0-1.0; for onset/transient emphasis, cmor0.5-1.0 or mexh.