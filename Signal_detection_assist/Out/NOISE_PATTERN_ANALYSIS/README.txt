NOISE PATTERN ANALYSIS — read from noisy music files only (non-oracle)
========================================================================

Sample: 8 files x 3 genres = 24 files.
Method: low-energy frame (bottom 10% by RMS) PSD pooling, per-file unit-sum normalisation, median across files.

----- TOP GLOBAL SPECTRAL LINES (pooled across all files) -----
freq_Hz   | PSD_dB   | prominence_over_continuum_dB
    51.1  |   -9.69  |  21.99
    99.6  |  -14.73  |  15.43
  4417.0  |  -19.05  |  15.27
  4403.5  |  -19.01  |  15.25
   150.7  |  -19.35  |  14.30
   199.2  |  -24.80  |  10.48
   293.4  |  -32.72  |   4.90
   263.8  |  -31.90  |   4.85
   498.0  |  -37.15  |   3.09
   393.0  |  -36.69  |   2.94
   247.6  |  -33.49  |   2.91
   444.1  |  -37.72  |   2.76
   702.5  |  -36.28  |   2.70

----- IMPULSIVE BURST ESTIMATE (envelope > median + 8·MAD) -----
mean rate across files          : 0.18 /s (spec says 3 /s)
median event duration           : 0.0 ms (spec says ~20 ms incl. tail)

----- LOW-FREQUENCY DRIFT -----
mean fraction of noise power    : 8.7% below 30 Hz

----- SANITY CROSS-CHECK vs noise.md -----
noise.md says the composite noise contains:
  1. mains 50 Hz + harmonics {100, 150, 200}
  2. switching tone at 0.4 * Nyquist (= 4410 Hz @ fs=22050)
  3. broadband EMI, bandpassed [0.05, 0.95]*Nyquist
  4. impulsive bursts at ~3 /s
  5. 1/f^2 drift (dominant low-frequency)
  6. 8-bit quantisation (negligible)
  7. +/- 3*peak clipping

Look for these in the plots and the peak list above. Any 
narrowband line that lines up with 50,100,150,200 Hz (or 
~4410 Hz) confirms component 1/2 and justifies a notch.