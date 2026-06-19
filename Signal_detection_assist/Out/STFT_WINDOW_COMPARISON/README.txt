STFT window-size comparison
===========================

Source files: jazz_01.wav, metal_01.wav, pop_01.wav (sr = 22050 Hz, 30 s, mono).
Fixed parameters: Hann window, 75% overlap ratio, magnitude in dB rel. per-plot max, floor -100 dB.
Variable: nperseg (window length in samples).

Settings tested (at 22050 Hz):
  nperseg =   256  ->  window =   11.6 ms  |  df =   86.13 Hz/bin  |  hop =   2.9 ms
  nperseg =  1024  ->  window =   46.4 ms  |  df =   21.53 Hz/bin  |  hop =  11.6 ms
  nperseg =  2048  ->  window =   92.9 ms  |  df =   10.77 Hz/bin  |  hop =  23.2 ms (DEFAULT in plot_stft.py)
  nperseg =  4096  ->  window =  185.8 ms  |  df =    5.38 Hz/bin  |  hop =  46.4 ms
  nperseg =  8192  ->  window =  371.5 ms  |  df =    2.69 Hz/bin  |  hop =  92.9 ms
  nperseg = 16384  ->  window =  743.0 ms  |  df =    1.35 Hz/bin  |  hop = 185.8 ms

How to read the results
-----------------------
STFT imposes a time-frequency trade-off (Heisenberg-Gabor): df * dt = constant.
Doubling the window halves df (finer frequency) and doubles the frame length
(coarser time localization of transients).

nperseg = 256  (~11.6 ms, df ~86 Hz)
  + Sharp percussion attacks (drum hits, plucks) appear as thin vertical lines.
  + Best time resolution: good for onset detection.
  - Bass lines and low harmonics are smeared together (df > a semitone below ~1.5 kHz).
  - Very poor frequency separation in the low end.

nperseg = 1024 (~46 ms, df ~22 Hz)
  + Better frequency detail: melodic lines start becoming visible as horizontal bands.
  + Still catches most percussive transients.
  - Low-frequency harmonics (< 100 Hz) still blurred.

nperseg = 2048 (~93 ms, df ~11 Hz)  [DEFAULT, librosa's default]
  + Balanced: harmonics and transients both reasonably visible.
  + This is the MIR canonical setting for 22 kHz audio.
  * Recommended starting point for general analysis and MFCC/mel-spec features.

nperseg = 4096 (~186 ms, df ~5.4 Hz)
  + Harmonic stacks (vocals, sustained notes) become very sharp.
  + Good for pitch analysis, tonal content, sustained instruments.
  - Transient smearing: kick/snare become horizontal streaks, losing temporal precision.

nperseg = 8192 (~372 ms, df ~2.7 Hz)
  + Semitone-level frequency resolution (1 semitone ~ 6% of freq -> < 3 Hz only below ~45 Hz).
  + Very clean harmonic series visible for sustained tones.
  - Poor time localization: a 93 ms drum transient is spread over 3-4 frames.

nperseg = 16384 (~743 ms, df ~1.35 Hz)
  + Extreme frequency resolution; useful for analyzing stationary tones.
  - Destroys time structure: groove/rhythm unreadable.
  - Overkill for genre features; only useful for long sustained content.

Genre-specific implications
---------------------------
jazz   : sustained brass/piano harmonics benefit from larger windows (2048-4096);
         ride-cymbal texture needs smaller (~1024) to stay crisp.
metal  : fast double-kick and palm-muted chugs demand smaller windows (1024-2048)
         to keep transients resolved. Distortion harmonics are broadband anyway.
pop    : vocal pitch and kick transients both matter -> default 2048 is the safest.

Recommendation
--------------
Keep nperseg = 2048 as the default for subsequent denoising/filtering work.
If the next step is pitch-oriented (tonal separation), bump to 4096.
If transient preservation matters (e.g. transient-protecting noise gate), drop to 1024.