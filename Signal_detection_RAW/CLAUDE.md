# Task: Blind denoising of three unknown signals

You are handed three noisy digital signals. Design a denoising pipeline that
removes as much of the unwanted content as possible and present a clean
before/after comparison. You do not know what the signals represent, where
they came from, or what kinds of noise they contain. Everything you need
must come from inspecting the waveforms themselves.

## Inputs — the only files you may read

| Path | Format |
|---|---|
| `..\Signal_generation\out\Samples\ecg_noisy.wav` | 32-bit float WAV |
| `..\Signal_generation\out\Samples\vibration_noisy.wav` | 32-bit float WAV |
| `..\Signal_generation\out\Samples\music_noisy.wav` | 32-bit float WAV |
| `..\Signal_generation\out\Samples\samples_info.txt` | filename / fs / duration table |

Load with `soundfile`:

    import soundfile as sf
    noisy, fs = sf.read(path, dtype="float64")

The filenames are labels for organising your output — treat the signal
content itself as opaque. Do not assume anything about what each signal
represents beyond what the waveform tells you.

## Hard constraint — you are blind

- The denoising function must accept only `(noisy, fs)` and return the
  filtered array. No side-channel inputs.
- Everything outside the four paths listed above is out of scope for reading.
  In particular, all of the following are off-limits:
    - anything under `..\Signal_generation\out\Samples\Clean\`
    - any other file inside `..\Signal_generation\` (code, markdown, PNG
      plots, summary.txt, dataset caches)
    - sibling projects (`Training_model\`, `Signal_detection_assist\`, etc.)
    - any remote dataset, on-disk dataset cache, or file outside the four
      paths listed above
- Do not import modules from `..\Signal_generation\`, do not run any
  generator/loader function you might find there, do not regenerate the
  signals or any variant of them. The three WAVs on disk are the entire
  world.
- Do not hard-code any numeric parameter that you did not measure from the
  noisy input. Every cutoff, threshold, and window length is derived from
  inspection, not assumed.

## Approach (free design)

1. Inspect each signal: time-domain trace, magnitude spectrum / Welch PSD,
   spectrogram, any robust local statistic you find informative. Describe
   what you observe.
2. Based only on those observations, design a filtering chain for each
   signal. Each stage must be justified by a specific feature you measured.
3. Treat the three signals as independent — a pipeline that works for one
   may be wrong for another. There is no requirement that all three use
   the same chain.
4. Prefer zero-phase filtering (`scipy.signal.sosfiltfilt`) where you want
   to preserve waveform shape.

## Output — stays inside this project

All generated artefacts go under `Signal_detection_RAW\out\`. Do not write
anywhere else — not into `..\Signal_generation\`, not into any sibling.

For each of the three signals produce one figure:

- `out\ecg_filtered.png`
- `out\vibration_filtered.png`
- `out\music_filtered.png`

Each figure contains:

- noisy (input) signal in the time domain
- filtered signal in the time domain (same y-axis as the noisy plot)
- Welch PSD of noisy and filtered overlaid on a log y-axis
- spectrogram of the filtered signal

Also write `out\summary.md` with one section per signal listing:

- what you observed during inspection (one short paragraph)
- the filter chain you designed, in order, with the numeric parameters you
  measured or derived for each stage
- reference-free quality indicators: variance reduction ratio, PSD-floor
  reduction in dB, and any other measurement you find informative

Do not report an SNR-vs-clean number — there is no clean reference available.

## Implementation

1. Python 3.12. Not on PATH; invoke directly:

       "/c/Users/robak/AppData/Local/Programs/Python/Python312/python.exe" script.py

2. Packages available: numpy, scipy, matplotlib, soundfile. Prefer
   `scipy.signal` for filter design. Ask before installing anything new.
3. Fix `np.random.seed(42)` if any randomness enters.
4. Factor the pipeline into one function per signal, or one generic
   `denoise(noisy, fs)` — whichever keeps the code clear. Keep the function
   self-contained; do not smuggle in global state or external lookups.

## Style

- Be concise. Justify each filter stage with a measurement from step 1.
- Start simple. Only add complexity when you can point at a specific
  residual problem a simpler pipeline failed to address.
- Document failures — if a filter introduces a worse artefact than it
  removes, say so in `summary.md` and revert.
