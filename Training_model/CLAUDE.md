# Task: Train and compare music-genre classifiers on clean vs noisy audio

You are given an already-built labelled music dataset produced in the sibling
project. Your job is to train classifiers for **genre prediction**, fit them
on **clean** and **noisy** audio, and produce a clear, numeric comparison
showing how much the noise costs in accuracy — and whether training on noisy
data closes that gap.

## Data

The dataset is at:

    C:\Robak\DSS2026\Claude\Signal_generation\out\Model\

Layout:

    out/Model/
      clean/  <genre>_<id>.wav   (250 files, 22050 Hz, ~30 s each, PCM16)
      noisy/  <genre>_<id>.wav   (250 files, same ids as clean)
      labels.csv                 (file_id, genre, clean_path, noisy_path, fs, …)

Genres present: **jazz (84), metal (83), pop (83)** — total 250 tracks.
Noisy files share the same composite EMC + measurement-noise model used in
`Signal_generation` (mains hum + switching + broadband EMI + impulsive bursts
+ 1/f drift + 8-bit quantisation, target SNR = −10 dB, ±3×peak ADC clip).
Each noisy file is an independent noise realisation of the same statistical
model (seed = `sha256(file_id)`), so clean and noisy versions are paired.

250 full tracks is workable but still small. **Split each 30 s track into
shorter windows** (e.g. 3 s) so the effective sample count multiplies by ~10×
before training. The train / test split must be **by track**, not by window —
never let windows of the same track appear in both splits (genre leakage).

## Objectives

1. Read `labels.csv`, load every clean and noisy WAV, split each into
   fixed-length windows (suggested 3 s, 50 % overlap).
2. Produce two aligned feature matrices from the same windows:
   `X_clean`, `X_noisy`, `y`, `track_id`.
3. Choose a feature representation — either
   - classical: per-window MFCC statistics (mean + std of ~20 MFCCs + spectral
     centroid / rolloff / zero-crossing rate), giving a fixed-length vector per
     window → fast to train with `sklearn`, CPU-only;
   - deep: log-mel spectrogram + small CNN (e.g. 2–3 conv blocks).
   Start with the classical path because the dataset is tiny and training
   has to be fast on CPU. Upgrade only if accuracy is clearly limited by the
   model, not by the data.
4. Use a stratified **GroupKFold** (groups = `track_id`, k = 5) so no track
   bleeds between folds. Report cross-validated accuracy, not a single split.
5. Run two matched experiments:

       | experiment | train on | test on |
       | A (clean)  | clean    | clean   |
       | B (noisy)  | noisy    | noisy   |

   Both use the **same track-level splits** (same GroupKFold folds, same
   track_ids in each fold), just swapping the audio source. That makes the
   accuracy difference directly attributable to the contamination.
   Expected: A > B; the gap is the cost of the noise model.

6. Reporting, for each of the two experiments:
   - mean ± std accuracy across folds,
   - per-genre accuracy,
   - a 3×3 confusion matrix (averaged over folds),
   - aggregation method (window-vote vs soft-probability average) used to
     produce the per-track prediction.

## Implementation requirements

1. Use Python 3.12 on this machine. Python is **not on PATH** — invoke as:

       "/c/Users/robak/AppData/Local/Programs/Python/Python312/python.exe" script.py

2. Available packages already installed (from the sibling project): numpy,
   scipy, matplotlib, pandas, scikit-learn, librosa, soundfile, numba.
   Prefer these before installing anything new. If you do need a new package
   (e.g. `xgboost`, `torch`), ask first.
3. Fix a random seed (42) everywhere that has one.
4. Cache feature matrices to disk (e.g. `.npy`) so re-runs don't repeat the
   librosa feature extraction.
5. Do **not** modify or re-download the dataset — read only.

## Output requirements

Produce:

1. **`train_compare.py`** (or similarly named script) that:
   - loads the data,
   - extracts features,
   - runs both experiments (clean→clean and noisy→noisy) under the same
     GroupKFold folds,
   - saves numeric results and plots.
2. **`results/accuracy_table.csv`** — one row per experiment
   (`clean_vs_clean`, `noisy_vs_noisy`) with mean_acc, std_acc, and per-genre
   accuracies.
3. **`results/confusion_clean.png`** and **`results/confusion_noisy.png`** —
   confusion-matrix heat-maps for the two experiments.
4. **`results/summary.md`** — short human-readable write-up:
   - what features and classifier were used,
   - accuracy of clean-vs-clean and noisy-vs-noisy,
   - the absolute and relative accuracy drop caused by the noise,
   - which genres suffered most.

## Output style

- Be practical and concise.
- State assumptions explicitly (window length, overlap, feature list,
  classifier choice, aggregation rule).
- Do not hide implementation choices — put the hyperparameters at the top of
  the script.
- Prefer a small, fast, reproducible pipeline over a bigger one that might
  overfit 50 tracks.

## Stretch goals (only if time allows, and only after both experiments are done)

- Add a small CNN on log-mel spectrograms and repeat the two experiments
  with it.
- Evaluate at several noise levels by re-running the noise generator at
  different SNRs (the generator lives in
  `..\Signal_generation\signal_contamination.py`) and plotting accuracy vs
  SNR for the noisy→noisy case.
