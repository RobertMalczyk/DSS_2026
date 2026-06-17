<div align="center">

# Hybrid GenAI + DSP for Embedded Signal Pipelines

### When does domain knowledge make AI a *strategic collaborator* instead of a micromanaged code generator?

**A four-project existence proof, presented at DSS 2026**

[![Talk](https://img.shields.io/badge/DSS-2026-0E7A99)](https://github.com/RobertMalczyk/DSS_2026)
[![Topic](https://img.shields.io/badge/topic-GenAI%20%2B%20DSP-156082)](#the-thesis)
[![Domain](https://img.shields.io/badge/domain-embedded%20audio-196B24)](#the-pipeline)
![Status](https://img.shields.io/badge/status-conference%20talk-E97132)

</div>

---

## TL;DR

A clean-trained music-genre classifier scores **98.8%**. Drop realistic
instrumentation noise on the input and it collapses to **34.4%** — the *cliff*.
This project is the story of climbing back up it, and the argument it makes is
not "use more AI" but **"use AI where the feedback loop closes, and use domain
knowledge where it doesn't."**

| Approach | What it is | Cliff recovered |
|---|---|---|
| Off-the-shelf AI denoiser | No DSP, no measurement | **9%** |
| Classical filter + AI denoiser | The two stacked | **14%** |
| Measurement-matched filter (Stage 7) | Every notch derived from the noisy audio | **34%** |
| Stage 7 + redesigned feature stack | Domain-informed features (48 → 173) | **70%** |

> Recovery = `(accuracy − 0.344) / (0.988 − 0.344)` — i.e. how much of the lost
> ground was clawed back, measured against the clean ceiling and the noisy floor.

---

## The thesis

LLMs are excellent at **complex but procedural** work — "build the API, the
queue, the dashboard, the tests." Each step has a locally verifiable output, so
the loop closes by itself.

They struggle with **empirical–iterative** work — "separate signal from noise."
Here the quality of step *N* depends on *interpreting* the noisy output of step
*N − 1*: did the filter remove signal along with noise? Is the metric measuring
what we want, or a proxy? Is the win causal or a benchmark artefact?

Without DSP grounding an LLM optimises locally, reaches for ever-heavier tools
(a CNN-on-spectrograms where a notch would do), and has no pruning mechanism.
**Domain knowledge supplies the missing closed loop:** which knob to turn, what
to measure, when to stop, and whether a win is real.

---

## The pipeline

Four sibling projects, each solving one piece of the same end-to-end problem.

| Project | Role |
|---|---|
| **`Signal_generation`** | Builds the dataset and a structured **seven-component noise model** (mains hum, switching tone, broadband EMI, impulsive bursts, thermal floor, 1/f drift, 8-bit quantisation) + ADC clip, at **SNR = −10 dB**. Produces a 250-track music corpus. |
| **`Signal_detection_RAW`** | **Blind** denoising of three opaque signals (ECG / vibration / music). Every filter parameter is measured from the noisy waveform alone. |
| **`Signal_detection_assist`** | Staged filter chain (stages 1–7) on the music corpus. **Stage 7 — "Component-Matched"**: HP @ 25 Hz + detected mains notches + adaptive switching notch + adaptive Wiener, re-discovered per input. |
| **`Training_model`** | RBF-SVM genre classifier (jazz / metal / pop) used as a stress test for everything upstream. Best result: **0.796 (70% cliff recovery)**. |

### The blindness rule

`Signal_detection_RAW` and `Signal_detection_assist` must derive **every** filter
parameter from the noisy input. The clean audio and the noise spec are off-limits
for *designing* a filter — clean is allowed only for *evaluating* one. This wall
is load-bearing: it proves the wins come from **domain reasoning, not oracle
access.**

---

## Key results

The frozen baseline is a 48-dim feature vector (MFCC + spectral descriptors)
into an RBF-SVM, on track-level StratifiedGroupKFold (no track leaks across folds).

**Anchors**

| Condition | Setup | Accuracy |
|---|---|---|
| `A` | clean → clean | **0.988** (ceiling) |
| `B` | noisy → noisy | 0.912 |
| `C` | clean → noisy | **0.344** (the cliff) |

**Test-time denoisers** (clean-trained model, filtered test audio)

| Condition | Filter | Accuracy | Recovery |
|---|---|---|---|
| `D` | off-the-shelf AI denoiser | 0.404 | 9.3% |
| `F` | classical + AI (stacked) | 0.436 | 14.3% |
| `G` | adaptive mean-subtraction | 0.492 | 23.0% |
| `H` | Component-Matched (Stage 7) | 0.560 | 33.5% |

Stage 7 alone wins **+8.5 dB SNR** — but a filter that raises SNR still leaves
**two-thirds of the accuracy gap** open, because the model sees a feature
distribution it was never trained on.

**Closing the rest** — match the training domain to the filter, then redesign the
features around the noise that's left:

| Variant | Change | Accuracy | Recovery |
|---|---|---|---|
| `H_bf4000` | train-side band-pass at 4 kHz | 0.660 | 49.1% |
| `H_v2` | 173-dim domain-informed feature stack | **0.796** | **70.2%** |

> `H_bf4000` (prefilter matching) and `H_v2` (feature redesign) are **two separate
> routes** from the 34% baseline — the feature redesign wins outright; it is not
> stacked on the 49%.

**The counter-result.** Training *on noisy* audio and testing on Stage-7 output
(`M_v2`) scores **0.332 — pure chance.** A noisy-trained model learns to *use the
noise* as part of its feature vocabulary; strip the noise out and it loses the
very features it relied on. Train and test must end up on the same side of the
noise boundary.

---

## What the four projects, together, taught us

1. **Noise has structure.** Modelling it as broadband Gaussian misses what
   actually hurts — mains lines, switching tones, impulsive bursts, slow drift.
2. **Blind filter design works when every choice traces to a measurement.**
3. **Restraint is a filter stage.** No measurement justifies a filter? Don't add
   one. (And the *opposite* band-limit decision can be correct once a model sits
   downstream — same knob, different success criterion.)
4. **Robustness comes from feature design, not just hyperparameter tuning.**
   median + MAD for bursts, Δ-MFCC for stationary noise, chroma + contrast for a
   second axis of discrimination.
5. **Domain knowledge outperforms brute tuning on small data** — every feature
   here traces to one sentence of noise physics or one musical fact, not a grid
   search.

---

## About

Conference talk for **DSS 2026** by **Robert Malczyk**, on combining generative
AI with classical DSP expertise on embedded signal-processing pipelines. The
audience is mixed — embedded engineers and ML practitioners — so the framing
favours intuition over notation.

*All numbers above are re-derivable from the project run logs.*
