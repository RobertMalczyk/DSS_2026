# DSS 2026 — project root

This tree holds the four sibling projects behind a DSS 2026 conference
talk on combining GenAI with DSP expertise on embedded pipelines.

## Talk plan

Read `conference.txt` first. It contains the talk's thesis (hybrid
GenAI + DSP outperforms either alone) and the key argument the slides
need to make: domain knowledge is what turns AI from a
micromanagement-prone code generator into a strategic collaborator.
Every implementation choice in the four sub-projects below should be
read against that thesis.

## Sibling projects

| Path | Role |
|---|---|
| `Signal_generation/` | Builds the dataset and the seven-component noise model (mains + switching + broadband EMI + bursts + thermal + 1/f drift + 8-bit quant + ADC clip, SNR = -10 dB). Produces the 250-track music corpus under `out/Model/`. |
| `Signal_detection_RAW/` | Blind denoising of three handed-over WAVs (ECG / vibration / music). Only the noisy WAVs are readable. |
| `Signal_detection_assist/` | Staged filter chain on the music dataset (stages 1–7). Stage 7 = "ComponentMatched" — HP25 + detected mains/switching notches + adaptive Wiener. |
| `Training_model/` | RBF-SVM genre classifier (jazz / metal / pop). Cross-conditions A–H plus a 173-dim v2 feature stack. Best result: H_v2 = 0.796 (70 % recovery from the C-cliff). |

Each sub-project has its own `CLAUDE.md` with the rules and outputs
specific to that project.

## Blindness rule between projects

`Signal_detection_RAW` and `Signal_detection_assist` must derive every
filter parameter from the noisy input alone. The clean WAVs and
`Signal_generation/noise.md` are off-limits to those two projects
(clean is allowed only for *evaluating* a filter, never for designing
one). This wall is load-bearing for the talk's credibility — do not
let project boundaries leak across it.

## Narrative artefacts (in this folder)

- `conference.txt` — the talk's thesis and the points the slides
  must make.
- `presentation_story_full.txt` — unified four-project story written
  with full visibility across all sub-projects. Use this as the
  source of truth when generating slide content or figures.
- `Training_model/presentation_story.txt` — original Training_model-only
  log (kept for reference; superseded by the unified story above).
- `DSSAI26-Presentation_Template_EN.pptx` — the conference template.

## Picture generation

`Training_model/` is the workspace for any new figures the talk needs.
It already contains the cached feature matrices and the `train_cross_*`
scripts that produced every accuracy number in the unified story.
