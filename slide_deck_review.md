# Slide deck review — Robert_Malczyk.pptx

Latest read: **2026-05-07** (25 slides, file mtime 2026-05-07 17:22).
Previous read: 2026-04-30 (20 slides). Sorted by impact, not by slide order.

## What changed since 2026-04-30

| Done | What |
|---|---|
| ✅ | `presentation_problem_diagram.png` dropped into slide 3 |
| ✅ | Slides 4, 5 filled with figure images (no longer empty) |
| ✅ | `presentation_denoising_effect.png` dropped into slide 16 |
| ✅ | New embedded-hardware slide inserted as **#19** ("Smarter beats bigger — and fits in flash") closing the conference.txt gap on latency / memory / hardware cost |
| ✅ | Takeaway slide (now **#20**) gained a fourth item, "Hardware-aware by design" |
| ✅ | Appendix slides added at the end: project structure (#22), repo tree (#23), evidence base (#25) |
| ✅ | Aliasing dropped from conference.txt goals — no longer a coverage gap |

## Conference.txt coverage (current)

| Topic | Slide(s) |
|---|---|
| GenAI mistakes in filtering / windowing / FFT–STFT–CWT params | 8, 9, 11 |
| Hybrid GenAI + DSP workflow | 12–19 |
| API cost blow-up, no termination, locally vs globally optimal | 11 |
| Domain knowledge → strategic decisions | 12, 13, 19, 20 |
| Embedded hardware: latency, memory, robustness | **19 (new)** |
| Lower hardware costs / energy efficiency | **19 (new)** |
| Faster prototyping / time-to-market | implicit in slide-20 tagline only |

The deck now matches conference.txt on every named topic. "Faster prototyping" is the only thing carried implicitly rather than explicitly.

---

## Outstanding issues, ranked

### 1. Slide 19 table — column mismatch
The new embedded-hardware slide is doing the right job, but the table row for "v2 features + RBF-SVM (ours)" puts **"~250 KB total"** in the **Parameters** column and **"Fits $5 MCU"** in the **Peak memory** column. The CNN rows below it use those columns the standard way (params count vs activation footprint).

**Fix:** keep one consistent semantics across rows. Two clean options:

| | MACs / inference | Parameters | Peak memory |
|---|---|---|---|
| v2 + RBF-SVM (ours) | ~13 M | ~300 SVs · 173 dims | ~250 KB |

…with the "$5 MCU" verdict moved into the bottom strap or a "Hardware class" 4th column. Or:

| | MACs / inference | Total memory | Hardware class |
|---|---|---|---|
| v2 + RBF-SVM | ~13 M | ~250 KB | $5 MCU |
| Small 3-block CNN | ~150 M | ~3 MB | $30 MCU+NPU |
| ResNet-18 | ~1.8 G | ~54 MB | $100+ Jetson |

If the audience is mixed embedded + ML, the "Hardware class" column carries the talk's punchline better than splitting params from peak memory.

### 2. Slide 19 strap — "~8× less memory" undercount
The bottom strap reads "~8× less memory." The arithmetic from the table itself is **~12×** (small CNN total = 1 MB params + 2 MB activations = 3 MB; v2 = 250 KB; 3 MB / 250 KB = 12×). The earlier figure draft (`presentation_embedded_compute.png`) annotates "12× memory" for this same reason.

**Fix:** change to "~12× less memory" or, for a defensible round number, "an order of magnitude less memory."

### 3. Slide 6 + slide 7 — still both titled "The experiment"
Two consecutive slides with the same title and only an image each. Flagged on 2026-04-30, still unresolved. Pick one purpose for each:
- **Slide 6 → "The dataset"** (250 tracks, jazz/metal/pop, SNR = −10 dB) using `presentation_music_contamination.png` (already at root, still unreferenced anywhere in deck).
- **Slide 7 → "The model"** (RBF-SVM, frozen across A–H, no leakage) — or delete if redundant with slide 10.

### 4. Slide 8 ↔ slide 9 — same "no universal default" point twice
Slide 8 (STFT window sweep) and slide 9 (CMOR bandwidth sweep) both make the "AI picks one default, the expert sweeps and chooses" point. Audience gets the lesson on slide 8; slide 9 is reinforcement.

**Fix (still pending):** merge into one slide ("Same tool, no universal default — windows AND bandwidths") or cut slide 9 and reclaim a slot for `presentation_metal_collapse.png` (still unused).

### 5. `presentation_metal_collapse.png` still unreferenced
The per-genre table showing metal = 0.00 in 4 of 8 conditions is one of the strongest pieces of evidence for *why* the v2 features matter. Without it, the audience hears "70 % recovery" on slide 18 but never sees the asymmetric collapse it fixes.

**Slot:** between slides 17 and 18, or as an inset/callout inside slide 18.

### 6. Slide 20 — four takeaways may break the rhythm
"What to take home" used to be a clean rule-of-three close. With #4 added, it's now four. Three options, in ascending impact:

a. **Leave it** — four is fine if visual density still works. Verify in slideshow mode.
b. **Re-merge into three** by folding the embedded angle into #3, e.g. *"Smarter beats bigger — 70 % recovery in ~25 MFLOPs and ~250 KB, fits a $5 MCU."* (This was Option A from the 2026-05-05 conversation.)
c. **Promote #4 to its own micro-slide** before #20, leaving #20 as the rule-of-three close. Risks redundancy with the new slide 19, so probably not worth it.

Recommendation: **(a) for now**, revisit if the slide reads cluttered when projected.

### 7. Slide 11 vs slide 3 — possible thesis restatement
Slide 3 is image-only (`presentation_problem_diagram.png` carrying the thesis visually). Slide 11's body still contains *"Each step is locally optimal; the path is not"* — the same line that appears on the slide-3 diagram. Worth verifying in slideshow that the two slides don't echo each other word-for-word; if they do, slide 11 should land a different angle (e.g. concrete API-cost number, not the thesis again).

### 8. Appendix slides 22, 23, 25 ordering vs slide 20
"What to take home" (#20) is meant to be the mic-drop close, but is followed by a divider image (#21), project structure (#22), repo tree (#23), divider (#24), and evidence base (#25). The audience may not know the talk has ended.

**Fix (low effort, high clarity):** insert a slide titled "Appendix" (or repurpose the #21 divider as one) between #20 and #22 so the closing rhythm is:
> takeaway → "Thanks / Questions" → "Appendix" → backup material.

Alternatively: move slide #25 (Evidence base) to **before** slide 20. The literature lands the takeaway with academic backing instead of trailing it.

---

## Density / typography (carryover from 2026-04-30)

- **Slide 2** still very text-heavy — eight paragraphs of three-line bodies. Splitting into two slides (DSP / ML on one, "the seam" on another) or condensing into a side-by-side diagram with one-line bullets remains the cleanest fix.
- **Slide 15** — six-row recovery list is OK but visually inconsistent with slide 18's staircase. Convert to a horizontal staircase like slide 18 for parallel structure across the recovery arc.

## Consistency checks (still good)

| Claim | Slides | Status |
|---|---|---|
| +8.5 dB SNR | 13, 15 | both phrase as "9-track starter set" — consistent |
| 34 % recovery | 15, 18, 20 | consistent |
| 48-dim baseline | 10, 18 | consistent |
| ~13 MMACs / ~250 KB / $5 MCU | 19, 20 | consistent (apart from the "8× memory" note above) |

---

## Quick wins, ranked by leverage

1. **Fix the slide-19 table column mismatch** and update "8× memory" → "12× memory" (or "an order of magnitude"). 5 min in PowerPoint, removes the only bug that an embedded reviewer would flag.
2. **Resolve the slide 6 / 7 duplicate** — give one to "the dataset" and use `presentation_music_contamination.png`. 10 min.
3. **Cut slide 9 or merge into 8** — saves a slot for `presentation_metal_collapse.png`. 15 min.
4. **Add an "Appendix" divider after slide 20** — cleanest fix for the appendix-after-takeaway problem. 2 min.
5. **Verify slide 3 ↔ slide 11 don't echo verbatim** in slideshow mode. 1 min, may surface a rewrite.

---

## Slide-by-slide one-line map (current state — 25 slides)

| # | Title | Status / note |
|---|---|---|
| 1 | Title slide | OK |
| 2 | DSP and AI/ML — two disciplines, one pipeline | dense — split or compress (carryover) |
| 3 | *(image-only)* — `presentation_problem_diagram.png` | OK; verify no echo with slide 11 |
| 4 | *(image-only)* | confirm content matches the dataset/setup beat |
| 5 | *(image-only)* | confirm content matches the dataset/setup beat |
| 6 | The experiment | **duplicate of 7** — give it content |
| 7 | The experiment | **duplicate of 6** — give it content |
| 8 | AI defaults aren't bad. They're just generic. | keep |
| 9 | Same tool, three answers — which one is right? | merge into 8 or cut |
| 10 | The cliff: 99 % → 34 % | OK |
| 11 | Approach A: what AI ships if you let it | verify no echo with slide 3 |
| 12 | Approach B: design starts with the data | OK |
| 13 | Recovering the noise — without reading the spec | OK |
| 14 | The filter we designed | add stages 1–6 context (carryover) |
| 15 | Filtering after training isn't enough | consider staircase visual (carryover) |
| 16 | Stage 7 in action — noise removed, music preserved | OK |
| 17 | Match training to what the filter outputs | OK |
| 18 | Rebuild the features around what's left | OK |
| 19 | **Smarter beats bigger — and fits in flash** (new) | fix table column mismatch + 8×→12× memory |
| 20 | What to take home | now four takeaways — verify density |
| 21 | *(image divider)* | consider relabelling as "Appendix" |
| 22 | Project structure | OK as appendix |
| 23 | Repository tree | OK as appendix |
| 24 | *(image divider)* | OK as appendix divider |
| 25 | Evidence base | consider moving before #20 to support the takeaway |
