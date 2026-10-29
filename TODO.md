# KB build — CS224R

Open work only. Every run reads this file in full before doing anything and does
only unchecked work. Tick a box the moment its artifact is written, and commit
this file with it. When a run is committed and pushed, move its section verbatim
to [`BUILD_LOG.md`](BUILD_LOG.md): nothing ticked stays here.

## Course backlog

Settings for every run unless the user changes them: images on (every figure-bearing
slide), PDFs link-only (gitignored), commit as chaimantec <nantapon@chaimantec.com>.
Catalog id 39921dee-a11a-41e8-9757-5cbea3b36a9d. Decks at
`https://cs224r.stanford.edu/spring_2025/slides/`.

- [ ] 07 Offline RL — video lRDaXnPIzks — `07_cs224r_offline_rl_2025.pdf`
- [ ] 08 Reward Learning — video PDIxDhA9Z6Y — `08_cs224r_reward_learning_2025.pdf`
- [ ] 09 RL for LLMs — video XKLGuwvSKvI — `09_cs224r-2025-rlhf.pdf`
- [ ] 10 RL for LLM Reasoning — video O2VpNnwB4lM — `10_cs224r-rl_for_reasoning_lecture.pdf`
- [ ] 11 Model-Based RL — video PvqyGnOirgA — `11_cs224r_mbrl_2025.pdf`
- [ ] 12 Multi-Task RL — video qNdsI_4AQJw — `12_cs224r_mtrl_gcrl_2025.pdf`
- [ ] 13 Meta RL — video wSiyEpvoGkA — `13_cs224r_metarl_2025.pdf`
- [ ] 14 Exploration — video 4tlSKdi8teU — `14_cs224r_exploration_2025.pdf`
- [ ] 15 Hierarchical RL and IL — video iKWYLSVAtfM — `15_cs224r_hierarchy_2025.pdf`
- [ ] 16 RL for Robots — video rbaWQQLrzl0 — `16_cs224r_autonomy_2025.pdf`
- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 7 — lecture 7 (Offline RL)

Run 7, 2026-10-29. Scope: lecture 7 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 7
- [x] raw/transcripts/original/07-offline-rl.md — verbatim captions (88 paragraphs, ~10,400 words, ~68 min)
- [x] Slide number map: 33 pages, 1:1. The map script reports pages 5 and 28 unnumbered; raw spans show page 5 prints "5" bottom right (0.92, 0.94). Page 28 prints none (label inferred)
- [x] raw/slides/07-offline-rl.md — deck transcribed (Sonnet), `--verify` clean (33 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 11, 22 (ten loss endpoints, five fitted curves) and 31 (table cell by cell, architecture arrows) agree; 14 and 16 ($s_7$ beside $s_3$, not on it; arrow-weight interpretation removed; advantage box not cut off) and 21 (dashed line, underline) small fixes. Printed slips checked on renders of 4, 5, 18, 19, 20, 21, 22
- [x] raw/transcripts/07-offline-rl.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 88/88 markers identical; 0 numbers lost; checker reports 18 gained, all in [Ed:] notes or decided restorations ("+ one" to +1, "e e^ squ" to e^2, ".5" and "0.5 five" to 0.5, "to8" to 8) — a per-paragraph digit diff with [Ed:] notes stripped leaves only those three; 1 of 88 paragraphs outside 0.72–1.10 (51:15 at 1.21, only two inserted [Ed:] notes); 17 student questions and 2 answers marked; parent added an inline [Ed:] note to the agent's "an accurate" to "an inaccurate" restoration (20:16)
- [x] wiki/07-offline-rl.md — all 14 images placed by hand at the argument (embed dry run: nothing missing; 3, 17 and 25 added after verify_kb's distribution table showed the page cited them without an image, which INDEX's images paragraph rules out); check_quotes 48 quotations, 93 citations, 0 unmatched
- [x] Topic pages: offline-rl, implicit-policy-constraints (new); offline-and-online (its "does not cover it yet" line replaced), imitation-learning, value-functions, policy-evaluation, q-learning, actor-critic, replay-buffers, rl-algorithm-families (coverage line now Lectures 1–7), mdp-and-notation, expressive-policy-distributions, bellman-equations and 06-q-learning (forward links) updated
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/07-offline-rl/ — 14 slides rendered (1.4MB)
- [x] Embedded in the slide file (14, script), the lecture page (14, by hand) and topic pages by hand with "Lecture 7, slide N" captions (offline-rl 11, 14, 26; implicit-policy-constraints 16, 22, 23); 14 hand-written captions on the lecture page
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (and Layer 2's "Lecture 1–6 decks" now "Lecture 1–7"; the deck's text layer has no licence notice, only "Slide adapted from Sergey Levine" on 25–28)

### Publish
- [x] kb.json — coverage 7 of 19, topic pages 19, images 104, figure audit 7 of 7, caveats (Lecture 7's coverage and skipped CQL section, live board work, the printed Monte Carlo index, the worked example's average; its [Ed:] count)
- [x] BUILD_LOG.md — lecture 7's record (the Findings below move with this section); AGENTS.md gains one general line: a lecture page shows every rendered slide it cites
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 104 and 14 of 33 checked; distribution 14 on disk, 14 in wiki); check_math --github: 2949 formulas, 0 errors, 0 warnings (slide file: 9 formulas opened straight after a quotation mark fixed first); check_quotes 389 quotations, 971 citations, 0 unmatched; check_links 865 links, 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 7 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** the recording does not name the speaker, and the deck prints no name. The site's
  schedule lists no guest for this lecture (sources.md), so the wiki writes "Finn" as for Lectures 1–6.
- **Licence:** none stated on the site or in the deck (text layer searched). Slides 25–28 print
  "Slide adapted from Sergey Levine". Citations: Peters et al. (REPS) and Rawlik et al. ("psi-learning")
  on 16; Peng, Kumar, Zhang, Levine, Advantage-Weighted Regression ('19) on 18–19; Kostrikov, Nair,
  Levine, Implicit Q-Learning (ICLR '22) on 21–23, the only credit on slide 22's two charts; Prabhakar,
  Yuan, Yang, Sun, Muralidharan ('22) on 31's diagram and table; Hansen-Estruch et al., IDQL (2023) on 32.
- **Deck:** 33 pages, printed number = page, 1:1. The map script reported pages 5 and 28 unnumbered;
  the raw spans show page 5 prints "5" bottom right (0.92, 0.94), and page 28 prints none (label
  inferred). `--verify` passes.
- **Recording coverage:** slides 3–23 and 30–32, 0:05 to about 1:07:51. The conservative-methods part
  (slides 24–28, CQL) is not taught: the lecture goes from IQL (slide 23) to the summary (≈1:05:55), and
  slide 31's LinkedIn example is passed over as an application "for the algorithm we didn't cover"
  (≈1:06:42). Slides 2 and 33 are not discussed. Much of the lecture is written or drawn live (the
  recap's targets, the formal setting, the car's randomly initialized Q-function, the stitching
  example, the filtered and weighted objectives, AWR's advantages, the histogram and asymmetric loss);
  the wiki reconstructs it from the spoken words and says so.
- **Figure audit:** Opus parent, six 1400px renders (11, 14, 16, 21, 22, 31) against the Sonnet
  transcription. 11, 22 (all ten loss-curve endpoints, five fitted curves) and 31 (every table cell,
  the architecture's arrows) agree. Corrections: on 14 and 16, $s_7$ is drawn beside $s_3$, touching but
  not on it (the lecturer says so at ≈36:29); on 16, the agent's reading that the thicker arrow "stands
  for a larger advantage weight" was an interpretation the slide does not make, and removed; on 16, the
  agent's claim that the advantage box is "cut off" is wrong (the phrase "how much better a_t is" is
  complete); on 21, the second dashed line runs through the last bar, and the blue underline sits under
  $Q(\mathbf{s}', \mathbf{a}')$ only. Slide 6's brace line contradicted itself (items 2–3 or all three);
  the agent's coordinate check says 2–3, and the file now says so.
- **Printed slips (checked on the page):** slides 4, 18 and 19 print the Monte Carlo target as
  $\sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$, index unused (renders of 4, 18, 19), noted in the slide
  file, the wiki and policy-evaluation; slide 5's step 4 prints $\hat{Q}^\pi$ without $\phi$; slide 21
  writes the asymmetric loss $\ell_2^\tau$, slides 22–23 $\ell_2^\lambda$; slide 20 strikes through
  $\mathbf{a}' \sim \pi(\cdot \mid \mathbf{s})$ (conditioned on s, not s') and prints literal asterisks
  around "not"; slides 3–11 write $\mathbb{E}$, $\mathcal{D}$ and slides 16–28 plain $E$, $D$. Recorded from
  the agent's list without a page check, so not claimed as slips: slide 21's AWAC update and slide 23's
  fit-V step without parameter subscripts.
- **Transcript edit:** Sonnet, given the parent's glossary (PPO/SAC, Q-function/Q-value garbles,
  out-of-distribution, KL, Monte Carlo, GAE, AWR, AWAC, expectile, implicit Q-learning, filtered behavior
  cloning, the worked example's digits, the λ value). 88/88 markers; 0 numbers lost; the checker's 18
  gains are [Ed:] note digits and three decided restorations ("+ one" to +1, "e e^ squ" to e^2, ".5" to
  0.5). 1 of 88 paragraphs outside the band (51:15 at 1.21, two [Ed:] notes). 17 student questions, 2
  answers. 8 [Ed:] notes in 7 paragraphs: the captions are much cleaner than Lecture 6's. Agent
  restorations outside the glossary: "an accurate" to "an inaccurate" Q-function (20:16, parent added
  an inline [Ed:] note), "a good off method" to "a good offline method" (28:47).
- **Unclear in the recording, said so on the wiki:** the λ value "0 2 or 3" (≈1:03:37, perhaps 0.2 or
  0.3; the slide says only below 0.5); the worked example (≈51:15), whose returns as captioned (+10, +1,
  −5) average to 2, not the 3 the lecturer uses, with the third advantage captioned "8".
- **Images:** rendered 14 (3, 4, 5, 11, 14, 16, 17, 19, 21, 22, 23, 25, 26, 31), 1.4MB: the ten
  figure-bearing slides and four algorithm and objective slides for their structure (5, 19, 23, 26),
  following Lectures 5–6. Not rendered: 8, 10, 15, 18, 20, 27, 28 (equation slides with transcribed
  underlines, labels, strikethrough, loop arrows and circles); 1, 2, 6, 7, 9, 12, 13, 24, 29, 30, 32, 33
  (text).
- **Embedding:** the script placed 14 under their slide-file headings. Lecture page by hand. It first
  left 3, 17 and 25 (near-duplicates of 4, 16 and 11) to the slide file. The embed dry run reported them
  and verify_kb's distribution table showed 11 of 14, and INDEX tells the chat a cited rendered slide is
  shown in the passage, so they were added. Topic pages by hand with "Lecture 7, slide N" captions:
  offline-rl 11, 14, 26; implicit-policy-constraints 16, 22, 23.
- **Math:** the chat check passed the slide file; the GitHub check found 9 inline formulas opened
  straight after a quotation mark (the transcriber quoted slide labels with math inside, despite the
  rule in AGENTS.md); rewritten with the math outside the quotes.
