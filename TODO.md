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

- [ ] 11 Model-Based RL — video PvqyGnOirgA — `11_cs224r_mbrl_2025.pdf`
- [ ] 12 Multi-Task RL — video qNdsI_4AQJw — `12_cs224r_mtrl_gcrl_2025.pdf`
- [ ] 13 Meta RL — video wSiyEpvoGkA — `13_cs224r_metarl_2025.pdf`
- [ ] 14 Exploration — video 4tlSKdi8teU — `14_cs224r_exploration_2025.pdf`
- [ ] 15 Hierarchical RL and IL — video iKWYLSVAtfM — `15_cs224r_hierarchy_2025.pdf`
- [ ] 16 RL for Robots — video rbaWQQLrzl0 — `16_cs224r_autonomy_2025.pdf`
- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 11 — lecture 11 (Model-Based RL)

Run 11, 2026-10-29. Scope: lecture 11 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 11
- [x] raw/transcripts/original/11-model-based-rl.md — verbatim captions (95 paragraphs, ~12,100 words, ~73 min)
- [x] Slide number map: 38 pages, printed 1–38, 1:1. The map script reports 3, 4, 7, 8, 35 unnumbered; raw spans show 4, 7, 8, 35 print theirs bottom right, and only page 3 prints none (label follows the 1:1 sequence, unambiguous)
- [x] raw/slides/11-model-based-rl.md — deck transcribed (Sonnet), `--verify` clean (38 headings); recording-coverage paragraph, video notes from the transcript (5, 6, 8, 34, 36) and audit corrections added by the parent; check_math 114 formulas, 0 errors
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 4, 27, 36 (every value of both series) agree; 34's two charts agree series by series, but PETS's Baoding curve runs behind the legend rather than stopping; 12's backward arrows are on every link but the first; 35's CEM curve rises from about 0.4 to 0.55 (had been flat) and random shooting swings 0.3–0.9 (had been 0.1–0.9). Printed oddities checked on renders: slide 32's author line prints "Kurt Konoglie" (kept as printed); 14's literal asterisks; 34's "Nagabandi et. al" in the legends; 28's and 30's non-bold symbols
- [x] raw/transcripts/11-model-based-rl.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 95/95 markers identical; 0 lost, gained 1 ×2 (z t plus 1 at 17:02, s t plus 1 at 19:24, both glossary restorations); 0 of 95 paragraphs outside 0.72–1.10 (overall 0.886); 20 student questions and 3 student answers marked; 6 [Ed:] notes. Parent fixes: the 1:04:23 [Ed:] note moved to the second "h" (the first was spoken as h, the second captioned "eight"); a student-question mark added before the finite-differences restatement at 38:50; the header's vague "28:00" deferral removed
- [x] wiki/11-model-based-rl.md — all 17 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing, after fixing a slide-22 `.png` link to `.jpg`); check_quotes 80 quotations, 146 citations, 0 unmatched; check_math 106 formulas, 0 errors
- [x] Topic pages: model-based-rl and planning-with-models (new); offline-and-online (the whole map so far, planning is off-policy), rl-algorithm-families (model-based row, coverage now Lectures 1–11, the recap), reward-learning (a reward model for planning), value-functions (terminal value function), imitation-learning (distilling a planner), mdp-and-notation (notation for model-based RL), policies (a planner as a "policy"), course-logistics (Homework 3 and milestone dates) updated; 01-class-intro, 04-actor-critic and 10-rl-for-reasoning forward links. check_quotes 73 quotations on the new and updated pages, 0 unmatched; check_math wiki 2353 formulas, 0 errors; check_links 1407 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–11 and 190 images, Lecture 11's recording coverage, lecture entry, two new topic entries and eight updated, SEE_ALSO pointer)

### Images
- [x] raw/images/11-model-based-rl/ — 17 slides rendered (1.9MB); embedded in the slide file (17, script), the lecture page (17, by hand, hand-written captions) and topic pages by hand with "Lecture 11, slide N" captions (offline-and-online 4; model-based-rl 8, 27; planning-with-models 13, 22)
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (`--format auto` mixes extensions, so take hand-placed paths from `ls` and run the embed dry run; a glossary restoration aimed at one of several identical words quotes its neighbours); LICENSE.md attribution (Layer 2 now "Lecture 1–11 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 11 of 19, topic pages 29, images 190, figure audit 11 of 11, caveats (Lecture 11's coverage, board work and covered chart; its [Ed:] count)
- [ ] BUILD_LOG.md — lecture 11's record (Findings move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: Princeton ROB 345 added (trajectory optimization and model predictive control with known dynamics; repo public, four raw links return 200)
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 190 and 17 of 38 checked; distribution 17 on disk, 17 in wiki); its GitHub math check: 4247 formulas, 0 errors; check_links 1407 links, 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 11 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Chelsea Finn (the title slide names only "CS 224R"). Finn says model-based RL has a
  special place in Finn's heart, having worked on it (≈4:36). Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©,
  Creative, rights). Third-party material: Veo 2 and Sora video stills (6), a maze image and a Sora video
  frame (8), a cliff photograph (18, 20), clip-art (21, 22), and the case-study paper's renderings, charts
  and photographs (33–36): Nagabandi, Konoglie (as printed), Levine and Kumar, "Deep Dynamics Models for
  Learning Dexterous Manipulation", Google Brain, September 2019 as printed on slide 32. No per-slide credit.
- **Deck:** 38 pages, printed 1–38, 1:1, numbers bottom right. The map script reports 3, 4, 7, 8, 35
  unnumbered; raw spans show 4, 7, 8, 35 print theirs, and only page 3 (a section divider) prints none.
  `--verify` passes.
- **Recording coverage:** about 73 minutes, slides 4–25 and 32–36. At ≈1:06:40 Finn says "I think I might
  cover this second approach on Friday" and goes to the case study, so slides 26–30 (model-based policy
  optimization's roll-outs and algorithm, when to use model-based RL, other kinds of models) are not
  discussed; Lecture 12's deck (slides 3–9 by its text layer) opens with its own version of them. Slides 2,
  37 and 38 are not discussed beyond the closing "next time" (≈1:12:52). Board work: a latent-state model
  with an encoder (≈15:23–17:49), the gradient-based planning algorithm with its chain rule and update
  (≈23:17–31:07), sketches of gradient-based and sampling-based optimization over actions (≈31:54–35:44),
  and open versus closed loop (≈51:13–52:52). At ≈32:40 Finn corrects a sign ("There should be a plus sign
  because we're maximizing not minimizing"), apparently in the update written at ≈26:28; the wiki writes
  the corrected update and says so.
- **Figure audit:** Opus parent, six 1400px renders (4, 12, 27, 34, 35, 36) against the Sonnet
  transcription, plus a crop of slide 32's author line. 4, 27 and 36 (every value of both series) agree;
  34's two charts agree series by series. Corrections: slide 12's backward arrows are on every link but the
  first; slide 35's CEM curve rises from about 0.4 to 0.55 (had been flat at 0.4–0.5), and random shooting
  swings 0.3–0.9 (had been 0.1–0.9); slide 34's PETS curve runs behind the legend rather than stopping.
- **Printed oddities (checked on renders):** slide 32 prints "Kurt Konoglie" (kept as printed, not
  corrected); slide 14 prints literal asterisks around "especially"; slide 34's legends print "Nagabandi
  et. al", and a text box covers the lower part of both its charts and its slide number; slides 28 and 30
  print some symbols non-bold. Finn calls the random-shooting baseline yellow (≈1:10:35) where slide 34 draws it
  orange; the wiki says so.
- **Transcript edit:** Sonnet, given the parent's glossary (REINFORCE, PPO, importance sampling, SAC, DQN,
  RL, model-based, dexterous, Veo 2, learned simulator, z t, p of s t plus 1 given s t, backprop, r of s, a,
  stochasticity and Gaussian, cross-entropy method, overparameterized, parallelized, open- and closed-loop,
  warm-start, h for "eight" at 1:04:23, PDDM, MBPO, PETS, soft actor-critic, Baoding balls). 95/95 markers;
  0 lost, gained 1 ×2 (glossary restorations); 0 of 95 paragraphs outside the band (overall 0.886). 20
  student questions and 3 student answers marked. 6 [Ed:] notes in 6 paragraphs: "ps1 given st" (19:24),
  "next action" for the next state (20:10), "RSI" (27:13), "110 steps" as captioned (59:01), "eight" for h
  (1:04:23), "soft actor" (1:09:47). Parent fixes: the 1:04:23 note moved to the second "h" (the agent had
  put it on the first, which was spoken as h); a student-question mark added before the finite-differences
  restatement (38:50); a vague deferral removed from the header. Every restored proper noun is printed in
  the deck except "Gates building", a spoken aside.
- **Unclear in the recording, said so on the wiki:** the replanning horizon "something like 110 steps"
  (≈59:01); which step the sign correction at ≈32:40 refers to.
- **Images:** rendered 17 (4, 6, 7, 8, 12, 13, 17, 18, 19, 20, 21, 22, 27, 33, 34, 35, 36), 1.9MB: every
  figure-bearing slide except slide 5 (an agenda whose video frame shows slide 36's scene), the build steps
  14, 15 and 26, and slide 32's paper title block, transcribed in full. Not rendered also: 10, 16, 23, 25,
  28, 29, 30 (text, equation and algorithm slides) and 1, 2, 3, 9, 11, 24, 31, 37, 38 (title, divider,
  agenda, reminders).
- **Embedding:** the script placed 17 under their slide-file headings; the lecture page by hand (embed dry
  run: nothing missing, once a `slide-22.png` link was corrected to the file's `.jpg`); topic pages by hand
  with "Lecture 11, slide N" captions.
- **SEE_ALSO:** Princeton ROB 345 added for trajectory optimization and receding horizon control (MPC) with
  known dynamics; it does not learn models.
