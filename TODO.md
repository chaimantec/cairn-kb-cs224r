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

- [ ] 13 Meta RL — video wSiyEpvoGkA — `13_cs224r_metarl_2025.pdf`
- [ ] 14 Exploration — video 4tlSKdi8teU — `14_cs224r_exploration_2025.pdf`
- [ ] 15 Hierarchical RL and IL — video iKWYLSVAtfM — `15_cs224r_hierarchy_2025.pdf`
- [ ] 16 RL for Robots — video rbaWQQLrzl0 — `16_cs224r_autonomy_2025.pdf`
- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 13 — lecture 13 (Meta RL)

Run 13, 2026-10-31. Scope: lecture 13 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 13
- [x] raw/transcripts/original/13-meta-rl.md — verbatim captions (90 paragraphs, ~12,000 words, ~69 min)
- [x] Slide number map: 39 pages, 1:1. The map script found no number (it scans the corners); the raw spans show bottom-centre numbers on 15 pages, each equal to its page index, and none on the other 24, which sit between consecutive numbered pages with no gap
- [x] raw/slides/13-meta-rl.md — deck transcribed (Sonnet), headings 1..39 (`--verify` OK on the 1..N fallback, which is right for this deck); check_math 172 formulas, 0 errors; recording coverage, slide 17's new-task note, slide 20's video note and audit corrections added by the parent
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 12, 19, 21, 22, 26, 35, with crops of 21's table and the charts of 22 and 26. 12, 21 (every cell), 22 and 35 agree; 26's series, colours and values agree but its dashed lines had been called the other methods' final performance (the slide does not say; removed); 19's attention network has seven rows of nodes (corrected). Slide 20's images found to be frames of two videos (≈33:08–34:40), which the transcription had not said. Text layer confirms slide 22's printed "ICML 2019", slide 9's "using on", plain n on slide 13
- [x] raw/transcripts/13-meta-rl.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 90/90 markers identical; 0 lost, gained 1 ×2 and 2 (s 1, s 2 from the caption S1, S2) and 19, 26, 8 (slide numbers in [Ed:] notes); 0 of 90 paragraphs outside 0.72–1.10, overall 0.914; 15 student-question and 9 student-answer markers; 4 [Ed:] notes. Restored names all in the deck except GPT-3, Aviral, deep set (spelled so in the captions) and positional encoding
- [x] wiki/13-meta-rl.md — lecture page; all 19 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 80 quotations, 121 citations, 0 unmatched (after converting the slide file's [colour] tags to prose); check_math 98 formulas, 0 errors
- [x] Topic pages: meta-rl (new, with slides 11, 17, 25, 33 placed by hand as "Lecture 13, slide N"); multi-task-rl (meta-RL against the aggregated MDP), policies (policies that learn in context), mdp-and-notation (the meta-RL POMDP; notation for meta-RL), llm-reasoning (test-time compute as meta-RL), replay-buffers (trajectory buffers for meta-RL), offline-and-online (on- and off-policy meta-training), rl-algorithm-families (coverage now Lectures 1–13), course-logistics (Lecture 13's reminders, evening office hours, part of HW4), 12-multi-task-rl, 01-class-intro and 10-rl-for-reasoning forward links. check_quotes 309 quotations, 621 citations, 0 unmatched; check_math wiki 2698 formulas, 0 errors; check_links 1664 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–13 and 218 images, Lecture 13's recording coverage, lecture entry, one new topic entry and eight updated)

### Images
- [x] raw/images/13-meta-rl/ — 19 slides rendered (2.3MB) by `extract_slide_images.py` (its 1..N fallback map is right for this deck); embedded in the slide file (19, script), the lecture page (19, by hand, hand-written captions) and meta-rl (11, 17, 25, 33, by hand)
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (a video's frame may carry no player chrome; colours in prose, never inline tags); LICENSE.md attribution (Layer 2 now "Lecture 1–13 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 13 of 19, topic pages 32, images 218, figure audit 13 of 13, decks transcribed 13, caveats (Lecture 13's coverage, board work, video frames, slide 22's printed venue, [Ed:] notes)
- [x] BUILD_LOG.md — lecture 13's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: no change; no sibling KB covers meta-RL
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 218 and 19 of 39 checked; distribution 19 on disk, 19 in wiki); its GitHub math check: 4891 formulas, 0 errors; slide_number_map --verify OK after the image insertions
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 13 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Chelsea Finn (the title slide names only "CS 224R"). Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, Creative,
  rights). Third-party material: an espresso photograph (6), Braque and Cezanne paintings (8–9), maze diagrams
  "adapted from Duan et al. '17" (10–11), a legged robot, a manipulation-task grid and clip-art (12), the RL²,
  SNAIL and PEARL architecture figures (19), SNAIL's maze video frames (20, Mishra et al., ICLR 2018), DeepSeek
  R1 excerpts and the MRT chart (22, Qu et al., venue printed as ICML 2019), robot and walker pictures (23),
  PEARL's charts and stills (26, Rakelly et al., ICML 2019), hallway, robot and document pictures (30), kitchen
  photographs (32), emoji (33), PEARL's trajectory panels (35), a trajectory panel beside MAME and MetaCURE
  citations (36).
- **Deck:** 39 pages, printed 1–39, 1:1. The map script found no number (it scans the corners); the raw spans
  show bottom-centre numbers on 15 pages (1, 2, 5, 8, 13–18, 20–22, 26, 27, 38, 39), each equal to its page
  index, and none on the other 24, which sit between consecutive numbered pages; the renders show none either.
  `--verify` passes on the 1..N fallback, which is right here, and the renderer's map is right too.
- **Recording coverage:** about 69 minutes, slides 3–37, ending at about 1:09:10 on slide 37's summary. Not
  discussed: slide 2's reminders, 38–39. Board work: the black-box algorithm (≈24:34–26:55), its meta-test
  (≈30:03–30:49), a discount factor (≈36:10). Slide 20's two maze videos play at ≈33:08–34:40 and are replayed
  ("watching that again. Whoops").
- **Figure audit:** Opus parent, six 1400px renders (12, 19, 21, 22, 26, 35), with crops of 21's table and the
  charts of 22 and 26. Agree: 12, 21 (every cell), 22 (series, colours, arrows, values), 35, and 26's series,
  colours and values. Corrections: 26's dashed lines had been called "the final performance of the other
  methods", which the slide does not say (removed, and each line's colour and level given); 19's attention
  network has seven node rows (red, two orange, green, two orange, green from the bottom), not the eight
  given. The transcriber reported no video frames; the transcript shows slide 20's images are video frames
  (added). The text layer confirms 22's "ICML 2019", 9's "using on", plain n on 13.
- **Printed oddities:** slide 9's "using on a few examples"; slide 22's venue "ICML 2019" for the MRT paper;
  slides 17–18 print $a$, $s$ non-bold; slide 29 says "w.r.t. Reward" where 31 and 34 say "w.r.t. Task
  Reward"; slide 26's first and third panels have clipped tick labels; the deck writes the training set as
  $\mathscr{D}_ {\text{tr}}$, $D_{\text{train}}$ and $\mathcal{D}_ {\text{train}}$ on different slides, and the
  episode count as N where Finn says n.
- **Slide file:** the transcriber wrote colours as inline tags ([red]…[/red]); seven wiki quotations of those
  phrases failed check_quotes until the tags were rewritten as prose (lesson in AGENTS.md).
- **Transcript edit:** Sonnet, given the parent's glossary (multi-task, meta-RL, meta-learning at 6:59, 7:45,
  9:16, meta-training, zero-shot transfer, test data points at 8:31, T i and T j, n, positional encoding, s 1,
  s 2, z i and z, parallelize, DeepSeek R1 and R1 at 39:58, test-time compute and training, goal-conditioned
  and k-shot at 43:03, off-/on-policy, end-to-end, hyperparameters, training process at 50:46, RL squared kept,
  decouple at 1:03:59). 4 [Ed:] notes: "test video points" (8:31), a word after PPO (49:13), RL² (50:46),
  "get couple of" (1:03:59).
- **Unclear in the recording, said so on the wiki:** the word after "PPO" at ≈49:13; "decouple" at ≈1:03:59;
  the exact form of the board's objective (reconstructed from the spoken description and labelled so).
- **Images:** rendered 19 (3, 6, 8, 9, 11, 12, 16, 17, 19, 20, 22, 23, 25, 26, 30, 32, 33, 35, 36), 2.3MB.
  Not rendered: build steps 10 (its mazes are on 11) and 24 (slide 16's chain with a question, answered by 25);
  21's table, transcribed cell by cell; 14 (input-output diagram, transcribed) and 27, 31, 34 (repeats of 16's
  or 17's chain beside text); 4, 7, 13, 18, 29, 37 (equation and text); 1, 2, 5, 15, 28, 38, 39 (title,
  reminders, agenda, divider).

