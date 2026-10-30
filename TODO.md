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

- [ ] 14 Exploration — video 4tlSKdi8teU — `14_cs224r_exploration_2025.pdf`
- [ ] 15 Hierarchical RL and IL — video iKWYLSVAtfM — `15_cs224r_hierarchy_2025.pdf`
- [ ] 16 RL for Robots — video rbaWQQLrzl0 — `16_cs224r_autonomy_2025.pdf`
- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 14 — lecture 14 (Exploration)

Run 14, 2026-10-31. Scope: lecture 14 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 14
- [x] raw/transcripts/original/14-exploration.md — verbatim captions (94 paragraphs, 11,743 words, ~73 min)
- [x] Slide number map: 34 pages, 1:1. The map script found no number (it scans the corners); the raw spans show bottom-centre numbers on 6 pages (2, 3, 14, 29, 33, 34), each equal to its page index, and none on the other 28, which sit between consecutive numbered pages with no gap
- [x] raw/slides/14-exploration.md — deck transcribed (Sonnet), headings 1..34 (`--verify` OK on the 1..N fallback, right for this deck); check_math 94 formulas, 0 errors after the parent fixed underscore spacing, RL² after quotes and slide 23's quoted formula; recording coverage, slide 10's drawn regret curves, slide 12's drug game, video notes on 27, 30, 31 and audit corrections added by the parent
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 9, 17, 20, 24, 26, 32. 9, 17, 20, 24 (series, colours, legend, values) and 32 (every cell of both tables) agree; 26's series, colours and reference lines agree, but DREAM had an early dip to −0.2 the chart does not show (removed) and IMPORT's and RL²'s opening dips were missing (added); 17, 19, 22, 23 print script 𝒟 (text layer U+1D49F), written as calligraphic (corrected). Possible-video notes on 4 and 25 removed (the transcript presents neither as video); 27, 30, 31 confirmed as video frames from the transcript. Text layer confirms the printed slips on 3, 8, 19, 21, 23, 26, 34
- [x] raw/transcripts/14-exploration.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 94/94 markers identical; numbers identical (0 gained, 0 lost); 0 of 94 paragraphs outside 0.72–1.10, overall 0.905; 15 student-question, 2 caught-question and 3 student-answer markers, one drug-game marker; 4 [Ed:] notes. Restored names all in the deck except Pong (spoken only). Parent fixed the coverage paragraph (opening overview follows slide 3; ends ≈1:12:40) and recorded the agent's "his" → "this" estimate (28:48)
- [x] wiki/14-exploration.md — lecture page; all 18 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 118 quotations, 135 citations, 0 unmatched; check_math 128 formulas, 0 errors
- [x] Topic pages: exploration (new, with Lecture 14's slides 10, 11, 20 placed by hand as "Lecture 14, slide N"); meta-rl (intro no longer says Lecture 14 is missing; new DREAM section with slide 26 placed by hand), multi-task-rl (DREAM's execution policy as multi-task RL), q-learning (the bandit game on epsilon-greedy), mdp-and-notation (notation for exploration), rl-algorithm-families (coverage now Lectures 1–14), course-logistics (Lecture 14's reminders, DREAM in HW4), 13-meta-rl (forward links), 06-q-learning, imitation-learning and llm-reasoning (where-this-appears entries). check_quotes over all wiki 1112 quotations, 2274 citations, 0 unmatched (after restoring the transcript's "multitask" to multi-task, as Lecture 13's did); check_math wiki 2911 formulas, 0 errors; check_links 1791 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–14 and 236 images, Lecture 14's recording coverage, lecture entry, one new topic entry and six updated)

### Images
- [x] raw/images/14-exploration/ — 18 slides rendered (1.9MB) by `extract_slide_images.py` (its 1..N fallback map is right for this deck); embedded in the slide file (18, script), the lecture page (18, by hand, hand-written captions), exploration (10, 11, 20) and meta-rl (26), by hand
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (standing spellings in every glossary; a transcriber's possible-video flag checked against the transcript, and a live demo described from the transcript); LICENSE.md attribution (Layer 2 now "Lecture 1–14 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 14 of 19, topic pages 33, images 236, figure audit 14 of 14, decks transcribed 14, caveats (Lecture 14's coverage, the game, board work, N(0, 1) correction, video frames, [Ed:] notes)
- [x] BUILD_LOG.md — lecture 14's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: no change; CS221's KB has epsilon-greedy, which Lecture 6 already covers, and no sibling covers bandits or meta-exploration
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims slides 3–32, 18 of Lecture 14 and 18 of 34 checked; distribution 18 on disk, 18 in wiki); its GitHub math check found two slide-file formulas ending in ")" before ")" (fixed); slide_number_map --verify OK after the image insertions
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 14 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Chelsea Finn (the title slide names only "CS224R"). Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, creative,
  rights). Slides 4–10 print "Slide adapted from Sergey Levine"; slide 8 credits David Silver's lecture notes.
  Third-party material: Breakout and Montezuma's Revenge frames (4), a playing-card photograph (6), kitchen
  photographs and an LLM reasoning screenshot (15), emoji (16, 20, 22), a trajectory panel (17), DREAM's chart,
  environment, learning curves and video frames (24–27, Liu et al., ICML 2021), Bounce and Breakout frames (30),
  frames from Liu et al., NeurIPS 2022 (31), the grading interface and tables of Liu et al., SIGCSE 2024 (32).
- **Deck:** 34 pages, printed 1–34, 1:1. The map script found no number (it scans the corners); the raw spans
  show bottom-centre numbers on 6 pages (2, 3, 14, 29, 33, 34), each equal to its page index, and none on the
  other 28, which sit between consecutive numbered pages. `--verify` passes on the 1..N fallback, which is
  right here, and the renderer's map is right too. The training set is script 𝒟 (U+1D49F) in the text layer.
- **Recording coverage:** about 73 minutes (last caption ends ≈1:12:39), slides 3–32. Not discussed: slide 2's
  reminders, 33–34. Slide 12 is a live web game (≈24:53–31:54), described in the slide file and wiki from the
  transcript; Finn says the website will be put in the slides, and the PDF does not print it. Drawn where the
  recording does not show: regret curves on slide 10 (≈12:32–18:46, "The blue markers never work that well"),
  the noise and bottleneck construction (≈46:34–49:41), four example tasks (≈51:13), the execution objective
  (≈52:46–55:52), the exploration reward (≈1:00:32–1:03:38). Videos on 27 (≈1:07:29–1:08:16), 30 (≈1:10:35)
  and 31 (≈1:11:20).
- **Lecturer's correction:** slides 21–23 print the KL penalty's unit Gaussian as $\mathcal{N}(0, 1)$; Finn says
  "this should be an I" since z is a vector (≈55:52). Noted under slides 21–23; the wiki writes
  $\mathcal{N}(0, I)$ and says the slides print otherwise.
- **Figure audit:** Opus parent, six 1400px renders (9, 17, 20, 24, 26, 32). Agree: 9, 17, 20, 24 (series,
  colours, legend, approximate values), 32 (every cell of both tables). Corrections: 26's DREAM curve had been
  given an early dip to −0.2 that the chart does not show (removed) and IMPORT's (−0.28) and RL²'s (−0.15)
  opening dips were missing (added); 17, 19, 22, 23's script 𝒟 had been written as calligraphic. The
  transcriber's possible-video flags on 4 and 25 were removed (the transcript presents neither as video), and
  27, 30, 31 confirmed as video frames from the transcript.
- **Printed oddities (text layer confirms each):** "Meta-Exploration in for CS Education" (3, 14, 33); slide 8's
  URL ends "XX.pdf"; slide 19's "allows accurate task prediction from f" where the model is q; slide 21's
  "Add noise the representation." and "my discard good info"; slide 23's em dash for a minus; slide 26's
  "Rakelly, et. al."; slide 34's "extra long-horizon"; $\mathcal{N}(0, 1)$ on 21–23 (above).
- **Transcript edit:** Sonnet, given the parent's glossary (meta-learning, meta-RL, Montezuma's Revenge, Mao,
  weakly, multi-armed, MDPs at 8:40 and 24:07, sum of rewards and capital T at 11:01–11:47, a 1, exploiting at
  27:14, patience at 33:26, KL divergence, Gaussian, mu i, z i, z bar, one-hot, heuristic, Q-learning and
  A squared log A at 1:05:58, DREAM and its expansion at 1:04:26, interleave, CS106A, Bounce, Breakout).
  4 [Ed:] notes: "We have one patient's eye" (28:48), "the reward for s a t" (1:02:05), captions "meta or"
  (1:04:26), "recover ever" (1:04:26). The agent changed "his estimate" to "this estimate" (28:48), recorded
  in the restorations table; the parent restored "multitask" (41:09, 41:55) to multi-task. The agent kept some
  of Finn's stumbles in 31:07 to hold its ratio in the band.
- **Unclear in the recording, said so on the wiki:** the reward's arguments at ≈1:02:05; the spoken labels for
  the noisy and mean vectors at ≈53:32–54:19 (the wiki uses the slide's notation); the drawn regret curves
  (described from the spoken words).
- **Images:** rendered 18 (4, 6, 9, 10, 11, 15, 16, 17, 19, 20, 22, 24, 25, 26, 27, 30, 31, 32), 1.9MB. Not
  rendered: 5 (its only image is slide 4's frame); 7, 8, 13, 18, 21, 23, 28 (equation and text, transcribed
  exactly); 1, 2, 3, 12, 14, 29, 33, 34 (title, reminders, agenda, divider). The variational information
  bottleneck (≈45:46–52:46, about 7 minutes) has no figure-bearing slide; the wiki carries it in prose.
