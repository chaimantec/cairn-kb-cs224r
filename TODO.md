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

- [ ] 15 Hierarchical RL and IL — video iKWYLSVAtfM — `15_cs224r_hierarchy_2025.pdf`
- [ ] 16 RL for Robots — video rbaWQQLrzl0 — `16_cs224r_autonomy_2025.pdf`
- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 15 — lecture 15 (Hierarchical RL and IL)

Run 15, 2026-10-31. Scope: lecture 15 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 15
- [x] raw/transcripts/original/15-hierarchical-rl.md — verbatim captions (91 paragraphs, 12,318 words, ~70 min)
- [x] Slide number map: 46 pages, 1:1. The map script reports 13 pages unnumbered; the raw spans show 39–41 print their number at bottom right (missed by the script), and 5, 8, 10, 11, 31–35 and 43 print none, each between consecutive numbered pages with no gap
- [x] raw/slides/15-hierarchical-rl.md — deck transcribed (Sonnet), `--verify` OK (46 headings), check_math 94 formulas, 0 errors; recording coverage, board-work list, video notes (23, 31–33, 35, 36, 39, 43) and audit corrections added by the parent
- [x] Figure spot-check from 1400px renders (Opus parent; transcriber was Sonnet): 24, 27, 34, 39, 40, 41, 43, plus 44 and a high-zoom crop of 24's Helix diagram. 27 (every label) and 40 (every cell and bold) agree, and 34's right-hand chart. Corrected: 24 (π0.5's dashed subtask-to-command line; Helix frequencies in full), 34 (left chart's bar heights as fractions of an axis-less panel replaced by relative heights; IA/TP from the transcript), 41 (arrow direction), 43 (two stages of two tasks, not views). Video: 31–33, 35, 36 confirmed from the transcript, 39 added (≈57:00, 1:01:38), 23's speed tags, 43's black rectangle as an unexported video. Text layer confirms the slips on 9, 34, 39
- [x] raw/transcripts/15-hierarchical-rl.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 91/91 markers identical; 1 number lost (the stutter "1-hertz 1-hertz" at 49:20 → "1 hertz", deliberate), 0 gained; 0 of 91 paragraphs outside 0.72–1.10, overall 0.916; 14 student-question and 4 student-answer markers; no [Ed:] notes. Parent fixed one restoration timestamp (3:58) and a lower-case "in" at 1:02:23
- [x] wiki/15-hierarchical-rl.md — lecture page; all 20 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing, after fixing slide 13's extension); check_quotes 114 quotations, 75 citations, 0 unmatched; check_math 39 formulas, 0 errors
- [x] Topic pages: hierarchical-rl (new, with Lecture 15's slides 7 and 30 placed by hand as "Lecture 15, slide N"); policies (hierarchical policies), mdp-and-notation (notation for hierarchy), imitation-learning (hierarchical imitation learning), hindsight-relabeling (relabeling in hierarchical RL), multi-task-rl (slide 22's pointer now links Lecture 15), rl-algorithm-families (coverage now Lectures 1–15), course-logistics (Lecture 15's reminders), exploration and llm-reasoning (where-this-appears entries); forward links from 01-class-intro, 12-multi-task-rl and 14-exploration. check_quotes over all wiki 1250 quotations, 2383 citations, 0 unmatched; check_math wiki 2981 formulas, 0 errors; check_links 1914 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–15 and 256 images, Lecture 15's recording coverage, lecture entry, one new topic entry and eight updated)

### Images
- [x] raw/images/15-hierarchical-rl/ — 20 slides rendered (2.3MB) by `extract_slide_images.py` (map 1:1); embedded in the slide file (20, script) and the lecture page (20, by hand, hand-written captions)
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (PyMuPDF and `python3 -I`; charts with no axis values); LICENSE.md attribution (Layer 2 now "Lecture 1–15 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 15 of 19, topic pages 34, images 256, figure audit 15 of 15, decks transcribed 15, caveats (Lecture 15's coverage, board work, video frames, slide 34's axis-less chart; [Ed:] counts for Lectures 12–15, which the transcript caveat had stopped listing after Lecture 11)
- [x] BUILD_LOG.md — lecture 15's record (the Findings below move with this section); index rows for Lectures 1, 12, 14 and 15
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: no change; no sibling KB covers hierarchical RL or imitation
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims checked: slides 3–44 and 46, 20 of Lecture 15, 20 of 46; distribution 20 on disk, 20 in wiki); its GitHub math check passes (5377 formulas); slide_number_map --verify OK after the image insertions
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 15 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Chelsea Finn (the site lists no guest; the title slide names only "CS 224R"). Pronouns not stated;
  the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, creative,
  rights). Third-party material: focaccia and beach photographs (13), stills and architecture diagrams of π0.5,
  Gr00t N1, Helix and Gemini Robotics (23–24), Yell At Your Robot's data, workcell, diagram and video frames
  (27–33, 35), Hi Robot's and Yell At Your Robot's charts (34), a π0.5 video frame (36), SuSIE's frames (39), a
  figure from Chen, Belkhale, et al. 2025 (41), maze scenes from Nachum et al. NeurIPS '18 (43), and frames from
  "Diversity is all you need" (44).
- **Deck:** 46 pages, printed 1–46, 1:1. The map script reported 13 pages unnumbered; the raw spans show 39–41
  print their number at bottom right (missed by the script), and 5, 8, 10, 11, 31–35 and 43 print none, each
  between consecutive numbered pages. `--verify` passes and the renderer's map is right. Slide 4 lists the
  design choices in a different order from slides 12 and 45 and from the lecture.
- **Recording coverage:** about 69 minutes (last caption ends ≈1:09:28), slides 3–44, then slide 46's "next
  time" (≈1:09:15). Not discussed: slide 2's reminders, 45, 46's reminders. Drawn where the recording does not
  show: the hierarchical policy (≈3:58–6:16), the roll-out procedure before slide 8 (≈10:09–11:41), the pasta
  example's sets of states, on no slide (≈25:31–28:36, ≈30:57–37:04), a demonstration's variable-length
  segments (≈47:01), an image-subgoal policy (≈1:01:38). Videos on 31–33 (≈51:39–53:56), 35 (≈54:41), 36
  (≈55:27) and 39 (≈57:00–58:34; "you saw the noise kind of moving into the image", ≈1:01:38); 23's tagged
  stills; 43's black square is most likely an unexported video (≈1:07:44).
- **Lecturer's correction:** on the board, "I wrote this as low-level this should be high-level" (≈5:30); no
  printed slide is corrected aloud.
- **Figure audit:** Opus parent, 1400px renders of 24, 27, 34, 39, 40, 41, 43 and 44, and a high-zoom crop of
  24's Helix diagram. Agree: 27 (every label), 40 (every cell and bold), 34's right-hand chart (27, 51, 55, the
  "20% gap" bracket). Corrections: 24's π0.5 dashed line from subtask prediction to low-level command, and
  Helix's frequencies (robot images and state 7-9 Hz to System 2 and 20 Hz to System 1, latent vector 7-9 Hz,
  output 200 Hz, GPU 2 and GPU 1); 34's left-hand bars had been given as fractions of an axis-less panel
  (replaced by relative heights and the printed gaps; IA and TP from ≈53:56–54:41); 41's arrows run from the
  vertical line into the boxes; 43's panels are two stages of two tasks, not views; 27's first-row labels are
  printed in quotation marks; 45's outline is slide 12's order, not slide 4's. Video notes: 31–33, 35, 36
  confirmed and given their content from the transcript, 39 added, 23 qualified, 43 explained.
- **Printed oddities (text layer confirms each):** slide 9's "running polices at different frequencies";
  slide 34's "Lucy Shi, Hu, Zhao, ..." where 29–33 print "Shi, Hu, Zhao, ..."; slide 39's "Black\*, Nakamoto\*"
  (equal-contribution marks); slide 27's repeated "put bread on lettuce" (render); slide 27's "Train high level
  and low-level policy".
- **Transcript edit:** Sonnet, given the parent's glossary (multi-task, Q-function, goal-conditioned, DAgger,
  subgoal, subtask; "multi-task imitation and reinforcement learning" at 0:05, pi LL at 5:30, s t at 11:41, the
  low-level policy at 20:56, adapted at 36:17 and 37:04, post hoc at 47:47, 1 hertz at 49:20, o t plus k at
  1:01:38, train at 1:03:09). No [Ed:] notes; the captions are clean. The agent also made "end to end" →
  "end-to-end" at 28:36. Parent fixed one restoration timestamp and a lower-case "in" at 1:02:23.
- **Unclear in the recording, said so on the wiki:** the board drawings (described from the spoken words); the
  four systems' videos on slide 23 (only their tasks are named); which skill each frame of slide 44 shows.
- **Images:** rendered 20 (3, 7, 11, 13, 23, 24, 27, 28, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 43, 44), 2.3MB.
  Not rendered: 8, 10, 29 (build steps of 7, 11, 30); 14, 16, 17, 18, 20, 21 (slide 7's or 13's small diagram
  beside text, transcribed); 40 (table, transcribed cell by cell); 5, 6, 9 (text); 1, 2, 4, 12, 15, 19, 22, 25,
  26, 37, 42, 45, 46 (title, reminders, agenda, dividers). The pasta example (≈25:31–37:04, about 9 minutes
  with the supervision discussion) has no figure-bearing slide; the wiki carries it in prose.
