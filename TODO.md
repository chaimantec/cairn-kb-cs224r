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

- [ ] 16 RL for Robots — video rbaWQQLrzl0 — `16_cs224r_autonomy_2025.pdf`
- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 16 — lecture 16 (RL for Robots: Autonomous Learning)

Run 16, 2026-10-31. Scope: lecture 16 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 16
- [x] raw/transcripts/original/16-autonomous-rl.md — verbatim captions (85 paragraphs, ~11,400 words)
- [x] Slide number map: 46 pages, 1:1. The text layer prints a number only on 28, 30 and 32 (bottom right, each equal to its page index); the other 43 pages print none, and with no gap anywhere the labels are the page index
- [x] raw/slides/16-autonomous-rl.md — deck transcribed (Sonnet), `--verify` OK (46 headings), check_math 113 formulas, 0 errors; recording coverage, board-work list, video notes (8, 26, 30–33, 37–39, 42–44) and audit corrections added by the parent
- [x] Figure spot-check from 1400px renders (Opus parent; transcriber was Sonnet): 13, 16, 17, 24, 25, 28, 30. 13 (equations, arrows), 16 (every series and value) and 28 (every arrow) agree. Corrected: 24 (early values of VaPRL, MEDAL, FBRL, the oracle; SAC's last), 25 (MEDAL at 8.0e5, the unlabelled green series, SAC), 17 (a partly covered green curve, not a blob), 30 (no finger count). Video: every flagged slide confirmed from the transcript; 37–38 are the first frame of 39's video
- [x] raw/transcripts/16-autonomous-rl.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 85/85 markers identical; 0 numbers lost or gained; 0 of 85 paragraphs outside 0.72–1.10, overall 0.885; 13 student-question and 3 student-answer markers; no [Ed:] notes. Every context-dependent restoration (H, MEDAL, Yevgen, z i, pi f, r f, r b, sum) sits only at its glossary timestamps; the agent added one more H (25:33, as slides 10 and 19). Parent restored a dropped "and" at 32:29 ("This, and forward-backward RL, addresses"), "a more high-level space" at 1:01:13, two sentence starts (22:26, 41:42), one capital (37:06) and the single-life row's timestamps
- [x] wiki/16-autonomous-rl.md — lecture page; all 25 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 134 quotations, 140 citations, 0 unmatched; check_math 53 formulas, 0 errors
- [x] Topic pages: autonomous-rl (new, with Lecture 16's slides 17, 19 and 40 placed by hand as "Lecture 16, slide N"); new sections in rl-objective (evaluating an agent that never resets), reward-learning (rewards for learning without resets), replay-buffers (without resets), multi-task-rl (practising tasks without resets), hierarchical-rl (a task proposer is not a high-level policy), imitation-learning (demonstrations for learning without resets) and mdp-and-notation (notation for autonomous RL); where-this-appears entries in those and in exploration, course-logistics (Lecture 16's reminders) and rl-algorithm-families (coverage now Lectures 1–16); forward links from 01-class-intro, 14-exploration and 15-hierarchical-rl. check_quotes over all wiki 1407 quotations, 2574 citations, 0 unmatched; check_math wiki 3092 formulas, 0 errors; check_links 2044 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–16 and 281 images, Lecture 16's recording coverage, lecture entry, one new topic entry and nine updated, course-logistics entry)

### Images
- [x] raw/images/16-autonomous-rl/ — 25 slides rendered (2.6MB) by `extract_slide_images.py` (map 1:1); embedded in the slide file (25, script) and the lecture page (25, by hand, hand-written captions)
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (Layer 2 now "Lecture 1–16 decks"; the deck's text layer has no licence notice). No new general lesson for AGENTS.md

### Publish
- [x] kb.json — coverage 16 of 19, topic pages 35, images 281, figure audit 16 of 16, decks transcribed 16, caveats (Lecture 16's coverage, board work, video frames, slide 13's t = 0; no [Ed:] notes)
- [x] BUILD_LOG.md — lecture 16's record (the Findings below move with this section); index rows for Lectures 1, 14, 15 and 16
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: no change; no sibling KB covers reset-free or single-life RL
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims checked: 281 images, slides 3–45, 25 of 46; distribution 25 on disk, 25 in wiki); its GitHub math check caught one inline formula split across a line in autonomous-rl (fixed); slide_number_map --verify OK after the image insertions
- [ ] Commit and push (kbUrl already set from run 1, so no PATCH)
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 16 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Chelsea Finn (the site lists no guest; the title slide names no lecturer); most slides are credited
  "Slide adapted from Archit Sharma". Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, creative,
  rights). Third-party material: a Gym code screenshot (6), illustrations (7), video frames from Chebotar et al.,
  Yahya et al. and Sharma et al. (8), the Fish render and chart (16), EARL's renders and charts (24–25), real-robot
  time-lapse frames (26), pictograms (28), figures and frames from Gupta et al. 2021 (30), Smith et al. 2021 and
  Han, Levine, Abbeel (31), SOAR (32), a DYNA-1 video frame credited to an x.com post (33), a door-key video frame
  (37–39), figures and frames from "You Only Live Once" (40–43), and RMA and Chen et al. ICRA 2025 frames (44).
- **Deck:** 46 pages, 1:1. The text layer prints a number only on 28, 30 and 32 (bottom right, equal to the page
  index); the other 43 pages print none, and with no gap the labels are the page index. `--verify` passes.
- **Recording coverage:** about 66 minutes (last caption ends ≈1:05:41), slides 3–45. Not discussed: slide 2's
  reminders and slide 46's reminders and "next time". Written where the recording does not show: an agent's
  reset-free experience as a red trajectory (≈9:16–11:35), the continuing-evaluation formula (≈13:53–14:40), the
  drift and collapse sketch (≈19:19–21:38), the forward-backward algorithm (≈23:14–25:33), a step "negative one"
  (≈30:12), the task-cycle algorithm (≈44:02–45:35). Videos on 8 (≈6:58–7:44), 26 (≈39:23–40:55), 30
  (≈45:35–47:09), 31 (≈47:09–47:55), 33 (≈52:39), 39 (≈54:13–54:59; 37–38 its first frame), 42 (≈56:35–57:20),
  43 (≈1:00:25) and 44 (≈1:00:25–1:02:00); slide 32's half-empty frame is most likely the first frame of a video or
  animation (≈48:42–51:06).
- **Lecturer's correction:** none of a printed slide. Writing the deployed objective, Finn adds "This should be,
  yeah, an expectation" (≈13:06). Slide 13's continuing-evaluation sum starts at t = 0 where Finn says t = 1, and
  it prints plain s, a where the deployed objective prints bold.
- **Figure audit:** Opus parent, 1400px renders of 13, 16, 17, 24, 25, 28 and 30. Agree: 13 (both equations and
  the three arrows), 16 (every series against its label and value), 28 (every arrow of the task graph).
  Corrections: 24's early values (VaPRL about 0.92 at 1.0e6, not 0.85; MEDAL about 0.35 at 1.0e6 and 0.92 at
  1.5e6; FBRL 0.5 and 0.72; the oracle about 0.7 at 1.0e6) and SAC's last value (0.35); 25's MEDAL at 8.0e5
  (about 0.75, not 0.4), the unlabelled green series (about 0.4 at 8.0e5, not 0.1) and SAC; 17's partly covered
  green line is a curve; 30's hand without a finger count. Video notes added from the transcript for every
  flagged slide.
- **Printed oddities:** slide 25's green series has no label (VaPRL's colour on slide 24); slide 13's mixed bold
  and plain notation; slide 38 is identical to slide 37 in the export.
- **Transcript edit:** Sonnet, given the parent's glossary (standing spellings; sim-to-real; Yevgen at 6:58 from
  Lecture 1's slide 30; Archit at 7:44; H at 3:54, 4:41, 7:44, 8:30, 9:16, 45:35; "sum" at 13:53; SAC; rho star;
  pi f, r f, r b; MEDAL at 33:15, 37:06, 40:08; EARL; z i; seven-stage; latter; adapting; bake in). No [Ed:]
  notes. The agent added H at 25:33. Parent restored a dropped "and" at 32:29 ("This, and forward-backward RL,
  addresses", where slide 23 credits MEDAL), "a more high-level space" at 1:01:13 (the agent had "higher level"),
  two sentence starts and one capital. Proper nouns: Archit, MEDAL, EARL, SAC, SOAR appear in the deck; Yevgen
  in Lecture 1's deck; Ali is kept as captioned (no printed counterpart).
- **Unclear in the recording, said so on the wiki:** the drawings (described from the spoken words); what
  slide 32's figure showed beyond its first frame.
- **Images:** rendered 25 (5, 6, 7, 8, 11, 12, 13, 16, 17, 19, 22, 23, 24, 25, 26, 28, 30, 31, 32, 33, 39, 40,
  42, 43, 44), 2.6MB. Not rendered: 10, 18, 29, 37, 38, 41 (build steps or repeats of rendered slides); 14, 20,
  21, 27, 35, 36 (text); 1, 2, 3, 4, 9, 15, 34, 45 (title, reminders, plan, summary); 46 (portrait of the next
  guest lecturer). Every topic the lecture dwells on has a figure-bearing slide except the safety question
  (≈1:02:00–1:04:20), which the wiki carries in prose.
