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

- [ ] 17 Advancing Robot Intelligence — video Hp1WBWghrak — `17_rl_for_robotics.pdf`
- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 17 — lecture 17 (Advancing Robot Intelligence, guest lecture)

Run 17, 2026-10-31. Scope: lecture 17 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 17
- [x] raw/transcripts/original/17-sim-to-real.md — verbatim captions (65 paragraphs, ~9,100 words; catalog length 2,989 s)
- [x] Deck confirmed by its title slide: "Advancing Robot Intelligence with Reinforcement Learning", Ashish Kumar, matching the video's first sentence (≈0:05)
- [x] Slide number map: 79 pages, 1:1. The text layer prints a number only on 62–67 (bottom right, x≈1841, each equal to its page index); the other 73 pages print none, and with no gap the labels are the page index
- [x] raw/slides/17-sim-to-real.md — deck transcribed (two Sonnet agents, pages 1–40 and 41–79, appended), `--verify` OK (79 headings), check_math 87 formulas, 0 errors (after quoted formulas, `\|` and two subscripts were fixed); recording coverage, section table, video notes (5–11, 22–27, 30, 31, 33–36, 45, 46, 55, 56, 58, 62, 67, 69, 70, 73, 74) and audit corrections added by the parent
- [x] Figure spot-check from 1400px renders (Opus parent; transcriber was Sonnet): 14, 16, 19, 34, 38, 51, 54, 65, 66. 14, 19, 51 (every box, label, arrow), 16 (all ten reward terms), 38 (every cell) and 54 agree. Corrected: 34 (the extrinsic curves cross three times; orange's second rise; knee torque's irregular middle third), 65 (the wide cycle), 66 (pink ovals on troughs 2–4). Video flags checked against the transcript: kept with notes on 5–11, 22–27, 30, 31, 33–36, 45, 62, 67, 69, 70, 73, 74, and 55–56 (motion described); 58 added (the limping demo); removed on 15, 49, 53, 57, 64
- [x] raw/transcripts/17-sim-to-real.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 65/65 markers identical; 0 numbers lost, 2 gained (the 1 of DeepSeek-R1 at 4:40, slide 3; phase 2 at 27:45, restored by the parent from the question restated at 24:44); 0 of 65 paragraphs outside 0.72–1.10, overall 0.969; 7 student-question markers. Agent's additions beyond the glossary are all standing spellings (AlphaGo at 4:40, Go, sim-to-real at 44:42, real-time, zero-shot, on-policy, general-purpose). Parent restored "Spot robot" at 35:32 (slide 54 prints Spot, the taller robot beside the A1) in place of the glossary's [Ed:] note, and added an [Ed:] note on "the relatives" at 25:29. [Ed:] notes remain at 4:40 (word), 10:00 (RAM, latter) and 25:29
- [x] wiki/17-sim-to-real.md — lecture page; all 39 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 135 quotations, 124 citations, 0 unmatched; check_math 42 formulas, 0 errors
- [x] Topic pages: sim-to-real (new); new sections in imitation-learning (distilling a privileged teacher), policies (adapting from a history of states and actions), reward-learning (hand-designed rewards for locomotion, and a general reward model), offline-and-online (imitation and RL that work) and mdp-and-notation (notation for sim-to-real); where-this-appears entries in those and in ppo, meta-rl, llm-reasoning, rlhf, autonomous-rl, course-logistics and rl-algorithm-families (coverage now Lectures 1–17); forward links from 01-class-intro, 05-off-policy-actor-critic and 16-autonomous-rl (twice). check_quotes over all wiki 1577 quotations, 2746 citations, 0 unmatched; check_math wiki 3166 formulas, 0 errors; check_links 2200 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–17 and 320 images, Lecture 17's recording coverage, lecture entry, one new topic entry and twelve updated, course-logistics entry)

### Images
- [x] raw/images/17-sim-to-real/ — 39 slides rendered (6.4MB) by `extract_slide_images.py` (map 1:1); embedded in the slide file (39, script) and the lecture page (39, by hand, hand-written captions); `--verify` OK after the insertions
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (Layer 2 now "Lecture 1–17 decks"; the deck's text layer has no licence notice). No new general lesson for AGENTS.md

### Publish
- [x] kb.json — coverage 17 of 19, topic pages 36, images 320, figure audit 17 of 17, decks transcribed 17, caveats (Lecture 17's coverage, questions, video frames, blank slide 46, undefined reward symbols; 4 [Ed:] notes)
- [x] BUILD_LOG.md — lecture 17's record (the Findings below move with this section); index rows for Lectures 1, 5, 16 and 17
- [x] sources.md "In this KB" column, and the deck match confirmed by the title slide
- [x] SEE_ALSO.md: no change; no sibling KB covers sim-to-real transfer (ROB 345's covered lectures are planning, control and estimation)
- [x] verify_kb.py clean (after staging the new files), review section read (no ranking claims; numeric claims checked: 320 images, 39 for Lecture 17, slides 2–78; distribution 39 on disk, 39 in wiki); its GitHub math check passed (5762 formulas)
- [x] Commit and push (chaimantec/cairn-kb-cs224r); kbUrl already set from run 1, so no PATCH
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 17 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Ashish Kumar, guest lecture (title slide prints the name; the course site lists it as "RL for
  Robots: Sim-to-Real Transfer", the catalog as "Advancing Robot Intelligence"). Pronouns not stated; the
  wiki writes "Kumar". Aviral Kumar (Lecture 10) is a different guest; the pages give the full name on first use.
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, creative,
  rights). Third-party material: RMA (RSS 2021) diagrams, renders, plots and video frames (12, 14–20, 23, 27,
  28, 30, 31, 34, 36), photographs (40), Miki et al. 2022 and Kim et al. 2020 terrain-map figures (42–43), the
  egocentric-vision paper's figures and frames (45, 47–58), Qi et al. CoRL 2022 frames and charts (62–66),
  Bristol demo photographs with a credit only in the text layer (67), drone frames (69–70), humanoid frames
  presented as Tesla's (73–74), Epic Kitchens frames (75).
- **Deck:** 79 pages, 1:1. The text layer prints a number only on 62–67 (bottom right, equal to the page
  index); the other 73 pages print none, and with no gap the labels are the page index. Transcribed by two
  Sonnet agents in parallel (1–40 into the slide file, 41–79 into a scratch file appended by the parent);
  `--verify` passes.
- **Recording coverage:** about 50 minutes (last caption ends ≈49:45), slides 2–78; slide 79's "Questions?"
  is not reached. Seven student questions (≈22:24–28:32) sit between slides 41 and 42 and are mostly about
  RMA (slides 14–20); the lecture page collects them in one section. Videos on 5–11 (≈6:57–8:29), 22–27
  (≈13:03–14:38), 30–31 (≈15:25–16:12), 33–36 (≈16:12–17:46), 45 (≈30:52–32:26), 58 (≈36:18), 62
  (≈38:36), 67 (≈40:07), 69–70 (≈40:53–41:40), 73 (≈42:26), 74 (≈43:11); 55–56 most likely (motion described
  at ≈35:32–36:18). Slide 46 is blank in the PDF.
- **Lecturer's correction:** none of a printed slide. Kumar's "four times heavier" and "three times" arm length
  (≈40:53) round slide 69's 792 g against 177 g and 16.6 cm against 5.8 cm; the wiki gives both.
- **Figure audit:** Opus parent, 1400px renders of 14, 16, 19, 34, 36, 38, 51, 54, 65 and 66. Agree: 14, 19,
  51 (every box, label and arrow), 16 (all ten reward terms; "yaw" is typewriter too), 38 (every cell), 54.
  Corrections: 34 (the extrinsic curves cross three times; orange's second rise to about 0.65; knee torque's
  irregular middle third), 36 (torque plateaus higher after the drop; blue's drop to about 0.4 a little before
  the middle, not early; one touch and one crossing), 65 (the wide cycle by position), 66 (pink ovals on
  troughs 2–4). Slide 53's 17cm (left) and 26cm (right) checked from the text layer's span positions.
- **Printed oddities:** slide 4 prints "What enabled these success ?" where slide 3 prints "successes"; slide
  38's row label "RMA w/o Adapt" against slide 30's "RMA w/o Adaptation"; slide 67's credit "Credit: Max Yang,
  University of Bristol" is in the text layer but not visible; slide 62's citation is partly covered by the
  middle panel; slide 16 defines none of its symbols, and its "Z Acceleration" term is printed with $v_z$.
- **Transcript edit:** Sonnet, given the parent's glossary (standing spellings; gait for "gate" throughout;
  proprioceptive; AlphaGo; DeepSeek-R1 at 4:40 from slide 3; PPO at 10:00; DAgger; z; Hz; torque profile; sys
  ID; RMA; testbed; CoRL; Isaac Gym; dexterous; quadruped; "agile, dexterous" at 43:56 and 48:35; seven
  student-question markers). The agent's additions were standing spellings only. Parent restored "Spot robot"
  at 35:32 (slide 54 prints Spot beside the small A1, where the glossary had an [Ed:] note) and "phase 2" at
  27:45 (the question restated at 24:44), and added an [Ed:] note on "the relatives" at 25:29. Proper nouns:
  AlphaGo, DeepSeek, RMA, CoRL, Isaac Gym, Spot, Epic Kitchens and Rich Sutton appear in the deck; Berkeley,
  CMU and Tesla are spoken only, with Berkeley and CMU as affiliations on slides 13 and 41.
- **Unclear in the recording, said so on the wiki:** a word at 4:40 ("word": world or board), a phrase after
  "about a billion samples" (≈10:00, "This is RAM."), the names of the first reward terms (≈10:00, "the
  forward and the latter"), the start of the answer on the extrinsics' size (≈25:29); which panel of slide 73
  is the dance; what slide 46 held.
- **Images:** rendered 39 (12, 14, 15, 17, 18, 19, 20, 23, 27, 28, 30, 31, 34, 36, 40, 42, 43, 45, 47–54, 56,
  57, 58, 62, 63, 65, 66, 67, 69, 70, 73, 74, 75), 6.4MB. Not rendered: 5–11 (frames slide 12's grid
  repeats), 22, 24, 25, 26, 33, 35, 55, 64 (repeats or frames with nothing a rendered slide lacks); 16, 38, 60
  (equation list and tables transcribed exactly); 46 (blank); 18 text slides; 1, 13, 41 (title pages). Every
  topic the lecture dwells on has a figure-bearing slide except the student questions (≈22:24–28:32) and the
  closing speculation on simulation and reward models (≈45:29–47:47), which the wiki carries in prose.

