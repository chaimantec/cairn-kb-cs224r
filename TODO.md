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

- [ ] 06 Q-Learning — video -7kv6jf0isQ — `06_cs224r_qlearning_2025.pdf`
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

## Run 6 — lecture 6 (Q-Learning)

Run 6, 2026-10-28. Scope: lecture 6 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 6
- [x] raw/transcripts/original/06-q-learning.md — verbatim captions (80 paragraphs, ~9,900 words, ~61 min)
- [x] Slide number map: 29 pages, 1:1. The map script reports 17 pages unnumbered; raw spans show 12 of them print the number at bottom centre (0.49–0.50, 0.96). Pages 3, 15, 20, 26, 27 print none (labels inferred)
- [x] raw/slides/06-q-learning.md — deck transcribed (Sonnet), `--verify` clean (29 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 7, 15, 24 agree, 26 cell by cell (49 percentages, table) but its human-level line corrected; 19 and 20 small re-reads; 25 checked for the printed exponent
- [x] raw/transcripts/06-q-learning.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 80/80 markers identical, 5 numbers lost and 0 gained, all decided restorations ("2 pi", "2 s", "F1 3D" ×2 digits, "U5 prime"), 2 of 80 paragraphs outside 0.72–1.10 (18:48 at 1.21 and 20:19 at 1.12, both only inserted [Ed:] notes); 24 student questions and 4 answers marked; 53 paragraphs carry [Ed:] notes
- [x] wiki/06-q-learning.md — 12 images placed by hand at the argument (embed dry run: nothing missing); check_quotes 31 quotations, 0 unmatched (3 fixed)
- [x] Topic pages: q-learning, bellman-equations (new); policies, offline-and-online, policy-evaluation, replay-buffers, importance-sampling, ppo, actor-critic, value-functions, rl-algorithm-families (its stale "Lectures 1–4" coverage line fixed), mdp-and-notation, 05-off-policy-actor-critic and 01-class-intro (forward links) updated
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/06-q-learning/ — 12 slides rendered (1.3MB)
- [x] Embedded in the slide file (12, script), the lecture page (12, by hand) and topic pages by hand with "Lecture 6, slide N" captions (q-learning 15, 18, 20; bellman-equations 12); 12 hand-written captions
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (and Layer 2's stale "Lecture 1, 2 or 3 decks" now "Lecture 1–6")

### Publish
- [x] kb.json — coverage 6 of 19, topic pages 17, images 90, figure audit 6 of 6, caveats (Lecture 6's coverage, board work and printed exponent; its [Ed:] density)
- [x] BUILD_LOG.md — lecture 6's record (the Findings below move with this section); AGENTS.md gains one general line on reading raw spans when the map reports pages unnumbered
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 2435 formulas, 0 errors, 0 warnings (slide file: 4 GitHub-raw formulas and 5 unspaced `$$` blocks fixed first); check_quotes 315 quotations, 0 unmatched; check_links 732 links, 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 6 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** the recording does not name the speaker, but slide 27 prints "Chelsea's advice" and
  the speaker introduces it as "my advice" (≈59:41), so the wiki writes "Finn" as for Lectures 1–5.
- **Licence:** none stated on the site or in the deck. Slides 8, 15, 17 and 19–23 print "Slide
  adapted from Sergey Levine". Slide 18 cites Mnih et al., Playing Atari with Deep Reinforcement
  Learning (2013) for DQN. Slide 26 credits "Mnih et al. 2015" (Atari chart) and "Kalashnikov et al.
  2018" (grasping photo and table, which cites "Levine et al. [27]"). Slides 19 and 20's charts are
  pasted from sources the slides do not name.
- **Deck:** 29 pages, printed number = page, 1:1. The map script reported 17 pages unnumbered; the
  raw spans show 12 of them print the number at bottom centre (0.49–0.50, 0.96), and pages 5, 8,
  12–14, 17, 19, 21–25 print it bottom right. Pages 3, 15, 20, 26 and 27 print none (labels
  inferred). `--verify` passes. Slides 3–5 revise Lecture 5's recap slides.
- **Recording coverage:** slides 3–29, 0:05 to 1:01:12, ending on slide 29's "Done with online RL
  methods!". Slide 2's reminders and slide 29's next-week topics are not discussed. Board work not
  visible: the algorithm with its targets (≈18:00–19:34). Slide 7's thought exercise is answered
  while pointing at the picture (≈7:08–12:32); the wiki reconstructs which rows are meant and says so.
- **Figure audit:** Opus parent, six 1400px renders (7, 15, 19, 20, 24, 26) against the Sonnet
  transcription, plus slide 25 for its exponent and a 500 dpi crop of slide 26's lower bars
  (Venture 32% confirmed). 7, 15, 24 and 25 agree; slide 26's 49 percentages and table agree cell by
  cell. Corrections: slide 26's human-level line had been placed at 100%, but it sits between
  H.E.R.O. (76%) and Asterix (69%), so 29 games are above it, several printed below 100%; slide 19's
  Breakout and Seaquest Q curves at epoch 25 (about 2 and 5, not 1.5 and 4); slide 20's Space
  Invaders DQN estimate at 100 million (about 8), its two true-value lines (coincide at about 4.9),
  and Zaxxon's Double DQN estimate leaving zero at about 75 million (not 50); slide 26's photograph
  shows arms at stations 2–4, not "four robot arms".
- **Printed slips (checked on the page):** slides 24–25 print the n-step discount as
  $\gamma^{t-t'}$, negative for $t' \gt t$, where Lecture 5's slides 12–13 print $\gamma^{t'-t}$;
  noted in the slide file and on the wiki and Q-learning pages, and the lecture reads the target
  aloud without discounts. Slides 12–13 print "Why does make sense?" (text layer). Slide 7 writes the
  estimate as $\hat{Q}^\pi$ in the text and $Q_\phi$ in the $\arg\max$. Recorded from the agent's
  list without a page check, so not claimed as slips: slide 14's $Q_\phi$ without a hat, the struck
  step 4 on slides 9–11 without $\phi$, slide 15's tuple order.
- **Transcript edit:** Sonnet, given the parent's glossary (Q-function/Q-value/Q-learning garbles,
  DQN, Bellman, Boltzmann, epsilon-greedy, n-step, phi prime, Seaquest, no-op, grasp, Ed). 80/80
  markers; 5 numbers lost, all restorations ("2 pi" and "2 s" to Q, the digits of "F1 3D" to
  epsilon-greedy, "U5 prime" to phi prime); 0 gained (the agent wrote its [Ed:] notes without
  digits). Ratios 2 of 80 outside the band, both only inserted [Ed:] notes (18:48 at 1.21, 20:19 at
  1.12). 24 student questions, 4 answers. The captions are much noisier than Lectures 1–5's: 53 of
  80 paragraphs carry [Ed:] notes, mostly garbled board dictation and mis-heard notation. Kept as
  captioned with notes: "08" and "7" at 11:00, "three" networks at 48:49, "is very impactful" at
  51:08, "a lot more biased" at 55:47. Parent fixes: three wrong slide numbers in the agent's
  restoration table (from the parent's own glossary: Boltzmann is slide 14, not 15).
- **Unclear in the recording, said so on the wiki:** the discounted values at 11:00, the search
  word for discrete spaces (≈15:40), the board algorithm's target (≈18:00–18:48), the rising-loss
  passage (≈44:57), "is/isn't very impactful" (≈51:08), two words of the double-Q answer (≈52:42),
  and "more biased" against the slide's "less biased" (≈55:47).
- **Images:** rendered 12 (3, 4, 5, 7, 8, 12, 15, 18, 19, 20, 24, 26), 1.3MB. Not rendered: 9–11,
  13, 14, 17, 21–23, 25 (equation slides with transcribed strikethroughs, highlights, loop arrows
  and annotations); 1, 2, 6, 16, 27, 28, 29 (title, reminders, plans, advice table, next time).
  Algorithm slides 12, 15 and 18 were rendered for their structure (colour-matched equations, the
  replay-buffer diagram, nested loops), following Lecture 5's rendering of its annotated algorithm
  slides.
- **Embedding:** lecture page by hand, all 12 at the argument (dry run: nothing missing); the script
  placed the 12 under their slide-file headings. Topic pages by hand with "Lecture 6, slide N"
  captions: q-learning 15, 18, 20; bellman-equations 12.
- **Math:** the offline check_math passed the slide file while the GitHub API check found four
  formulas GitHub would show as raw text (inline cases with `&`, a `)$)`) and five `$$` blocks
  without blank lines around them; all fixed before commit.

