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

## Run 12 — lecture 12 (Multi-Task RL)

Run 12, 2026-10-30. Scope: lecture 12 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 12
- [x] raw/transcripts/original/12-multi-task-rl.md — verbatim captions (92 paragraphs, ~11,700 words, ~70 min, ending at about 1:10:21 partway through slide 26)
- [x] Slide number map: 29 pages, printed 1–29, 1:1, every page numbered (bottom centre; bottom right on 12–19, 21, 23). The map script reported 6 missing numbers and 7 ambiguous pages because it missed the bottom-right numbers and took a stray "1" in page 14's content for its number; the raw spans settle it. `--verify` not usable; heading sequence checked 1..29 in Python
- [x] raw/slides/12-multi-task-rl.md — deck transcribed (Sonnet), headings 1..29; recording-coverage paragraph, video notes (21, 22), slide 24's spoken typo correction, slide 18's T-versus-n note and audit corrections added by the parent; check_math 121 formulas, 0 errors
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 6, 14, 19, 20, 24, 27, with crops of slide 27's chart and slide 18's equations. 6 and 20 agree; 27's series, colours and legend agree but its epoch-50 values were read about 10 epochs early (fixed); 19's sphere had been split into a video side and a language side (removed); 24's silhouette interpretation removed; 14's mannequin arms corrected
- [x] raw/transcripts/12-multi-task-rl.md — copy-edit (Sonnet) with the parent's glossary; 92/92 markers identical; 0 lost, gained 1 ×4 (s t plus 1, glossary restorations), 7 and 13 (slide numbers inside [Ed:] notes); 1 of 92 paragraphs outside 0.72–1.10 (7:47 at 1.12, its [Ed:] note; about 0.9 without it); overall 0.874; 19 student questions and 16 student answers marked; 6 [Ed:] notes. Parent fixes: a false start left at 25:25, a student-question mark before "Euclidean L2" at 43:08, a header phrase. No restored proper noun outside the deck
- [x] wiki/12-multi-task-rl.md — all 9 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 62 quotations, 88 citations, 0 unmatched; check_math 105 formulas, 0 errors
- [x] Topic pages: multi-task-rl and hindsight-relabeling (new); model-based-rl (data generation, when to use a model and other kinds of models now as Lecture 12 teaches them, with Lecture 12's slide 6 in place of Lecture 11's slide 27), offline-and-online (MBPO's offline and online versions; relabeled data is off-policy), replay-buffers (per-task and relabeled buffers), imitation-learning (multi-task imitation), mdp-and-notation (notation for multi-task RL; slide 7's letters), policies (task-conditioned policies), rl-algorithm-families (coverage now Lectures 1–12), reward-learning (rewards for synthetic and relabeled data), instruction-finetuning (SFT as multi-task imitation), importance-sampling (Finn's remark on relabeled data), course-logistics (Homework 4's HER), planning-with-models updated; 01-class-intro and 11-model-based-rl forward links (11's "Lecture 12's deck opens with its own version" pointers now say Lecture 12 teaches it). check_quotes 270 quotations, 501 citations, 0 unmatched; check_math wiki 2550 formulas, 0 errors; check_links 1551 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–12 and 199 images, Lecture 11's and 12's recording coverage, lecture entry, two new topic entries and twelve updated)

### Images
- [x] raw/images/12-multi-task-rl/ — 9 slides rendered (1.0MB) by page index through the script's render(), since its map lookup is wrong for this deck; embedded in the slide file (9, script), the lecture page (9, by hand, hand-written captions) and topic pages by hand with "Lecture 12, slide N" captions (model-based-rl 6; multi-task-rl 14, 19, 20; hindsight-relabeling 24, 27)
- [x] INDEX.md images paragraph; AGENTS.md table row and one general line (the map script can take a content digit for a page number, and the renderer inherits the map); LICENSE.md attribution (Layer 2 now "Lecture 1–12 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 12 of 19, topic pages 31, images 199, figure audit 12 of 12, caveats (Lecture 12's coverage, board work, slide 24's typo, slide 18's T; Lecture 11's caveat now says Lecture 12 teaches its deferred slides)
- [x] BUILD_LOG.md — lecture 12's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: no change; no sibling KB covers multi-task or goal-conditioned RL
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 199 and 9 of 29 checked; distribution 9 on disk, 9 in wiki); its GitHub math check: 4565 formulas, 0 errors; check_links 1551 links, 0 broken
- [ ] Commit and push (chaimantec/cairn-kb-cs224r); kbUrl already set from run 1, so no PATCH
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 12 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Chelsea Finn (the title slide names only "CS 224R"). Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©,
  Creative, rights). Third-party material: character-animation filmstrips, a garment simulation and robot-arm
  photographs (14, no source printed), a CNN drawing and an Earth picture (17), the BC-Z figure (19, Jang et
  al., CoRL 2021), the OpenVLA figure (20, Kim et al., CoRL 2024), π0.5 video frames (21–22, Pi Team et al.,
  arXiv 2025), hockey clip-art (24), the Hindsight Experience Replay paper's Figure 2 and chart (27,
  Andrychowicz et al., NeurIPS '17).
- **Deck:** 29 pages, printed 1–29, 1:1. The title slide's black title is "Multi-Task and Goal-Conditioned
  Reinforcement Learning", with "Model-Based RL and" inserted above it in pink. Slides 5–9 are a revised
  version of Lecture 11's 26–30 (slide 7 renames the policy $\pi_\theta$, the model $p_\psi$ and adds a critic
  $Q_\phi$, and updates on $D_ {env} \cup D_ {model}$).
- **Recording coverage:** about 70 minutes, slides 3–25 and the start of 26 (≈1:09:21), ending at about
  1:10:21. Not discussed: slide 2's reminders, the rest of slide 26, 27–29. Board work: the MBPO algorithm,
  with its generated data called D gen (≈7:47–9:20; "Whoops, I misspelled real"), the combined MDP's reward
  and dynamics (≈38:31–40:48), a one-hot vector (≈43:55), the multi-task imitation objective
  (≈44:40–46:11), the relabeling algorithm (≈1:00:54–1:02:26). At ≈1:00:08 Finn says "There's a typo there":
  slide 24 relabels the pass "with task 2 ID" where passing is task 1. At ≈1:08:35–1:09:21 Finn reconsiders a
  remark on importance weights for relabeled data; the wiki reports both versions as said.
- **Figure audit:** Opus parent, six 1400px renders (6, 14, 19, 20, 24, 27), with crops of 27's chart and
  18's equations. Corrections: 27's values near epoch 50 (pushing's Section 4.5 curve about 55% at 50, 78%
  at 60; blue about 15% at 50; pick-and-place about 95% by 60, not 30–50; the red dip at 140, not 150;
  sliding's red about 45% at 100); 19's embedding sphere (both encoder arrows end at the blue dots; no
  video/language sides); 24's large silhouette interpretation removed; 14's mannequin has both arms at its
  sides.
- **Printed oddities:** slide 14's title "A Few Examples Task Distributions"; slide 15's "How can do we
  identify"; slide 18's task sum runs to T (Finn says n); slides 5, 7, 19, 26 use non-bold or unbarred
  symbols; slide 25's "evaluatable" and an unexplained asterisk on "off-policy algorithm*"; slide 29 repeats
  slide 2's reminders.
- **Transcript edit:** Sonnet, given the parent's glossary (model-based, learned model, multi-task,
  relabeling, goal-conditioned, DAgger, trajectories at 5:28, shaped rewards, denser and sparse rewards,
  D n and D gen, pi theta, on-/off-policy, recommender, MDP, Markov, s bar, p i, r i, z i, D i at 43:55,
  Euclidean and mean squared, imitation at 46:11, heuristic, replay buffer at 1:00:08, s t / a t notation,
  p of s 1, dexterous, high-dimensional, high-level, captioning at 57:02). 6 [Ed:] notes: "directories"
  (5:28), "dn" (7:47), "r i of a" (38:31), "DI" (43:55), "and pen" (51:37), a garbled student answer
  (1:07:50).
- **Unclear in the recording, said so on the wiki:** the last words of the OpenVLA sentence (≈51:37); whether
  "DI" at ≈43:55 was $\mathbf{z}_ i$; the board's name for the real data ("dn").
- **Images:** rendered 9 (6, 14, 17, 19, 20, 21, 22, 24, 27), 1.0MB: every figure-bearing slide except slide
  5, a build step whose content slide 6 shows. Not rendered also: 7, 9, 13, 15, 16, 18, 23, 25, 26
  (equation, algorithm and labelled-formula slides, transcribed) and 1, 2, 3, 4, 8, 10, 11, 12, 28, 29
  (title, reminders, agenda, text).
- **Rendering:** `extract_slide_images.py --pages` resolves through `slide_number_map`, which mislabels this
  deck (page 14 as slide 1, pages 15–19 as 2–6), so it would have rendered wrong pages or refused. The
  parent called its `render()` by page index; a first attempt passed 0-based pages to the 1-based
  function and rendered every slide one page early; reading the function caught it, and all nine were re-rendered before any
  image was looked at.

