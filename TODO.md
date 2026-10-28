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

- [ ] 02 Imitation Learning — video WxRDyObrm_M — `02_cs224r_imitation_2025.pdf` (in progress: Run 2 below)
- [ ] 03 Policy Gradients — video KCAOXd4IO9o — `03_cs224r_policy_gradients_2025.pdf`
- [ ] 04 Actor-Critic Methods — video oejFZShW9hU — `04_cs224r_actor_critic_2025.pdf`
- [ ] 05 Off-Policy Actor Critic — video cRGKc-nAWho — `05_cs224r_offpolicy_actor_critic_2025.pdf`
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

## Run 2 — lecture 2 (Imitation Learning)

Run 2, 2026-10-28. Scope: lecture 2 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 2
- [x] raw/transcripts/original/02-imitation-learning.md — verbatim captions (87 paragraphs, 11,228 words, 67 min)
- [x] raw/slides/02-imitation-learning.md — deck transcribed (Sonnet), `--verify` clean (27 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 7, 8, 10, 12, 19, 21 all agree; three descriptive fixes
- [x] raw/transcripts/02-imitation-learning.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 87/87 markers identical, 0 numbers lost (7 gained, all inside [Ed:] notes), 0 of 87 paragraphs outside 0.72–1.10; 28 student turns marked; 7 [Ed:] notes (3 unclear)
- [x] wiki/02-imitation-learning.md
- [x] Topic pages: imitation-learning, expressive-policy-distributions, offline-and-online (new); policies, rl-algorithm-families, mdp-and-notation (updated)
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/02-imitation-learning/ — 15 slides rendered (1.8MB)
- [x] Embedded in the slide file (15), the lecture page (15) and two topic pages (imitation-learning 6, expressive-policy-distributions 4); 15 hand-written captions, reused on the topic pages
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution

### Publish
- [x] kb.json — coverage 2 of 19, images, caveats
- [x] BUILD_LOG.md — lecture 2's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 0 errors (10 slide-file formulas fixed first); check_quotes 0 unmatched (1 fixed); check_links 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 2 in the backlog; TODO.md keeps only open work

### Findings

- **Licence:** none stated on the site or in the deck. Slides 1 and 15 print "SpongeBob image
  rights & credits to Viacom International"; those two slides are not rendered.
- **Deck:** 27 pages, printed number = page, 1:1. Pages 10 and 25 print no number.
- **Recording coverage:** slides 3–21. The video opens on the recap (slide 3) and ends at ≈67:05
  in the questions after slide 21; slides 2 and 22–27 are not discussed.
- **Figure audit:** Opus parent, six 1400px renders (7, 8, 10, 12, 19, 21) against the Sonnet
  transcription. All six agreed on series, values, arrows, labels and equations. Three
  descriptive fixes: slide 7's arrows (traffic faces the camera, so the blue arrow to the image's
  right is the driver's left), Mario's position on slide 8, slide 12's photo (a tabletop, not a
  floor).
- **Printed slips:** none found. Slide 27 repeats slide 2's reminders verbatim ("Homework 1 out
  today").
- **Transcript edit:** Sonnet, given the parent's glossary of 23 decided restorations (Waymo/Wayve,
  OpenVLA, covariate shift, DAgger and GMM confirmed on slides 10–20). 28 student turns marked,
  two of them answers. Seven [Ed:] notes: three unclear passages (≈57:06 "this pen", ≈1:00:09
  "PBS", ≈1:04:45 "played with something"), one probable restoration (≈54:46 "co-training"), and
  three spoken slips kept as said (≈35:31 "10 to the fifth" for 100 bins in 10 dimensions;
  ≈37:48 "minimizing the log probability"; ≈1:01:42 "this name has a terrible algorithm").
- **Images:** rendered 15 (3, 5, 7–10, 12–14, 18–21, 23, 24), 1.8MB. Not rendered: slides 1 and 15,
  whose only images are SpongeBob stills credited to Viacom; text, agenda and equation-only slides
  2, 4, 6, 11, 16, 17, 22, 25–27. The raster test flagged 14 pages (1, 3, 5, 7–10, 12–15, 20, 23,
  24); slides 18, 19 and 21 are vector diagrams it cannot see, rendered on the slide file's
  evidence.
- **Math on GitHub:** the slide file had 10 formulas broken on github.com: `\{ \}` (5), `^*` (7
  stars), one heading formula ending `)$)` and one formula opening after a quote mark. Fixed with
  `\lbrace`/`\rbrace`, `^{\ast}`, and rewording. The wiki pages passed first time.
- **Embedding:** a coverage note citing "slides 3–21" would have pulled slide 3's image into the
  page intro (a range 12 or more wide cites its first slide), and sentences ending in `:` before a
  list would have split them; both reworded before embedding. Lesson added to AGENTS.md.

