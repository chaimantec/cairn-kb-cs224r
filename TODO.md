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

## Run 3 — lecture 3 (Policy Gradients)

Run 3, 2026-10-28. Scope: lecture 3 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 3
- [x] raw/transcripts/original/03-policy-gradients.md — verbatim captions (81 paragraphs, ~10,400 words, 62 min)
- [x] Slide number map: 28 pages, 1:1. The map script reports pages 3 and 23 as "1" and "0" (equation subscripts in the corner region); the raw spans show both print their own number. `--verify` will false-fail; check headings 1..28 in Python instead.
- [x] raw/slides/03-policy-gradients.md — deck transcribed (Sonnet), headings 1..28 checked (`--verify` false-fails on the map script's pages 3 and 23)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 6, 11, 12, 16, 23, 24 all agree; printed slips on 10, 11, 16, 17, 25 checked against the page; no corrections
- [x] raw/transcripts/03-policy-gradients.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 81/81 markers identical, 0 numbers lost (1 gained, inside an [Ed:] note), 0 of 81 paragraphs outside 0.72–1.10; three parent fixes (a word turned digit at ≈49:21, an unneeded [Ed:] at ≈33:00, an answer marker moved before "So the first answer was" at ≈37:44)
- [x] wiki/03-policy-gradients.md
- [x] Topic pages: policy-gradients, importance-sampling (new); offline-and-online (on-policy and off-policy section), rl-objective, rl-algorithm-families, imitation-learning, expressive-policy-distributions, policies, mdp-and-notation (updated)
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/03-policy-gradients/ — 11 slides rendered (1.1MB)
- [x] Embedded in the slide file (11), the lecture page (11, placed by hand at the argument; the script's dry run confirmed nothing missing), policy-gradients (8, script plus hand captions) and imitation-learning (slide 11, by hand, since that page cites Lecture 2 slides by the same numbers); 11 hand-written captions
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lessons; LICENSE.md attribution

### Publish
- [x] kb.json — coverage 3 of 19, images 42, figure audit 3 of 3, caveats
- [x] BUILD_LOG.md — lecture 3's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 942 formulas, 0 errors (9 formulas opening after a quotation mark fixed first); check_quotes 205 quotations, 0 unmatched (3 fixed); check_links 0 broken
- [x] Commit and push (chaimantec/cairn-kb-cs224r); kbUrl already set from run 1, so no PATCH
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 3 in the backlog; TODO.md keeps only open work

### Findings

- **Licence:** none stated on the site or in the deck. Slides 5 and 27 print "Source: Unitree";
  slides 7–10 print "Slide adapted from Sergey Levine".
- **Deck:** 28 pages, printed number = page, 1:1. `slide_number_map.py` reads pages 3 and 23 as
  "1" and "0": equation subscripts sit in its corner region. The raw spans show both pages print
  their own number (page 23's at an offset position), so `--verify` false-fails; headings were
  checked 1..28 in Python instead.
- **Recording coverage:** slides 3–28, 0:05–1:02:26. Slide 2's reminders are not discussed; the
  KL constraint on slide 25 is shown but skipped ("we don't have time to cover KL constraints
  today", ≈1:01:40).
- **Figure audit:** Opus parent, six 1400px renders (6, 11, 12, 16, 23, 24) against the Sonnet
  transcription, plus renders of 10, 17 and 25 to check the agent's listed slips. All agreed on
  marks, arrows, colour coding and equations; no corrections.
- **Printed slips:** slide 11 prints the BC gradient's inner lower limit as "t−1"; slide 16's
  baseline $b = \frac{1}{N} \sum_{i=1}^{N} r(\tau)$ has no index $i$ in the summand; slide 18's
  surrogate and gradient put the reward sum $\sum_{t'=t}$ outside the sum over $t$, which Finn
  acknowledges aloud (≈53:11: "this second sum should be inside the first sum"), and slides 20,
  22 and 23 print their first equation the same way (slide 13 nests it correctly, ≈33:48).
  Not slips, though the transcriber flagged them: slide 23's state-action ratio
  $\pi_{\theta'}(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$ is deliberate (≈1:00:54); slide 17's
  $-b$ inside the reward bracket is the trajectory form; slides 10, 12, 14, 17 drop the
  $\frac{1}{N}$ in step 2; slides 7–10 print a plain $E$.
- **Spoken slip:** ≈18:10 "run gradient descent" for the update that adds the gradient; noted on
  the wiki page.
- **Transcript edit:** Sonnet, given the parent's glossary of 26 decided restorations (REINFORCE,
  importance sampling, KL constraints confirmed on the slides). 25 student questions and 6
  student answers marked. Three [Ed:] notes: two flagged restorations (≈43:53 "v to the" → "B d
  tau", ≈58:34 "key of data" → "p theta") and one unclear passage (≈59:22 "this is not poorly
  defined"). Parent fixes: "1, two, and three" back to "one, two, and three" at ≈49:21 (the
  captions had the word); an [Ed: unclear] on "a 100" at ≈33:00 removed (spoken notation for the
  action at step 100); the first answer marker after the ≈37:44 think-pair-share moved before
  "So the first answer was".
- **Images:** rendered 11 (5, 6, 7, 10, 11, 12, 14, 17, 19, 24, 26), 1.1MB. Not rendered: 27 (a
  repeat of 5); 13, 15, 16 (equation slides whose only picture is a thumbnail of slide 12's
  humanoid); 8, 9, 21, 22 (the raster test flags them, but the rasters are pasted equation images
  that the slide file reproduces); 1–4, 18, 20, 23, 25, 28 (title, reminders, text and vector
  equations).
- **Embedding:** the embed script anchors on "slide N", so on a topic page citing two lectures'
  slides it would place a Lecture 3 image beside a Lecture 2 citation of the same number.
  imitation-learning's slide 11 image was placed by hand; lesson added to AGENTS.md. On
  policy-gradients, a sentence citing "slides 12, 14 and 17" would have stacked three images after
  one line; reworded so each lands with its own example.
- **Math on GitHub:** 9 formulas opened right after a quotation mark (`"$\tau^5$: …"`); reworded.
