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

## Run 5 — lecture 5 (Off-Policy Actor Critic)

Run 5, 2026-10-28. Scope: lecture 5 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 5
- [x] raw/transcripts/original/05-off-policy-actor-critic.md — verbatim captions (90 paragraphs, ~11,900 words, 69.4 min)
- [x] Slide number map: 30 pages, 1:1. Page 4 prints no number (label inferred). The map script reports pages 9, 15–16 and 19–28 as unnumbered too, but their raw spans show the number in the bottom-right corner (0.91–0.98, 0.94–0.96)
- [x] raw/slides/05-off-policy-actor-critic.md — deck transcribed (Sonnet), `--verify` clean (30 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 16, 18, 25 agree (25 cell by cell); 12, 13, 23 corrected (small); 20 checked for its subtitle slip
- [x] raw/transcripts/05-off-policy-actor-critic.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 90/90 markers identical, 7 numbers lost and 7 gained, all decided restorations (0.2, 0.8 ×2, 1,000 ×2, a stutter, "Q4" → "Q for"), 0 of 90 paragraphs outside 0.72–1.10; 25 student questions and 2 answers marked; 9 [Ed:] notes
- [x] wiki/05-off-policy-actor-critic.md — 15 images placed by hand at the argument; check_quotes 46 quotations, 0 unmatched
- [x] Topic pages: ppo, replay-buffers (new); actor-critic (off-policy section rewritten from Lecture 5), importance-sampling, offline-and-online, value-functions, policy-evaluation, rl-algorithm-families, mdp-and-notation, policy-gradients, 04-actor-critic (updated)
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/05-off-policy-actor-critic/ — 15 slides rendered (1.8MB)
- [x] Embedded in the slide file (15, script), the lecture page (15, by hand at the argument; the script's dry run found nothing missing) and topic pages by hand with "Lecture 5, slide N" captions (ppo 13; replay-buffers 16, 18, 23); 15 hand-written captions
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution

### Publish
- [x] kb.json — coverage 5 of 19, images 78, figure audit 5 of 5, caveats (Lecture 4's slides-only caveat now points to Lecture 5; Lecture 5's coverage and board work added)
- [ ] BUILD_LOG.md — lecture 5's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 2006 formulas, 0 errors (slide file: one formula opening after a quotation mark fixed first); check_quotes 266 quotations, 0 unmatched (4 fixed on the lecture page); check_links 593 links, 0 broken
- [ ] Commit and push; check_math --github-published
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 5 in the backlog; TODO.md keeps only open work

### Findings

- **Licence:** none stated on the site or in the deck. Slides 15, 16 and 19–21 print "Slide adapted
  from Sergey Levine". Slides 24–27 credit Haarnoja et al. (Soft Actor-Critic Algorithms and
  Applications, 2018), Luo, Xu, Wu, Levine (HIL-SERL, 2024), Tan et al. (Sim-to-Real, 2018) and
  OpenAI (Solving Rubik's Cube with a Robot Hand, 2019). Slides 13 and 23's charts are pasted from
  sources the slides do not name. Slide 12's meme is credited only "from one of the GAE authors".
  Slide 22 cites Haarnoja, Zhou, Abbeel, Levine, Soft Actor-Critic (2018).
- **Deck:** 30 pages, printed number = page, 1:1. Page 4 prints no number (label inferred). The map
  script also reports pages 9, 15–16 and 19–28 as unnumbered: they print the number in the
  bottom-right corner, outside the region it reads. `--verify` passes. The deck revises Lecture 4's
  slides 24–37: slide 7 = L4 25 and slide 22 = L4 34 verbatim; 14, 15 and 21 near-identical; the
  PPO slides (11–13), the Q-from-buffer slides (17–18) and the examples (24–28) are new.
- **Recording coverage:** slides 3–28, 0:05 to the end at about 1:09:20 on slide 28. Slide 2's
  reminders and slides 29–30 are not discussed. Board work not visible: learning-curve sketch
  (≈13:58), grid-world sketch (≈16:14), KL-penalized objective (≈21:39), clipping with max/min
  (≈29:24), recursive Q (≈53:18). The wiki reconstructs the KL objective from the spoken words and
  says so; slide 10 gives the constraint form.
- **Figure audit:** Opus parent, six 1400px renders (12, 13, 16, 18, 23, 25) against the Sonnet
  transcription, plus 20 for the agent's subtitle slip. 16, 18, 20 and 25 agree (25's table cell by
  cell, no total row). Corrections: slide 23's right chart, SAC about 700 (not 500) at 2 million and
  about 2900 (not 3400) at 4 million, PPO about 350 (not 500) at 2 million; slide 13's blue curve
  about 500 (not 400) at 0.5 million, and the thumbnail floor's colours dropped; slide 12's meme
  had been given a comic-book character name the slide does not print.
- **Printed slips (checked on the page):** slides 12–13's n-step advantage sums rewards to $t+n$
  before $\gamma^n \hat{V}(\mathbf{s}_ {t+n})$, where slide 5's n-step target stops at $t+n-1$
  (noted in the slide file and on the wiki); slide 20 repeats slide 19's subtitle "Fixing the value
  function" (Lecture 4's counterpart reads "Fixing the policy update") and still prints the
  V-based advantage in step 4; slide 18 writes $\pi_\theta(\cdot, \mathbf{s}'_ i)$ with a comma.
  Recorded from the agent's list without a page check, so not claimed as slips: the outline's
  "b." for the second item (slides 6, 29), "How to we" (17–18), the $(i,t)$ versus $(t,i)$ index order
  (as in Lecture 4), step 5's $\theta$ update under a $\theta'$ gradient (slide 8), slide 18's
  missing $r_i$ in step 1.
- **Transcript edit:** Sonnet, given the parent's glossary of decided restorations (PPO, SAC, KL,
  GAE, n-step, AlphaGo, Q-functions, seeded, the clipping numbers 0.2/0.8, negative and positive
  1,000, 1 over 9). 25 student questions and 2 answers marked. [Ed:] notes: five unclear passages
  (≈15:29 "the as anam", ≈17:00 "needs the thumb", ≈22:29 the KL objective's name "co", ≈54:49
  "after sat", ≈57:57 "fit QA") and four flagged restorations (≈18:33 "a network" → advantage,
  ≈36:20 "1,00" → 1,000, ≈37:06 "one over n" → nine, ≈37:51 "vi" → V pi). Number inventory: 7 lost
  and 7 gained, all restorations. The agent's additions not in the glossary (a question marker at
  ≈17:47, "different n's" at ≈40:56) were checked and kept.
- **Unclear in the recording, said so on the wiki:** a word in the grid-world answer (≈17:00), the
  name of the KL-penalized objective (≈22:29), and a few words of the Q derivation (≈54:49).
- **Images:** rendered 15 (3, 4, 5, 10, 13, 15, 16, 18, 19, 20, 23, 24, 25, 26, 27), 1.8MB. Not
  rendered: 12 (meme); 8 and 9 (build steps of 7 and 10); 7, 11, 17, 21, 22 (equation slides with
  transcribed highlights, arrows and boxes); 1, 2, 6, 14, 28, 29, 30 (title, reminders, plan, text).
  Selection followed Lecture 4's precedent for the shared slides.
- **Embedding:** lecture page by hand, all 15 at the argument (dry run: nothing missing); the
  script placed the 15 under their slide-file headings. Topic pages by hand with "Lecture 5, slide
  N" captions: ppo 13; replay-buffers 16, 18, 23.
