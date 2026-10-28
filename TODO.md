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

## Run 4 — lecture 4 (Actor-Critic Methods)

Run 4, 2026-10-28. Scope: lecture 4 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 4
- [x] raw/transcripts/original/04-actor-critic.md — verbatim captions (83 paragraphs, ~10,300 words, 63.5 min)
- [x] Slide number map: 37 pages, 1:1. Page 6 prints no number (label inferred). The map script reports pages 21 and 29–34 as unnumbered too, but their raw spans show the number in the bottom-right corner (0.91, 0.94), outside the region the script reads
- [x] raw/slides/04-actor-critic.md — deck transcribed (Sonnet), headings 1..37 checked in Python
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 6, 14, 21, 30 agree; 19 clarified, 35 corrected (one series' band and two values); 22, 26, 27 checked for the agent's doubts, 27 corrected (it had slide 26's loop arrow)
- [x] raw/transcripts/04-actor-critic.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 83/83 markers identical, number inventory identical, 0 of 83 paragraphs outside 0.72–1.10; 28 student questions and 6 answers marked; one parent fix (≈22:22 "learning to play" back to "learning to practice playing")
- [x] wiki/04-actor-critic.md
- [x] Topic pages: actor-critic, policy-evaluation (new); value-functions (rewritten around V, Q and A), policy-gradients, rl-algorithm-families, rl-objective, offline-and-online, importance-sampling, mdp-and-notation, 03-policy-gradients (updated)
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/04-actor-critic/ — 21 slides rendered (2.1MB)
- [x] Embedded in the slide file (21, script), the lecture page (21, by hand at the argument; the script's dry run found nothing missing) and topic pages by hand with "Lecture 4, slide N" captions (actor-critic 3, policy-evaluation 4, value-functions 2, rl-objective 1); 21 hand-written captions
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution

### Publish
- [x] kb.json — coverage 4 of 19, images 63, figure audit 4 of 4, caveats
- [x] BUILD_LOG.md — lecture 4's record (the Findings below move with this section); AGENTS.md: one general math lesson (no inline math straight after a quotation mark)
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 1524 formulas, 0 errors (slide file: `\,` `\;` `\{` escapes, 15 formulas opening after a quotation mark, a `$$` block inside a list item and one `…)$)` fixed first); check_quotes 132 quotations, 0 unmatched (1 fixed); check_links 463 links, 0 broken
- [x] Commit and push; check_math --github-published
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 4 in the backlog; TODO.md keeps only open work

### Findings

- **Licence:** none stated on the site or in the deck. Slides 21 and 29–33 print "Slide adapted
  from Sergey Levine" (31 bottom-right, the others bottom-left). Slide 7's drum-notation sheet is
  credited on the sheet to Theo Lawrence / TL Music Lessons. Slide 35's two learning-curve charts
  are pasted from a paper the slide does not name. Slide 34 cites Haarnoja, Zhou, Abbeel, Levine,
  Soft Actor-Critic (2018).
- **Deck:** 37 pages, printed number = page, 1:1. Page 6 prints no number (label inferred). The
  map script also reports pages 21 and 29–34 as unnumbered: they print the number in the
  bottom-right corner (0.91, 0.94), outside the region it reads. `--verify` still passes, since the
  mapping is 1:1.
- **Recording coverage:** slides 3–23, 0:05 to the end of the video at about 1:03:30, closing on
  the review slide (23) at ≈1:03:07. Slide 2's reminders and slides 24–37 (off-policy
  actor-critic, the SAC/PPO chart, the plan recap, next week) are not discussed. The wiki presents
  them as slides-only. The schedule's next lecture is titled Off-Policy Actor Critic; slide 37
  names Q-learning for next week.
- **Figure audit:** Opus parent, six 1400px renders (6, 14, 19, 21, 30, 35) against the Sonnet
  transcription, plus 22, 26 (scratch render), 27, 20 and 32 to check the agent's doubts and
  slips. 6, 14, 21, 22 and 30 agree (30's "not the right target value" arrow genuinely ends between
  step 3's oval and step 4's term; the [unclear] stands). Corrections: slide 19's trajectory layout
  rewritten (both start together; the grey one loops round the pink state and joins at the blue
  one); slide 27 had been described with slide 26's small step-5-to-step-4 arrow, which it lacks;
  slide 35's widest shaded band is PPO's, not SAC's, PPO is about 6000 (not 5000) at 6 million
  steps, and SQL about 5000 by 3 million; slide 20 gained the short magenta bracket it omitted.
  Added under slide 14: the drawn network has many output nodes, and Finn says it outputs "a single
  scalar value, not as pictured here" (≈34:00).
- **Printed slips (checked on the page):** slides 26–27 index the importance ratio $(i,t)$ and the
  other terms $(t,i)$; slides 22 and 25–27 drop the $\frac{1}{N}$ that slides 30–32 print;
  slide 32's step 4 prints $Q(\mathbf{s}_ i, \mathbf{a}_ i)$ with no hat. Not slips: slide 20's
  n-step target has no $\gamma$ because discounting arrives on slide 21 (slide 22 prints
  $\gamma^n$); slide 5's "more efficient" becomes "better" on slides 13, 24 and 36. The agent's
  slips on 18, 33 and 34 (subscript omissions) were not checked against the page and are not
  recorded as slips.
- **Transcript edit:** Sonnet, given the parent's glossary of 16 decided restorations (V pi / Q pi
  / A pi, spoken s t notation, "black trajectory", "n-step", "gamma to the n" confirmed on the
  slides). 28 student questions and 6 student answers marked. [Ed:] notes: two unclear passages
  (≈4:41 "used to write ingredients", ≈46:59 part of a question on backpropagation) and two flagged
  restorations (≈49:17 "a V hat of", caption "the hat of"; ≈54:41 "one and T", caption "one and n
  t"). One parent fix: ≈22:22 "learning to play the drums" back to the spoken "learning to
  practice playing the drums". The agent's "Actor-critic methods are learning" (≈1:02:21, caption
  "Actor-critic are learning") was kept as a grammatical repair.
- **Unclear in the recording, said so on the wiki:** a few words of the value-function definition
  (≈4:41), part of the backpropagation question (≈46:59), and the range for $n$ (≈54:41, where the
  slide's $n \gt 1$, $n \lt T$ is used). Finn's summary at ≈51:38 gives $-1$ for states after
  the blue state twice. It is not quoted; the page gives the per-trajectory Monte Carlo targets
  from ≈50:04–50:51 instead.
- **Images:** rendered 21 (3, 6, 7, 8, 9, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 27, 29, 30, 31,
  32, 35), 2.1MB. Not rendered: 11, 15 and 26 (build steps of 12, 16 and 27; 26's one extra arrow
  is described in the slide file); 4, 10, 33 and 34 (equation slides with transcribed highlights,
  strike-throughs and pointer arrows; 29–34's rasters are pasted equations, but 29–32 carry the
  replay-buffer diagram or hand-drawn ovals and so were rendered); 1, 2, 5, 13, 24, 25, 28, 36 and
  37 (title, reminders, plan and text). The raster test missed this deck's vector diagrams (slides
  3, 6, 11, 12 have no raster), so selection followed the slide file's prose.
- **Embedding:** lecture page by hand, all 21 at the argument, captions hand-written; the embed
  script then placed the 21 under their slide-file headings (alt text with $V^\pi$ rewritten as
  "V-pi"). Topic pages by hand with "Lecture 4, slide N" captions: actor-critic 12, 22, 35;
  policy-evaluation 17, 18, 19, 20; value-functions 6, 7; rl-objective 21.
- **Math on GitHub:** the slide file arrived with `\,`, `\;`, `\{`, `\|` escapes, 15 formulas
  opening right after a quotation mark, `$n > 1$` in inline math, a `$$` block inside a list item
  and one `…)$)`; all fixed. Opening after a quotation mark recurred from run 3, so it is now in
  AGENTS.md's math bullet.
