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

## Run 8 — lecture 8 (Reward Learning)

Run 8, 2026-10-29. Scope: lecture 8 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 8
- [x] raw/transcripts/original/08-reward-learning.md — verbatim captions (86 paragraphs, ~10,700 words, ~66 min)
- [x] Slide number map: 34 pages, 1:1. The map script reports pages 2, 4, 8, 19, 27, 28, 29, 30, 34 unnumbered; raw spans show pages 2 and 34 print their number bottom right (0.98, 0.95). Pages 4, 8, 19, 27–30 print none (labels inferred; page 4's only bottom-corner digit is a "2" at y 0.84, not a page number)
- [x] raw/slides/08-reward-learning.md — deck transcribed (Sonnet), `--verify` clean (34 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 9 (table cell by cell, architecture arrows) and 32 agree; 30 (dark-grey series' last three points re-read), 27 (green curve's peak inside the plot; lane count not legible), 15 (second classifier photo: pencil case to the right and further back, not in front), 23 (green arrow ends level with the star) small fixes. Slide 29's repeated sub-line confirmed on the render
- [x] raw/transcripts/08-reward-learning.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 86/86 markers identical; 0 numbers lost; 20 gained, all glossary restorations (0.5, K choose 2, phi's [Ed:] note), L2's digit, or spoken number words written as digits (eight, five, zero, one); 0 of 86 paragraphs outside 0.72–1.10; 17 student questions and 4 answers marked (parent moved one question marker at 15:23 before its restatement); 4 [Ed:] notes
- [x] wiki/08-reward-learning.md — all 18 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 0 unmatched
- [x] Topic pages: conservative-q-learning, reward-learning, preference-learning (new); offline-rl (CQL section now points to Lecture 8 and its own page), 07-offline-rl (forward links), implicit-policy-constraints (why only V gets the asymmetric loss), imitation-learning, q-learning, ppo, rl-objective (where the reward comes from), rl-algorithm-families (coverage line now Lectures 1–8), mdp-and-notation (notation for reward learning) and 01-class-intro (forward link) updated
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/08-reward-learning/ — 18 slides rendered (1.9MB); embedded in the slide file (18, script), the lecture page (18, by hand, hand-written captions) and topic pages by hand with "Lecture 8, slide N" captions (conservative-q-learning 5, 6, 8, 9; reward-learning 15, 17, 19, 32; preference-learning 23, 25, 26, 27, 29)
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (and Layer 2's "Lecture 1–7 decks" now "Lecture 1–8"; the deck's text layer has no licence notice, only "Slide adapted from Sergey Levine" on 5–8)

### Publish
- [x] kb.json — coverage 8 of 19, topic pages 22, images 122, figure audit 8 of 8, caveats (Lecture 7's CQL now taught in Lecture 8; Lecture 8's coverage, board work, unspoken Q-value, slide 29's copy slip; its [Ed:] count)
- [x] BUILD_LOG.md — lecture 8's record (the Findings below move with this section); AGENTS.md gains two general lines: an image anchored on a sentence that runs into a displayed equation lands inside it, colon or not; describe a photographed person by the printed name, not apparent gender
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 122 and 18 of 34 checked; distribution 18 on disk, 18 in wiki); check_math --github: 3290 formulas, 0 errors, 0 warnings (three fixed first: two `)$)` endings in the slide file, one tau followed by a plural s in the wiki); check_quotes 716 quotations, 1159 citations, 0 unmatched; check_links 1026 links, 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 8 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** the recording does not name the speaker, and the deck prints no name. The site's
  schedule lists no guest for this lecture (sources.md), so the wiki writes "Finn" as for Lectures 1–7.
- **Licence:** none stated on the site or in the deck (text layer searched). Slides 5–8 print "Slide
  adapted from Sergey Levine"; slide 13's video frame prints "© Warneken & Tomasello" in the image.
  Citations: Prabhakar et al. ('22) on 9; Mnih et al. '15 on 12; Sharma, Ahmed, Ahmad, Finn ('23) on 19;
  Yu et al. ViT-VQGAN (ICLR '22) and Villegas et al. Phenaki (ICLR '23) on 20; Christiano et al.
  (NeurIPS 2017) and Sadigh et al. (RSS 2017) on 27; Bai et al. Constitutional AI (2022) on 30;
  Sukhbaatar et al. (ICLR 2018) on 32.
- **Deck:** 34 pages, printed number = page, 1:1. The map script reported pages 2, 4, 8, 19, 27–30 and 34
  unnumbered; the raw spans show 2 and 34 print their number bottom right (0.98, 0.95); 4, 8, 19 and
  27–30 print none (labels inferred). `--verify` passes. The title page reads "Reward Learning" with
  "Conservative Offline RL and" inserted above it; slides 5–8 and 9–10 match Lecture 7's 25–28, 31 and 30.
- **Recording coverage:** slides 3–32, 0:05 to about 1:05:46. Slides 2 and 34 (identical reminders,
  announcing the guest lectures by Archit Sharma and Aviral Kumar) and 33 (the plan repeated) are not
  discussed. The first part, to about 24:48, teaches CQL aloud, which Lecture 7's recording skipped.
  Board work: the IQL recap's example values, the CQL objective over the critic's parameters ("the
  notation on the slides is slightly different", ≈13:52), a lower-bound sketch, and a 2D picture of
  goal states and a decision boundary; the wiki reconstructs them from the spoken words.
- **Figure audit:** Opus parent, six 1400px renders (9, 15, 23, 27, 30, 32) against the Sonnet
  transcription. 9 (every table cell, the architecture's arrows) and 32 agree. Corrections: on 30, the
  dark-grey "With Chain of Thought" series' last three points were (78, 113), (82, 157), (90, 190), now
  (49, 114), (78, 151), (84, 191); on 27, the green Heading curve was "cut off by the right edge" at its
  peak, but the peak (about 2.3 at 0.87) is inside the plot and the curve is cut off after it, and the
  agent's "three-lane road" is not legible at that size; on 15, the second classifier photo has the
  pencil case to the right of the notebook and further back, not "in front of (below)" it; on 23, the
  green trajectory ends level with the star, not above it. The agent also described the guest lecturers'
  photographs by apparent gender; rewritten with the printed names.
- **Printed slips (checked on the page):** slide 29's third box ("RL from human feedback") repeats the
  second box's sub-line "on higher-quality (prompt, response)"; the lecturer describes the step as
  preference data (≈59:38). Noted in the slide file and the wiki. Recorded from the agent's list without
  a page check, so not claimed as slips: slide 4's IQL lines in a plain E beside the critic objective's
  blackboard-bold E; slide 8's inner expectation subscripted only pi where 5 and 6 have a' drawn from pi;
  slide 16's "classifiers weaknesses" without an apostrophe.
- **Transcript edit:** Sonnet, given the parent's glossary (pi beta/theta, V pi/Q pi, Q-value garbles,
  out-of-distribution, RL, phi, mu, behavior cloning, log-sum-exp, Ed, A/B, GAN, taus, dispreferred,
  ChatGPT, PPO, adversarial training, K choose 2, 0.5, the sound tags). 86/86 markers; 0 numbers lost;
  20 gained, all glossary restorations, L2's digit, or spoken number words written as digits ("eight",
  "five", "zero", "one"), contrary to the prompt's keep-as-is instruction but harmless, so accepted.
  0 of 86 paragraphs outside the band. 17 student questions and 4 answers; the parent moved one question
  marker (15:23) from after its restatement to before it. 4 [Ed:] notes: "Fine" read as phi (8:29), "a
  function of f" (10:02), "Q hati" (13:52), "mop out ... this step" (15:23). Agent restoration outside the
  glossary: a repeated clause at 13:06 ("is not a very good marker that's inside the supported of the
  data") dropped as a false start. Caption sound tags (three snorts, one sigh) removed.
- **Unclear in the recording, said so on the wiki:** the Q-value of the first action in the IQL
  example (written on the board, never read out, ≈3:55); the symbol Finn names in answer to "Q or Q
  hat?" (≈13:52); the verb in the question about pushing down Q outside the support (≈14:37–15:23);
  the CQL discrete-action read-out, captioned "maximum probability" (≈17:45), where slide 7 gives the
  argmax of Q.
- **Images:** rendered 18 (4, 5, 6, 8, 9, 12, 15, 17, 19, 20, 23, 25, 26, 27, 28, 29, 30, 32), 1.9MB:
  the thirteen figure-bearing slides other than the guest photographs (2, 34) and the black video frame
  (13), plus five algorithm and objective slides for their structure (6, 8, 17, 25, 26), following
  Lectures 5–7. Not rendered: 7, 18, 24 (loop arrow, emoji and brace transcribed; 24 repeated on 25);
  1, 3, 10, 11, 14, 16, 21, 22, 31, 33 (text).
- **Embedding:** the script placed 18 under their slide-file headings. Lecture page by hand; the
  embed dry run had proposed 4 and 5 after sentences that run on into displayed equations ("the
  off-policy critic objective (slide 4) is", "Slide 5 writes it ... learned Q-function:"), so both were
  placed after the equations instead. Topic pages by hand with "Lecture 8, slide N" captions.
- **Math:** the GitHub check found two inline formulas ending `)$)` in the slide file (slide 15–16's
  classifier labels) and a tau followed straight by a plural s in the wiki; rewritten.
