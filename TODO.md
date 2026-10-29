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

## Run 10 — lecture 10 (RL for LLM Reasoning)

Run 10, 2026-10-29. Scope: lecture 10 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 10
- [x] raw/transcripts/original/10-rl-for-reasoning.md — verbatim captions (92 paragraphs, ~15,100 words, ~70 min)
- [x] Slide number map: 41 pages, printed 1–41, 1:1. Numbers bottom right; pages 6, 31, 32, 35, 36, 40 print none in the text layer (raw spans read; labels follow the 1:1 sequence, unambiguous)
- [x] raw/slides/10-rl-for-reasoning.md — deck transcribed (Sonnet), `--verify` clean (41 headings); recording-coverage paragraph, animation notes (10, 19, 23) and slip notes added by the parent; math fixed for GitHub (underscores, `\gt`, prose dollar signs and asterisks)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 11 (every series and value), 17 (every Q-value, advantage and arrow), 23, 40 (every bar) agree; 27 agrees but for where the 9B chart's 5× arrow starts; 35's early points had been placed one step late (Verifications and Backtracking re-read at every step). Printed slips checked in the text layer and on renders: 19 "GMS8k", 14 "multiple Eq1" (missed by the agent), 31 "Thoughtolgy", 40 "browsing**", 8 against 24–28's footers (title and venue)
- [x] raw/transcripts/10-rl-for-reasoning.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 92/92 markers identical; lost 01, .15, .05, 4 and gained 0.15, 0.05, 24, 1, 0 ×6, 5 ×2, 6 ×2, 8, all glossary restorations (o1, D to the minus 0.15 and 0.05, NeurIPS '24, a0, pass@5) or [Ed:] notes; two slips fixed in the parent ("whether 8" had become "eight", "2 plus two" had become "2 plus 2"); 0 of 92 paragraphs outside 0.72–1.10 (overall 0.847); 31 student questions marked; 7 [Ed:] notes, one added by the parent at 1:04:39 (math and GSM8K) where the agent had capitalized "math" as the MATH benchmark
- [x] wiki/10-rl-for-reasoning.md — all 27 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 60 quotations, 129 citations, 0 unmatched
- [x] Topic pages: llm-reasoning and process-rewards (new); value-functions (a solution prefix's Q-value and advantage), mdp-and-notation (notation for RL for LLM reasoning), imitation-learning (rejection finetuning, spurious steps), implicit-policy-constraints (advantage filtering of steps), direct-preference-optimization (per-step preference pairs), policy-gradients (0/1-reward gradient), ppo (GRPO), offline-rl, reward-learning (process rewards and verifiers), offline-and-online, rlhf and rl-algorithm-families (coverage now Lectures 1–10) updated; 09-rl-for-llms and course-logistics forward links. check_quotes 180 quotations, 440 citations, 0 unmatched; check_math wiki 2152 formulas, 0 errors; check_links 1240 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–10 and 173 images, Lecture 10's recording coverage and animated charts, lecture entry, two new topic entries and eleven updated, SEE_ALSO pointer)

### Images
- [x] raw/images/10-rl-for-reasoning/ — 27 slides rendered (3.3MB); embedded in the slide file (27, script), the lecture page (27, by hand, hand-written captions) and topic pages by hand with "Lecture 10, slide N" captions (llm-reasoning 7, 11, 35; process-rewards 15, 17, 27)
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (animated charts export as their first frame; context-dependent glossary restorations apply only at their timestamps); LICENSE.md attribution (Layer 2 now "Lecture 1–10 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 10 of 19, topic pages 27, images 173, figure audit 10 of 10, caveats (Lecture 10's coverage, board work and animated charts; its [Ed:] count)
- [x] BUILD_LOG.md — lecture 10's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: CS329A added (test-time compute, verifiers and PRMs, train-time RL for reasoning); CS336's entry now points at its Lecture 16 (RLVR, GRPO)
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 173 and 27 of 41 checked; distribution 27 on disk, 27 in wiki); its GitHub math check: 3932 formulas, 0 errors; check_quotes 0 unmatched on the lecture page and the fourteen edited or new pages; check_links 1240 links, 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 10 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Aviral Kumar, guest lecturer (title slide, with the Carnegie Mellon University School
  of Computer Science logo). Kumar mentions teaching similar content in Kumar's own class (≈0:05).
  Pronouns not stated; the wiki writes "Kumar".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©,
  Creative, rights). Citations: Setlur et al., NeurIPS 2024 (slides 8, 10–23; Kumar an author) and
  Setlur et al., "Rewarding Progress" (slide 8 as ICLR 2025, footers of 24–28 as arXiv 2024; Kumar an
  author); Marjanović et al. (31); Gandhi et al. (35); Kimi team (36); Snell et al., ICLR 2025 (37–39);
  OpenAI o3 and o4-mini blog post (40). Slide 2's AIME solution is signed "~eevee9406". Slides 4 and 7
  print no source.
- **Deck:** 41 pages, printed 1–41, 1:1, numbers bottom right. Pages 6, 31, 32, 35, 36, 40 print none
  in the text layer (raw spans read; labels follow the 1:1 sequence). `--verify` passes.
- **Recording coverage:** about 70 minutes, slides 2–27 in order, then at ≈1:06:56 "I'm going to skip
  some of these things" and slide 32 from ≈1:07:42 to the end (about 1:10:30). Slides 28–31 and 33–41
  not discussed; slide 14's example and slide 26's pop-up shown but not read out. Board work: the
  language MDP (≈10:03–10:50, ≈46:58), the RFT algorithm (≈13:54–18:29), the per-step DPO algorithm
  (≈55:22–56:09). At ≈58:27 Kumar says the plots "were supposed to be gifs": slides 10, 19 and 23 hold
  only the first frame (a few points at the left edge), so their results come from the transcript.
- **Figure audit:** Opus parent, six 1400px renders (11, 17, 23, 27, 35, 40) against the Sonnet
  transcription, plus slide 19 for its printed title. 11 (every series and value), 17 (every Q-value,
  advantage and arrow), 23 and 40 (every bar) agree; 27 agrees except that the 9B chart's 5× arrow
  starts near x = 50, not 45. Correction: slide 35's first chart had its early points one step late
  (5.6 is at 30, not 40; a 7.0 at 40 was missing); Verifications and Backtracking re-read at every step.
- **Printed slips (checked in the text layer and on renders):** slide 19's title "GMS8k"; slide 14
  "multiple Eq1 by 2" (missed by the agent); slide 31 "Thoughtolgy" with a malformed accent in
  "Marjanovi´c"; slide 40 "browsing**" and a cut-off o4-mini label; slide 8's "Scaling up ... ICLR 2025"
  against the footers' "Scaling ... arXiv 2024". Noted in the slide file and the wiki.
- **Transcript edit:** Sonnet, given the parent's glossary (AIME, next-token, p theta and p star, GPT,
  o1 and the o-series, DeepSeek-R1, Claude Sonnet, Markov, s-naught and a0, r of 8 comma ground truth,
  NeurIPS '24, RFT and SFT, S tilde, pi tilde, GSM8K and MATH, DeepSeek V2 kept as spoken, D to the
  minus 0.15 and 0.05, spurious, Q of s, a and V of s, advantage-weighted regression, RLHF, DPO, D pref,
  Llama 2, pass@5, GRPO, PPO, ORM RL, advantage-induced policy alignment). 92/92 markers; numbers
  adjudicated above; 0 of 92 paragraphs outside the band (overall 0.847, the speech is filler-heavy).
  31 student questions marked. 7 [Ed:] notes in 5 paragraphs: D pref (55:22), the temperature "6"
  twice (1:00:46), "someplace solid the solid" (1:00:46), "first step", possibly per-step (1:01:32),
  "math" and "GSM8K" (1:04:39, added by the parent after the agent had capitalized "math" as the MATH
  benchmark), "deepc carbon" read as DeepSeek-R1 (1:05:25). Parent fixes: two number slips ("whether
  8" had become "eight"; "2 plus two" had become "2 plus 2"), a `|D|` that broke the restorations
  table, sharper "confirmed by" sources for eight restored names.
- **Unclear in the recording, said so on the wiki:** the sampling temperature (≈1:00:46); a remark
  about a removed plot line (≈1:01:32); the two datasets in the generalization answer (≈1:04:39); the
  model name at ≈1:05:25; the model in slides 10 and 23, captioned "DeepSeek V2" but printed "DeepSeek".
- **Images:** rendered 27 (3, 4, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
  28, 34, 35, 36, 39, 40, 41), 3.3MB: every figure-bearing slide except panels of text transcribed in
  full (2, 5, 31), slide 26's text pop-up over a faded build, and build steps whose content a later
  slide shows (33 for 34; 37 and 38 for 39). Not rendered also: 1, 6, 8, 13, 29, 30, 32 (title,
  outline, dividers, text).
- **Embedding:** the script placed 27 under their slide-file headings; the lecture page by hand
  (embed dry run: nothing missing); topic pages by hand with "Lecture 10, slide N" captions.
- **Math:** the slide file needed GitHub fixes (13 `}_{` underscores, `}_i`, `>` in inline math, math
  inside italics, inline math opened after a quotation mark, prose dollar signs in slides 12 and 14,
  intraword asterisks in slide 12's calculator annotations); the lecture page needed `pass@` written as
  math.
- **Parent mistake caught:** a `sed` on sources.md's "| none | not yet |" suffix also rewrote rows
  16–19, whose readings column is also "none"; restored before commit.

