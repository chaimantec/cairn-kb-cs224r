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

## Run 9 — lecture 9 (RL for LLMs)

Run 9, 2026-10-29. Scope: lecture 9 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 9
- [x] raw/transcripts/original/09-rl-for-llms.md — verbatim captions (81 paragraphs, ~10,300 words, ~63 min)
- [x] Slide number map: 57 pages, printed 1–99, not 1:1 (42 hidden slides not exported). Numbers bottom left on every numbered page; pages 1, 12, 22, 24, 34, 39, 48, 49 print none (12, 22, 48, 49 ambiguous). Page 57's bottom-right "4" is not a slide number
- [x] raw/slides/09-rl-for-llms.md — deck transcribed (Sonnet), `--verify` clean (57 headings); recording-coverage paragraph and slide 50's citation note added by the parent
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 72, 78, 94 (every series and value), 61 agree; 50 (table cell by cell agrees; decoder sketch has two rows of red boxes) and 87 (three orange diamonds, not two, in the type column at 200 dpi; the two hard-to-read model names confirmed) small fixes. Printed slips checked in the text layer and on renders: 56, 70, 50, 78, 99
- [x] raw/transcripts/09-rl-for-llms.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 81/81 markers identical; lost 40 ×2 and 5 ×2, gained 4 ×2, 2 and 2027 ×2, all glossary restorations (GPD40 to GPT-4o, RM5 to RM phi, NC2 to N choose 2) or the AI 2027 [Ed:] note; 0 of 81 paragraphs outside 0.72–1.10 (overall 0.945); 25 student questions and 1 answer marked; 6 [Ed:] notes
- [x] wiki/09-rl-for-llms.md — all 24 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 89 quotations, 101 citations, 0 unmatched
- [x] Topic pages: instruction-finetuning, rlhf, direct-preference-optimization (new); preference-learning (its "does not cover it yet" line replaced: Bradley-Terry, Plackett-Luce), rl-objective (a KL-penalized objective and its closed-form optimum), policy-gradients (for language models), ppo (PPO in InstructGPT, ChatGPT and slide 74's pipeline), mdp-and-notation (notation for preference optimization), offline-and-online (RLHF's online sampling against DPO's fixed data), reward-learning (over-optimized reward models), imitation-learning (instruction finetuning), rl-algorithm-families (coverage line now Lectures 1–9), 08-reward-learning and course-logistics (forward links) updated
- [x] INDEX.md table of contents and coverage (coverage note, Lecture 9's numbering and unrecorded slides, lecture entry, three new topic entries and nine updated, SEE_ALSO pointer)

### Images
- [x] raw/images/09-rl-for-llms/ — 24 slides rendered (2.8MB); embedded in the slide file (24, script), the lecture page (24, by hand, hand-written captions) and topic pages by hand with "Lecture 9, slide N" captions (instruction-finetuning 43, 50, 56; rlhf 61, 70, 72, 94; direct-preference-optimization 76, 77, 78)
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (and Layer 2's "Lecture 1–8 decks" now "Lecture 1–9"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 9 of 19, topic pages 25, images 146, figure audit 9 of 9, caveats (Lecture 9's numbering, unrecorded slides, board work and garbled number; its [Ed:] count)
- [x] BUILD_LOG.md — lecture 9's record (the Findings below move with this section); AGENTS.md gains one general line: a guest deck may revise one in a sibling KB; list it in SEE_ALSO.md but never borrow its slide labels
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md (new): CS224N (its Lecture 11 is an earlier version of this deck, same lecturer, with the prompting and chain-of-thought slides hidden here; its numbering differs) and CS336 (post-training lectures 15–16)
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 146 and 24 of 57 checked; distribution 24 on disk, 24 in wiki; slide-heading gaps are the deck's hidden slides); check_math --github: 3630 formulas, 0 errors (one inline formula opened after a quotation mark fixed first); check_quotes 0 unmatched on the lecture page and the eleven edited or new topic pages; check_links 1207 links, 0 broken
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 9 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Archit Sharma, guest lecturer (title slide; Lecture 8's slide 2 calls Sharma a
  researcher on the Gemini team and lead author of DPO). Pronouns not stated; the wiki writes "Sharma".
- **Licence:** none stated on the site or in the deck (text layer searched). The title slide reads
  "Based on slides in CS224N". Citations: The Economist and Sevilla et al. (2022) on 2; Andreas (2022)
  on 6; Larnerd (2023) on 9; Ouyang et al. (2022) on 39–40 and 80–81; Chung et al. (2022) on 43 and
  50–52 (50's bullet prints 2020); Raffel et al. (2018) on 50; Williams (1992) on 62–64; Knox and Stone
  (2009) on 65; Phelps et al. (2015) and Clark et al. (2018) on 66–67; Bradley-Terry (1952) on 67;
  Stiennon et al. (2020) on 72 and 94; Zheng et al. (2023) on 74; Dubois et al. (2023) on 88; Bai et
  al. (2022) on 97–99; Huang et al. (2022) and Zelikman et al. (2022) on 98–99; Kirk et al. (2024) and
  Singh, Hsu et al. (2024) on 99.
- **Deck:** 57 pages, printed numbers 1–99 bottom left, not 1:1 (42 hidden slides not exported).
  Pages 1, 12, 22, 24, 34, 39, 48, 49 print none; 12, 22, 48, 49 are ambiguous (lowest fitting label
  used). Page 57's bottom-right "4" is the last digit of "2024" in a figure caption. `--verify`
  passes. CS224N's KB has an earlier version of the deck (99 printed slides over 94 pages) whose
  numbering differs, so it could not settle the ambiguous labels.
- **Recording coverage:** about 63 minutes, slide 2 to the closing "What's next?" slides in order,
  except a jump at ≈58:14 to slide 95 (sycophancy) before returning to 92, 94, 96 and 99. Slides 81,
  93, 97 and 98 are not discussed. Board work: the cancellation of Z(x) in the DPO reward difference
  (≈42:37–44:11); the wiki reconstructs it from the spoken words and slide 77.
- **Figure audit:** Opus parent, six 1400px renders (50, 61, 72, 78, 87, 94) against the Sonnet
  transcription, plus a 200 dpi crop of slide 87's type column. 72, 78 and 94 (every series and value),
  61, and slide 50's table (cell by cell) agree. Corrections: slide 50's encoder-decoder sketch has two
  rows of red decoder boxes, inputs below and outputs above; slide 87 has three orange diamonds in the
  type column (NexoNimbus, SamirGPT, Lelantos), not two. The two hard-to-read model names were confirmed.
- **Printed slips (checked on the page or in the text layer):** slide 56 "there a mismatch" (no "is");
  slide 70 "drifting too for from"; slide 50 cites Flan-T5 as "Chung et al., 2020" in the bullet and
  2022 in the footer, and recalls T5 "from lecture 10" (a CS224N reference); slide 78's legend colours
  PFT dark green in the left chart and brown in the right; slide 99 says "by the next lecture (2024)".
  Noted in the slide file and the wiki. The agent's other candidates were not slips (slide 88's
  "Dubois" is printed as such; slides 62–64's switch from y to s is the generic notation Sharma
  announces at ≈20:49).
- **Transcript edit:** Sonnet, given the parent's glossary (GPT-3, InstructGPT, SFT, MMLU and
  BIG-bench, Iroh, Geoffrey Hinton, p PT and p RL theta, RM phi, Bradley-Terry, Gumbel-softmax,
  Kullback–Leibler, KL-constrained, Z of x, y w and y l, N choose 2, Plackett–Luce, sycophantic, GPT-4o,
  Llama and Mistral, PPO). 81/81 markers; numbers adjudicated above; 0 of 81 paragraphs outside the
  band. 25 student questions and 1 answer (27:02). 6 [Ed:] notes: the name of the reasoning lecturer
  (0:05), AGI 2027 against slide 2's AI 2027 site (0:05), a garbled reward value (24:42), "non-b
  blackbox" (27:02), "perpetrated" (48:09), "the sigma" (48:55). Agent restorations outside the
  glossary: "assistance" to "assistants" (3:54), "rational" to "rationale" (6:11, slide 43),
  "corpuses" to "corpora", a filler "U" dropped (31:45).
- **Unclear in the recording, said so on the wiki:** the first reward value in the scale example
  (≈24:42); the closed-form solution is stated without derivation, as in the lecture (≈40:18).
- **Images:** rendered 24 (2, 4, 7, 41, 42, 43, 50, 56, 61, 63, 64, 67, 70, 72, 74, 76, 77, 78, 80, 87,
  92, 94, 98, 99), 2.8MB: every figure-bearing slide except a duplicate of slide 4 (12) and slides whose
  pictures are screenshots or panels of text transcribed in full, or clip-art (6, 8, 9, 10, 39, 40, 51,
  52, 60, 65, 81, 82, 83, 84, 85, 88, 93, 95, 96, 97), plus nine diagram and equation slides for their
  structure (41, 42, 56, 63, 64, 67, 70, 76, 77). Not rendered: 1, 5, 13, 14, 57, 62, 66, 69, 73, 75,
  79, 89 (title, plan, dividers, text and equations).
- **Embedding:** the script placed 24 under their slide-file headings; the lecture page by hand (embed
  dry run: nothing missing); topic pages by hand with "Lecture 9, slide N" captions.
- **Math:** the GitHub check found one inline formula opened straight after a quotation mark on the
  lecture page; rewritten.

