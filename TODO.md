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

- [ ] 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`
- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 18 — lecture 18 (Frontiers)

Run 18, 2026-10-31. Scope: lecture 18 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 18
- [x] raw/transcripts/original/18-frontiers.md — verbatim captions (92 paragraphs, ~12,800 words; catalog length 4,249 s; last caption ≈1:10:22)
- [x] Deck confirmed by its title slide: "Frontiers & open problems in deep RL + How to do (deep RL) research", matching the video's first sentence (≈0:05). The deck prints no lecturer name
- [x] Slide number map: 53 pages, 1:1. The map script reads 1–51 at bottom centre except 23, which prints at bottom right (x≈1830, read from the spans); 40, 52 and 53 print none, labelled by page index (no gap)
- [x] raw/slides/18-frontiers.md — deck transcribed (two Sonnet agents, pages 1–27 into the slide file, 28–53 into a scratch file appended by the parent), `--verify` OK (53 headings), check_math 5 formulas, 0 errors; header, numbering, section table and recording coverage added by the parent, and the literal dollar sign on slide 18 rewritten
- [x] Figure spot-check from 1400px renders (Opus parent; transcriber was Sonnet): 5, 10, 15, 22, 23 (table from a crop), 26, 32, 39. Agree: 10, 23 (every cell, Average row), 26, 32. Corrected: 5 (seven rows, not eight), 22 (the post-training panel's bins against the diagonal), 39 (about ten arms); added: 15 (bar shades), 10 (RLPD line style). Slide 28: item c is printed under the black line (text layer), not missing. Video flags checked against the transcript: kept on 39's two panels (≈44:53); 8 and 12 noted as frames of a YouTube video and of Veo videos, not said to play; removed on 6, 19, 23, 31 and 39's lab photograph
- [x] raw/transcripts/18-frontiers.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 92/92 markers identical; 1 number lost and 2 gained, all deliberate (GBT40 → GPT-4o at 30:10, the 40 lost and a 4 gained; 43 in the [Ed:] note at 56:25 citing slide 43); 0 of 92 paragraphs outside 0.72–1.10, overall 0.904; 5 student-question markers (19:22, 32:29, 51:01, 52:33, 1:01:04). Agent's additions beyond the glossary are standing spellings (chatbots, safety-critical, real-world, state-of-the-art, idea-/problem-driven) and four false starts removed. Every restored name (Sharma, Tong, Ball, Dong, Tian, Luo, AlphaFold, CASP, GPT-4o) is in the deck; DPO is spoken only, in an answer. [Ed:] notes at 33:16 (competent/confident) and 56:25 (a feeling / something)
- [x] wiki/18-frontiers.md — lecture page; all 15 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 183 quotations, 161 citations, 0 unmatched; check_math 7 formulas, 0 errors
- [x] Topic pages: no new topic page (the research half has no counterpart elsewhere in the course, so the lecture page holds it); new sections in reward-learning (domains with no reward to verify), rlhf (what preferences reward, and calibration), offline-and-online (batch online RL, and prior data), model-based-rl (video generation models as world models), exploration (optimizing for learning, exploring safely), replay-buffers (prior data in the buffer, not the weights), a paragraph in expressive-policy-distributions' evidence and in course-logistics (slide 2's reminders; research advice); where-this-appears entries in those and in preference-learning, ppo, value-functions, llm-reasoning, offline-rl and rl-algorithm-families (coverage now Lectures 1–18); forward links from 01-class-intro, 05-off-policy-actor-critic (HIL-SERL), 09-rl-for-llms and 11-model-based-rl. check_quotes over all wiki 1782 quotations, 2931 citations, 0 unmatched; check_math wiki 3176 formulas, 0 errors; check_links 2399 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–18 and 335 images, Lecture 19 the only lecture without technical content; Lecture 18's recording notes, lecture entry, fourteen topic entries updated, course-logistics entry)

### Images
- [x] raw/images/18-frontiers/ — 15 slides rendered (1.7MB) by `extract_slide_images.py` (map 1:1); 31 and 38 rendered then dropped (decorative photograph; the bottleneck pun); embedded in the slide file (15, script) and the lecture page (15, by hand, hand-written captions); `--verify` OK after the insertions
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (Layer 2 now "Lecture 1–18 decks"; the deck's text layer has no licence notice). No new general lesson for AGENTS.md

### Publish
- [x] kb.json — coverage 18 of 19, images 335, figure audit 18 of 18, decks transcribed 18, caveats (Lecture 18's coverage, questions and video-tied slides; 2 [Ed:] notes)
- [x] BUILD_LOG.md — lecture 18's record (the Findings below move with this section); index rows for Lectures 1, 5, 9, 11 and 18
- [x] sources.md "In this KB" column, and the deck match confirmed by the title slide
- [x] SEE_ALSO.md: no change; no sibling KB covers research practice, and the RLHF and reasoning siblings are already listed
- [x] verify_kb.py clean (after staging the new files), review section read (no ranking claims; numeric claims checked: 335 images, 15 of 53 for Lecture 18; distribution 15 on disk, 15 in wiki); its GitHub math check passed (5778 formulas)
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 18 in the backlog; TODO.md keeps only open work
- [ ] Backlog line, moved from TODO.md: 18 Frontiers — video FacJ_1tTSx4 — `18_cs224r_frontiers_how_to_research.pdf`

### Findings

- **Lecturer:** Chelsea Finn, the course instructor; the deck prints no lecturer name, but it is the course's
  last lecture (slide 53's thank-you) and the speaker speaks of advising PhD students. Pronouns not stated; the
  wiki writes "Finn". Slide 5's "Sharma & Tong et al., 2023" is a citation, not Lecture 9's guest lecturer;
  the pages do not equate them.
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, creative
  commons, rights reserved). Third-party material: the Sharma & Tong preference chart (5), robot shirt photos
  (6), Zhao et al.'s recommendation paper page (7), a playroom photograph credited to a YouTube video (8), Ball
  et al.'s AntMaze chart (10), Veo frames (12), Dong et al.'s bar chart (15), TIME and a second news site's
  headlines (18), a cyclist photograph (19), a chat screenshot and a quotation cited to Rajpurkar & Topol and
  Goh et al. (21), OpenAI's and Tian et al.'s calibration plots (22), Luo et al.'s photos and table (23), an
  x.com screenshot with an NBC News headline (26), AlphaFold renderings and logo (32), and Finn's internship
  photographs and video frames (39).
- **Deck:** 53 pages, 1:1. Pages 1–39 and 41–51 print their number, at bottom centre except page 23, at bottom
  right (x≈1830), which `slide_number_map.py`'s corner read missed and the spans show; 40, 52 and 53 print none
  and are labelled by page index (no gap). Slides 5–8 carry a second, smaller copy of the number at the lower
  right. Transcribed by two Sonnet agents in parallel (1–27 into the slide file, 28–53 into a scratch file
  appended by the parent); `--verify` passes.
- **Recording coverage:** about 71 minutes (last caption starts ≈1:10:22), opening on the plan for both parts
  and ending on slide 52's closing thoughts. Slide 2's reminders (Homework 4 due today, poster session next
  Wednesday, final report the following Monday with no extensions or late days) are not mentioned; slide 53's
  thank-you is not read aloud. Five student questions: ≈19:22 (offline methods such as DPO), 32:29 (restated
  only as "why does this happen"; the answer is about calibration after RLHF, slide 22), 51:01 (problems with
  few ideas), 52:33 (an example of front-loading a core unknown), 1:01:04 (speed against thorough
  evaluation). Video frames: slide 39's two panels (≈44:53–45:38); slides 8 and 12 hold frames of a YouTube
  video and of Veo videos, and the recording does not say whether they played.
- **Lecturer's correction:** none of a printed slide.
- **Figure audit:** Opus parent, 1400px renders of 5, 10, 15, 22, 26, 32 and 39, and a crop of slide 23's
  table. Agree: 10, 23 (every cell, including the Average row, 49.7 against 100), 26, 32. Corrections: 5 (seven
  rows, not eight: the faded "Ma…" label is the top row's own label under "Agrees with me"), 22 (the
  post-training panel: bins 2 and 3 above the diagonal, bin 4 on it, bins 5–10 below), 39 (about ten arms, not
  eight). Added: 15 (Pre bars lighter, Post bars darker), 10 (RLPD's black line with a white dashed centre).
  Slide 28: the transcriber said item c was not printed; the text layer has "c. Using world models" at the
  height of the black line laid over it, so it is printed and covered.
- **Printed oddities:** slides 33, 41, 45 and 49 print "How do to (deep RL) research"; slide 21 prints "Human are
  often the recipient"; slide 33's brace note reads "first three are equally important!" where 41, 45 and 49
  read "equally important!"; slide 4's outline letters run a to g across the three numbered groups; slide 18
  prints a dollar sign in a headline, written in the slide file as "2 Dollars" with a note.
- **Transcript edit:** Sonnet, given the parent's glossary (standing spellings; Sharma and Tong at 1:38, Ball et
  al. at 8:36, Dong et al. at 18:36, Tian et al. and logits at 27:04, Luo / et al. across 27:50 and 28:37, RLHF,
  PPO, GPT-4o, gooder at 30:56 from slide 26's screenshot, meta-learning, sinusoid, drawing board, front-load,
  five student-question markers, caption sound tags dropped). The agent's additions were standing spellings and
  four false starts. Proper nouns: Sharma, Tong, Ball, Dong, Tian, Luo, AlphaFold, CASP, ChatGPT, GPT-4o,
  OpenAI and Google all appear in the deck; DPO is spoken only, in an answer.
- **Unclear in the recording, said so on the wiki:** a word in the answer on calibration (≈33:16, "competent",
  possibly "confident"); the phrase "starting with a feeling that works" (≈56:25), where slide 43 prints "Start
  from something that works" (transcript only); what the student questions at ≈19:22 and 32:29 actually asked.
- **Images:** rendered 15 (5, 6, 7, 8, 10, 12, 15, 18, 19, 21, 22, 23, 26, 32, 39), 1.7MB. Not rendered: 31 (a
  decorative photograph of an indoor space, never mentioned) and 38 (the "Bottlenecks" pun of four bottles),
  both rendered and then dropped; 1, 3, 29, 52 (title and section slides); 4, 9, 11, 13, 16, 20, 24, 28, 33,
  41, 45, 49 (agenda and outline slides); 23 text-only slides. Every topic the lecture dwells on has a
  figure-bearing slide except the second part's advice (≈38:43–1:10:50), which is text slides throughout and
  which the wiki carries in prose.
- **No new topic page:** the research half has no counterpart in another lecture, so the lecture page holds
  it; the frontiers half updates thirteen topic pages instead.

