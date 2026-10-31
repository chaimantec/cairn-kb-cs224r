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

- [ ] 19 Tutorial Session: Review of Q-Learning — video 07MQNMcxhZU — `section_q_learning_tutorial.pdf`

## Run 19 — lecture 19 (Tutorial Session: Review of Q-Learning)

Run 19, 2026-11-01. Scope: lecture 19 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 19
- [x] raw/transcripts/original/19-q-learning-tutorial.md — verbatim captions (65 paragraphs, ~8,600 words; catalog length 3,039 s; last caption ≈49:49)
- [x] Deck confirmed by its title slide: "CS 224R Tutorial / Review of Q-Learning", Anikait Singh, matching the video's first sentence (≈0:05)
- [x] Slide number map: 35 pages; the deck prints a number only on its title page ("1", bottom centre, read from the spans), so every label is the page index, 1:1
- [x] raw/slides/19-q-learning-tutorial.md — deck transcribed (two Sonnet agents, pages 1–17 and 18–35 into scratch files, assembled by the parent), `--verify` OK (35 headings), check_math 180 formulas, 0 errors after the parent's fixes (stars as `\ast`, spaced underscores, `\cr` in slide 33's cases, three `$$` blocks inside list items made inline, quotes around inline math removed); header, numbering, notation notes, section table and recording coverage added by the parent, and transcript notes under slides 16, 20, 25, 29, 31, 32–34
- [x] Figure spot-check from 1400px renders (Opus parent; transcriber was Sonnet): 4, 15 (every cell of the four grids), 19, 24, 27, 31 (average errors), and slide 20's equations from a render. All agree. Grid builds 8–14 checked arithmetically against slide 7's update with discount 0.9 (each new value is 0.9 times the best next-state value, rounded): consistent throughout, including A* = Q* − V* on 14 (−1.7, −1.6)
- [x] raw/transcripts/19-q-learning-tutorial.md — copy-edit (Sonnet) with the parent's glossary of 50 restorations; check_transcript.py: 65/65 markers identical; 1 number lost and 5 gained, all deliberate (081 → 0.81 at 11:40; 18, 28, 29 and 59 in [Ed:] notes); 0 of 65 paragraphs outside 0.72–1.10, overall 0.924. Student turns: three captioned questions kept in quotation marks after `*[Student question]*` (7:47, 10:09, 12:26 across 13:13), five inaudible (19:20, 30:32, 37:28, 41:19 twice). Agent's additions beyond the glossary: "key value" → Q value also at 19:20, "endstep returns" at 30:32, and "that both of them" → "but both of them" at 34:24. Restorations table's "confirmed by" corrected by the parent for 0:05, 0:51, 11:40 and 14:44. Proper nouns: Anikait (slide 1), Polyak (26), Huber (27), SAC-N and EDAC (30) in the deck; Sutton and Barto, Decision Transformer and LoRA spoken only. [Ed:] notes at 18:32, 27:49 and 42:52 (two)
- [x] wiki/19-q-learning-tutorial.md — lecture page; all 21 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 13 quotations, 98 citations, 0 unmatched; check_math 172 formulas, 0 errors. The TD loss on the board (≈35:09–38:14) is reconstructed from the spoken words and marked so
- [x] Topic pages: no new topic page (the session reviews Lecture 6, and every topic it covers has a page); new sections in q-learning (worked through in the tutorial, with Lecture 19's slide 31 image), bellman-equations (iterating the Bellman equation) and mdp-and-notation (notation in the Q-learning tutorial); paragraphs in policy-evaluation (the derivation and table, with slide 21's image), value-functions (the grid world's optimal objects, with slide 13's image), replay-buffers (in Q-learning), offline-rl (stitching; critic ensembles offline) and course-logistics (Singh among the staff; the Homework 2 preview); where-this-appears entries in those and in actor-critic and policies; forward link from 06-q-learning (slide 2's extra section) and its related pages. check_quotes over all wiki 1797 quotations, 3050 citations, 0 unmatched; check_math wiki 3409 formulas, 0 errors; check_links 2533 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage now all 19 recordings and 356 images, the sentence saying the KB cannot answer Lecture 19 removed; the audible-students note in How to read; Lecture 19's recording notes; lecture entry; ten topic entries)

### Images
- [x] raw/images/19-q-learning-tutorial/ — 21 slides rendered (2.1MB on disk) by `extract_slide_images.py` (map 1:1, labels inferred from page index); embedded in the slide file (21, script) and the lecture page (21, by hand, hand-written captions); three also on topic pages (13, 21, 31, with Lecture 19 captions); `--verify` OK after the insertions
- [x] INDEX.md images paragraph ("Every lecture, 1 to 19, has images"); AGENTS.md table row; LICENSE.md attribution (Layer 2 now "Lecture 1–19 decks"; the deck's text layer has no licence notice). General lesson in AGENTS.md: keep captioned student words in quotation marks after `*[Student question]*`

### Publish
- [x] kb.json — coverage 19 of 19, images 356 (lecturesWithImages 19), decks transcribed 19, figure audit 19 of 19, caveats (the stale first caveat, "Only Lectures 1-17 of 19 are covered", left over from run 17, removed; Lecture 19's coverage, whiteboard and captioned students; the transcripts caveat now notes Lecture 19's audible students)
- [x] BUILD_LOG.md — lecture 19's record (the Findings below move with this section); index rows for Lectures 6 and 19
- [x] sources.md "In this KB" column, and the deck match confirmed by the title slide
- [x] SEE_ALSO.md: no change; the listed siblings have no Q-learning tutorial to add
- [x] verify_kb.py clean (after staging the new files), review section read (no ranking claims; numeric claims checked: 356 images, 21 of 35 for Lecture 19; distribution 21 on disk, 21 in the lecture page); its GitHub math check passed after the parent put blank lines around five display blocks in the slide file and removed math from one alt text and one wrapped inline formula. The two bad anchors check_links reports point at this run's BUILD_LOG heading, which exists once this section moves
- [ ] Commit and push
- [ ] Move this run's section to BUILD_LOG.md; tick lecture 19 in the backlog; TODO.md keeps only open work

### Findings

- **Lecturer:** Anikait Singh, a TA ("I'm a TA here for CS224R", ≈0:05; name from slide 1, captioned
  "Anakiette"). Pronouns not stated; the wiki writes "Singh". Slide 2 thanks Pieter Abbeel, David Silver,
  Aviral Kumar and Justin Fu.
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, creative
  commons, rights reserved). Third-party material: an uncredited cartoon grid world and map (4, 7–12), robot
  door frames with a Q-value plot and paddle-game frames with action-value bars (19, uncredited), a maze credited
  "Should I Use Offline RL or Imitation Learning?" (24), gradient-clipping contours and a Huber plot (27), and a
  paper figure (31) that Singh attributes to the double Q-learning paper; the title page's game screenshots are
  not rendered.
- **Deck:** 35 pages, 1:1. Only the title page prints a number ("1", bottom centre, from the spans);
  `slide_number_map.py` found none, and every label is the page index. Transcribed by two Sonnet agents in
  parallel (1–17 and 18–35 into scratch files, assembled by the parent); `--verify` passes.
- **Recording coverage:** about 51 minutes (catalog 3,039 s; last caption ≈49:49), ending on slide 31 during
  the critic-ensemble discussion; slides 32–35 (the Homework 2 walkthrough: DQN and SAC pseudocode, thanks) are
  not in it. Slide 31's figure is shown first at ≈46:47, before slide 30. The whiteboard carries the MC/TD/n-step
  derivation (≈21:39–28:59; slide 20 prints its lines) and the TD loss (≈35:09–38:14; reconstructed on the wiki
  and marked so). Slide 17's pause passes without a question; slide 25's is "Any questions about this?".
- **Students:** unlike Lectures 1–18, three questions are captioned (7:47, 10:09, 12:26 across 13:13) and kept
  in quotation marks; five inaudible ones are marked (19:20, 30:32, 37:28, 41:19 twice).
- **Lecturer's correction:** none of a printed slide. Singh corrects two spoken claims: the advantage "should
  always be negative" (≈14:44), then zero for the policy's actions (≈15:30); and sampling as a second cause of
  overestimation, withdrawn (≈44:28–45:14). Singh calls slide 31's orange curve "the red one" (≈49:04).
- **Figure audit:** Opus parent, 1400px renders of 4, 15, 19, 24, 27 and 31, and slide 20's equations: all
  agree, including every cell of slide 15's four grids and slide 31's average errors. The Q-value-iteration
  builds (8–12) and slides 13–14 were checked arithmetically against slide 7's update with discount 0.9 (each new
  entry is 0.9 times the best next-state value, rounded; A* = Q* − V*): consistent throughout.
- **Printed oddities:** slides 6–12 put $a_t$ under the max of a term in $a_{t+1}$ (text layer: "max | at |
  Q*(st+1, at+1)"); slide 23 prints "1 Steps"; slide 28 "They … prevents recency bias"; slide 29 "there
  results in error"; slide 30 "Sac-N"; slides 26 and 30 "Practical Implementation Detail" against "Details"
  on 27 and 28; slide 33's title "Q-Learning : DQN"; slide 14's sentence "how much better $\mathbf{a}_ t$ is"
  ends there. Notation is mixed (bold on slide 5 and slide 14's formula, plain elsewhere, capital $S_t$ on 20,
  $w, w'$ on 26 against $Q_\theta, Q_{\text{target}}$); the slide file's header says so.
- **Transcript edit:** Sonnet, given the parent's glossary of 50 restorations (Anikait, Markov, V pi, Q value for
  "key value", grid world, Bellman, 0.81, the lava state, argmax, policy gradient, Q star and Q pi, colab,
  scalar, Sutton and Barto, n-step, bias, agentic, Polyak, MSE-style, loss landscape, gradient clipping, Huber,
  tuples, Decision Transformer, Q approx, EDAC and SAC-N, LoRA, the student turns). The agent's additions were
  "key value" at 19:20, "endstep returns" at 30:32, and "but" for "that" at 34:24. [Ed:] notes at 18:32
  ("not conditional on state" as said), 27:49 (captions break off), 42:52 (slide 29's footer; "RD").
- **Unclear in the recording, said so on the wiki:** the board itself (both passages); whether 0.99 is $\tau$
  or $1 - \tau$ in slide 26's Polyak formula; the inaudible questions, known only from the answers.
- **Images:** rendered 21 (3, 4, 5, 7–15, 18, 19, 21, 22, 24, 27, 28, 30, 31), 2.1MB. Not rendered: 1 (title,
  with game screenshots) and 2 (outline); 20, 23, 26, 29, 33 and 34 (equations, a table and pseudocode
  transcribed exactly); 6, 16, 17, 25, 32 and 35 (text only). Every topic the session dwells on has a
  figure-bearing slide except the semi-gradient loss on the board (≈35:09–38:14), which the wiki carries in prose.
- **No new topic page:** the session reviews Lecture 6 and every topic it covers has a page; it updates eleven.

