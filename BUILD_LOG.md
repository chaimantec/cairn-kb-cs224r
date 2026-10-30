# Build log — CS224R knowledge base

The record of how this knowledge base was built and checked, one section per
finished run, in order: each deck's licence notices and lecture pointers, each
transcript's edit and checks, what each figure audit found, each deck's printed
slips, and which slides were rendered as images and why the others were not.
A run's section is its `TODO.md` checklist, moved here verbatim once the run
was committed and pushed. It holds no unchecked box: open work lives only in
`TODO.md`.

It is a build record, **not needed to answer questions**: every fact a learner
needs is on the wiki page, slide file or transcript it concerns. The rules
every build follows, and each lesson in general form, are in
[`AGENTS.md`](AGENTS.md).

## Index

| Lecture | Runs |
| --- | --- |
| 1 Class Intro | [Run 1](#run-1--lecture-1-class-intro-and-scaffold) (2026-10-28) |
| 2 Imitation Learning | [Run 2](#run-2--lecture-2-imitation-learning) (2026-10-28) |
| 3 Policy Gradients | [Run 3](#run-3--lecture-3-policy-gradients) (2026-10-28) |
| 4 Actor-Critic Methods | [Run 4](#run-4--lecture-4-actor-critic-methods) (2026-10-28) |
| 5 Off-Policy Actor Critic | [Run 5](#run-5--lecture-5-off-policy-actor-critic) (2026-10-28) |
| 6 Q-Learning | [Run 6](#run-6--lecture-6-q-learning) (2026-10-28) |
| 7 Offline RL | [Run 7](#run-7--lecture-7-offline-rl) (2026-10-29) |
| 8 Reward Learning | [Run 8](#run-8--lecture-8-reward-learning) (2026-10-29) |
| 9 RL for LLMs | [Run 9](#run-9--lecture-9-rl-for-llms) (2026-10-29) |
| 10 RL for LLM Reasoning | [Run 10](#run-10--lecture-10-rl-for-llm-reasoning) (2026-10-29) |
| 11 Model-Based RL | [Run 11](#run-11--lecture-11-model-based-rl) (2026-10-29); [Run 12](#run-12--lecture-12-multi-task-rl) (2026-10-30, pointers to Lecture 12's teaching of the deferred slides) |
| 12 Multi-Task RL | [Run 12](#run-12--lecture-12-multi-task-rl) (2026-10-30) |
| 13 Meta RL | [Run 13](#run-13--lecture-13-meta-rl) (2026-10-31) |

## Runs

_Each finished run's section is appended below._

## Run 1 — lecture 1 (Class Intro) and scaffold

Run 1, 2026-10-28. Scope: lecture 1 only, plus the repo scaffold, crawl and publish. Images on, PDFs
link-only, commit as chaimantec <nantapon@chaimantec.com>.

### Scaffold and crawl
- [x] init_kb.sh; repo-local git identity
- [x] Crawl https://cs224r.stanford.edu/spring_2025/ (excluding the student project archive and the 2023 offering): 19 decks + 6 handouts on disk, 197MB, gitignored
- [x] sources.md — per-lecture table by hand (links only), with the site's readings and homework dates

### Lecture 1
- [x] raw/transcripts/original/01-class-intro.md — verbatim captions (69 paragraphs, 9,460 words)
- [x] raw/slides/01-class-intro.md — deck transcribed (Sonnet), `--verify` clean (45 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 24, 25, 30, 33, 36, 37 all agree; slide 30 given the lecture's context, one slip note corrected
- [x] raw/transcripts/01-class-intro.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 69/69 markers identical, 0 numbers lost (4 gained, all in [Ed:] notes or the "1 million dollars" spelling), 0 of 69 paragraphs outside 0.72–1.10; 22 student turns marked; 3 [Ed: unclear] spots
- [x] wiki/01-class-intro.md
- [x] Topic pages: mdp-and-notation, policies, rl-objective, value-functions, rl-algorithm-families; course page course-logistics
- [x] INDEX.md table of contents

### Images
- [x] raw/images/01-class-intro/ — 16 slides (1.9MB) rendered, embedded in the slide file, the lecture page (16) and three topic pages (5); 16 hand-written captions
- [x] INDEX.md — the chat's image rules; AGENTS.md — Images conventions and table row; BUILD_LOG.md — what was rendered and why; LICENSE.md — attribution

### Publish
- [x] kb.json — coverage 1 of 19, method page-images, four caveats
- [x] BUILD_LOG.md — lecture 1's record (the Findings below move with this section)
- [x] verify_kb.py clean, review section read; check_math.mjs --github: 307 formulas, 0 errors; check_quotes 0 unmatched; check_links 0 broken
- [x] Commit and push (chaimantec/cairn-kb-cs224r, public): 910f655; check_math --github-published 308 formulas, 0 errors
- [x] PATCH kbUrl onto the catalog entry; re-fetched, kbUrl = https://github.com/chaimantec/cairn-kb-cs224r
- [x] Move this run's section to BUILD_LOG.md; TODO.md keeps only open work

### Findings

- **Licence:** none stated on the site (checked 2026-10-28) or on any page of the Lecture 1 deck.
- **Recording gap:** the video skips the logistics slides 5–6 and 10–15 (slide 9 to slide 17 at
  ≈4:43); slides 1–3 and 45 are not discussed. Recorded in the slide file, transcript header,
  wiki page, course-logistics page and kb.json caveats.
- **Deck:** 45 pages, printed number = page, 1:1. Pages 25 and 29 print no number. Page 30's
  "30" is in the text layer but covered by the image (the deck agent noticed; the map script
  cannot).
- **Figure audit:** Opus parent, six 1400px renders (24, 25, 30, 33, 36, 37) against the Sonnet
  transcription. All six agreed on structure, arrows, labels and equations. Two fixes: slide 30
  given the lecture's context (the hockey-puck reset video, ≈13:53–14:39) and slide 26 the toy
  airplane task (≈11:35); the agent's slip note that slide 37's $t+1$ subscript is not bold was
  wrong (the product prints it bold), corrected.
- **Printed slips:** slide 45 "due Weds 4/18" (slide 15 and the site: Friday; April 18, 2025 was
  a Friday); slide 29 "offine RL"; slide 34 writes the chatbot reward as r(s, a) in an
  observation column; slide 4 "Key learnings goals"; bold "t+1" subscripts on slides 37, 38, 40
  and 43 (cosmetic).
- **Spoken slip:** ≈43:09 "plus the product" where the slides multiply; kept and marked [Ed:].
- **Transcript edit:** Sonnet, given the parent's glossary of 13 decided restorations. It added
  i.i.d., roll-out, Q-function and reset-free from the slides. Three [Ed: unclear] spots remain
  (≈9:19 "discovery town", ≈13:08 "into the sink", ≈23:07 "how the power"). One student-turn
  attribution is uncertain (≈40:49, non-stationary decision processes, marked as a student
  answer).
- **Images:** rendered 16 (19–21, 23–28, 30, 32–37). Not rendered: text-only, title and agenda
  slides (1, 2, 4–7, 9–12, 14–18, 29, 31, 39, 41, 42, 44, 45); staff photographs (3); clip-art
  icons (8); meme (13); logo wall (22); equation-only rasters (38, 40, 43). The raster test
  flagged 23 pages; the 7 it flagged and we skipped are 3, 8, 13, 22, 38, 40, 43.
- **Math on GitHub:** 43 formulas first failed on github.com. 38 were `}_` subscripts
  (`\mathbf{s}_t`) pairing into italics, fixed by a space after the underscore. 2 were
  `−$\lambda$` (no space before the `$`). 5 were formulas ending `)$)`. Lessons added to AGENTS.md.
- **Site reading links:** lecture 7's IQL and lecture 11's MBPO share an arXiv id on the site;
  lecture 12's two readings share one link. Recorded in sources.md as given.

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): bcd1bd9; check_math --github-published 540 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 2 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 02 Imitation Learning — video WxRDyObrm_M — `02_cs224r_imitation_2025.pdf`

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 32d388a; check_math --github-published 950 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 3 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 03 Policy Gradients — video KCAOXd4IO9o — `03_cs224r_policy_gradients_2025.pdf`

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 6fa138d; check_math --github-published 1535 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 4 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 04 Actor-Critic Methods — video oejFZShW9hU — `04_cs224r_actor_critic_2025.pdf`

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
- [x] BUILD_LOG.md — lecture 5's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 2006 formulas, 0 errors (slide file: one formula opening after a quotation mark fixed first); check_quotes 266 quotations, 0 unmatched (4 fixed on the lecture page); check_links 593 links, 0 broken
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 376e3ba; check_math --github-published 2006 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 5 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 05 Off-Policy Actor Critic — video cRGKc-nAWho — `05_cs224r_offpolicy_actor_critic_2025.pdf`

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

## Run 6 — lecture 6 (Q-Learning)

Run 6, 2026-10-28. Scope: lecture 6 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 6
- [x] raw/transcripts/original/06-q-learning.md — verbatim captions (80 paragraphs, ~9,900 words, ~61 min)
- [x] Slide number map: 29 pages, 1:1. The map script reports 17 pages unnumbered; raw spans show 12 of them print the number at bottom centre (0.49–0.50, 0.96). Pages 3, 15, 20, 26, 27 print none (labels inferred)
- [x] raw/slides/06-q-learning.md — deck transcribed (Sonnet), `--verify` clean (29 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 7, 15, 24 agree, 26 cell by cell (49 percentages, table) but its human-level line corrected; 19 and 20 small re-reads; 25 checked for the printed exponent
- [x] raw/transcripts/06-q-learning.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 80/80 markers identical, 5 numbers lost and 0 gained, all decided restorations ("2 pi", "2 s", "F1 3D" ×2 digits, "U5 prime"), 2 of 80 paragraphs outside 0.72–1.10 (18:48 at 1.21 and 20:19 at 1.12, both only inserted [Ed:] notes); 24 student questions and 4 answers marked; 53 paragraphs carry [Ed:] notes
- [x] wiki/06-q-learning.md — 12 images placed by hand at the argument (embed dry run: nothing missing); check_quotes 31 quotations, 0 unmatched (3 fixed)
- [x] Topic pages: q-learning, bellman-equations (new); policies, offline-and-online, policy-evaluation, replay-buffers, importance-sampling, ppo, actor-critic, value-functions, rl-algorithm-families (its stale "Lectures 1–4" coverage line fixed), mdp-and-notation, 05-off-policy-actor-critic and 01-class-intro (forward links) updated
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/06-q-learning/ — 12 slides rendered (1.3MB)
- [x] Embedded in the slide file (12, script), the lecture page (12, by hand) and topic pages by hand with "Lecture 6, slide N" captions (q-learning 15, 18, 20; bellman-equations 12); 12 hand-written captions
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (and Layer 2's stale "Lecture 1, 2 or 3 decks" now "Lecture 1–6")

### Publish
- [x] kb.json — coverage 6 of 19, topic pages 17, images 90, figure audit 6 of 6, caveats (Lecture 6's coverage, board work and printed exponent; its [Ed:] density)
- [x] BUILD_LOG.md — lecture 6's record (the Findings below move with this section); AGENTS.md gains one general line on reading raw spans when the map reports pages unnumbered
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read; check_math --github: 2435 formulas, 0 errors, 0 warnings (slide file: 4 GitHub-raw formulas and 5 unspaced `$$` blocks fixed first); check_quotes 315 quotations, 0 unmatched; check_links 732 links, 0 broken
- [x] Commit and push (chaimantec/cairn-kb-cs224r): ebcaf9e; check_math --github-published 2445 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 6 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 06 Q-Learning — video -7kv6jf0isQ — `06_cs224r_qlearning_2025.pdf`

### Findings

- **Lecturer:** the recording does not name the speaker, but slide 27 prints "Chelsea's advice" and
  the speaker introduces it as "my advice" (≈59:41), so the wiki writes "Finn" as for Lectures 1–5.
- **Licence:** none stated on the site or in the deck. Slides 8, 15, 17 and 19–23 print "Slide
  adapted from Sergey Levine". Slide 18 cites Mnih et al., Playing Atari with Deep Reinforcement
  Learning (2013) for DQN. Slide 26 credits "Mnih et al. 2015" (Atari chart) and "Kalashnikov et al.
  2018" (grasping photo and table, which cites "Levine et al. [27]"). Slides 19 and 20's charts are
  pasted from sources the slides do not name.
- **Deck:** 29 pages, printed number = page, 1:1. The map script reported 17 pages unnumbered; the
  raw spans show 12 of them print the number at bottom centre (0.49–0.50, 0.96), and pages 5, 8,
  12–14, 17, 19, 21–25 print it bottom right. Pages 3, 15, 20, 26 and 27 print none (labels
  inferred). `--verify` passes. Slides 3–5 revise Lecture 5's recap slides.
- **Recording coverage:** slides 3–29, 0:05 to 1:01:12, ending on slide 29's "Done with online RL
  methods!". Slide 2's reminders and slide 29's next-week topics are not discussed. Board work not
  visible: the algorithm with its targets (≈18:00–19:34). Slide 7's thought exercise is answered
  while pointing at the picture (≈7:08–12:32); the wiki reconstructs which rows are meant and says so.
- **Figure audit:** Opus parent, six 1400px renders (7, 15, 19, 20, 24, 26) against the Sonnet
  transcription, plus slide 25 for its exponent and a 500 dpi crop of slide 26's lower bars
  (Venture 32% confirmed). 7, 15, 24 and 25 agree; slide 26's 49 percentages and table agree cell by
  cell. Corrections: slide 26's human-level line had been placed at 100%, but it sits between
  H.E.R.O. (76%) and Asterix (69%), so 29 games are above it, several printed below 100%; slide 19's
  Breakout and Seaquest Q curves at epoch 25 (about 2 and 5, not 1.5 and 4); slide 20's Space
  Invaders DQN estimate at 100 million (about 8), its two true-value lines (coincide at about 4.9),
  and Zaxxon's Double DQN estimate leaving zero at about 75 million (not 50); slide 26's photograph
  shows arms at stations 2–4, not "four robot arms".
- **Printed slips (checked on the page):** slides 24–25 print the n-step discount as
  $\gamma^{t-t'}$, negative for $t' \gt t$, where Lecture 5's slides 12–13 print $\gamma^{t'-t}$;
  noted in the slide file and on the wiki and Q-learning pages, and the lecture reads the target
  aloud without discounts. Slides 12–13 print "Why does make sense?" (text layer). Slide 7 writes the
  estimate as $\hat{Q}^\pi$ in the text and $Q_\phi$ in the $\arg\max$. Recorded from the agent's
  list without a page check, so not claimed as slips: slide 14's $Q_\phi$ without a hat, the struck
  step 4 on slides 9–11 without $\phi$, slide 15's tuple order.
- **Transcript edit:** Sonnet, given the parent's glossary (Q-function/Q-value/Q-learning garbles,
  DQN, Bellman, Boltzmann, epsilon-greedy, n-step, phi prime, Seaquest, no-op, grasp, Ed). 80/80
  markers; 5 numbers lost, all restorations ("2 pi" and "2 s" to Q, the digits of "F1 3D" to
  epsilon-greedy, "U5 prime" to phi prime); 0 gained (the agent wrote its [Ed:] notes without
  digits). Ratios 2 of 80 outside the band, both only inserted [Ed:] notes (18:48 at 1.21, 20:19 at
  1.12). 24 student questions, 4 answers. The captions are much noisier than Lectures 1–5's: 53 of
  80 paragraphs carry [Ed:] notes, mostly garbled board dictation and mis-heard notation. Kept as
  captioned with notes: "08" and "7" at 11:00, "three" networks at 48:49, "is very impactful" at
  51:08, "a lot more biased" at 55:47. Parent fixes: three wrong slide numbers in the agent's
  restoration table (from the parent's own glossary: Boltzmann is slide 14, not 15).
- **Unclear in the recording, said so on the wiki:** the discounted values at 11:00, the search
  word for discrete spaces (≈15:40), the board algorithm's target (≈18:00–18:48), the rising-loss
  passage (≈44:57), "is/isn't very impactful" (≈51:08), two words of the double-Q answer (≈52:42),
  and "more biased" against the slide's "less biased" (≈55:47).
- **Images:** rendered 12 (3, 4, 5, 7, 8, 12, 15, 18, 19, 20, 24, 26), 1.3MB. Not rendered: 9–11,
  13, 14, 17, 21–23, 25 (equation slides with transcribed strikethroughs, highlights, loop arrows
  and annotations); 1, 2, 6, 16, 27, 28, 29 (title, reminders, plans, advice table, next time).
  Algorithm slides 12, 15 and 18 were rendered for their structure (colour-matched equations, the
  replay-buffer diagram, nested loops), following Lecture 5's rendering of its annotated algorithm
  slides.
- **Embedding:** lecture page by hand, all 12 at the argument (dry run: nothing missing); the script
  placed the 12 under their slide-file headings. Topic pages by hand with "Lecture 6, slide N"
  captions: q-learning 15, 18, 20; bellman-equations 12.
- **Math:** the offline check_math passed the slide file while the GitHub API check found four
  formulas GitHub would show as raw text (inline cases with `&`, a `)$)`) and five `$$` blocks
  without blank lines around them; all fixed before commit.

## Run 7 — lecture 7 (Offline RL)

Run 7, 2026-10-29. Scope: lecture 7 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 7
- [x] raw/transcripts/original/07-offline-rl.md — verbatim captions (88 paragraphs, ~10,400 words, ~68 min)
- [x] Slide number map: 33 pages, 1:1. The map script reports pages 5 and 28 unnumbered; raw spans show page 5 prints "5" bottom right (0.92, 0.94). Page 28 prints none (label inferred)
- [x] raw/slides/07-offline-rl.md — deck transcribed (Sonnet), `--verify` clean (33 headings)
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): slides 11, 22 (ten loss endpoints, five fitted curves) and 31 (table cell by cell, architecture arrows) agree; 14 and 16 ($s_7$ beside $s_3$, not on it; arrow-weight interpretation removed; advantage box not cut off) and 21 (dashed line, underline) small fixes. Printed slips checked on renders of 4, 5, 18, 19, 20, 21, 22
- [x] raw/transcripts/07-offline-rl.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 88/88 markers identical; 0 numbers lost; checker reports 18 gained, all in [Ed:] notes or decided restorations ("+ one" to +1, "e e^ squ" to e^2, ".5" and "0.5 five" to 0.5, "to8" to 8) — a per-paragraph digit diff with [Ed:] notes stripped leaves only those three; 1 of 88 paragraphs outside 0.72–1.10 (51:15 at 1.21, only two inserted [Ed:] notes); 17 student questions and 2 answers marked; parent added an inline [Ed:] note to the agent's "an accurate" to "an inaccurate" restoration (20:16)
- [x] wiki/07-offline-rl.md — all 14 images placed by hand at the argument (embed dry run: nothing missing; 3, 17 and 25 added after verify_kb's distribution table showed the page cited them without an image, which INDEX's images paragraph rules out); check_quotes 48 quotations, 93 citations, 0 unmatched
- [x] Topic pages: offline-rl, implicit-policy-constraints (new); offline-and-online (its "does not cover it yet" line replaced), imitation-learning, value-functions, policy-evaluation, q-learning, actor-critic, replay-buffers, rl-algorithm-families (coverage line now Lectures 1–7), mdp-and-notation, expressive-policy-distributions, bellman-equations and 06-q-learning (forward links) updated
- [x] INDEX.md table of contents and coverage

### Images
- [x] raw/images/07-offline-rl/ — 14 slides rendered (1.4MB)
- [x] Embedded in the slide file (14, script), the lecture page (14, by hand) and topic pages by hand with "Lecture 7, slide N" captions (offline-rl 11, 14, 26; implicit-policy-constraints 16, 22, 23); 14 hand-written captions on the lecture page
- [x] INDEX.md images paragraph; AGENTS.md table row; LICENSE.md attribution (and Layer 2's "Lecture 1–6 decks" now "Lecture 1–7"; the deck's text layer has no licence notice, only "Slide adapted from Sergey Levine" on 25–28)

### Publish
- [x] kb.json — coverage 7 of 19, topic pages 19, images 104, figure audit 7 of 7, caveats (Lecture 7's coverage and skipped CQL section, live board work, the printed Monte Carlo index, the worked example's average; its [Ed:] count)
- [x] BUILD_LOG.md — lecture 7's record (the Findings below move with this section); AGENTS.md gains one general line: a lecture page shows every rendered slide it cites
- [x] sources.md "In this KB" column
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 104 and 14 of 33 checked; distribution 14 on disk, 14 in wiki); check_math --github: 2949 formulas, 0 errors, 0 warnings (slide file: 9 formulas opened straight after a quotation mark fixed first); check_quotes 389 quotations, 971 citations, 0 unmatched; check_links 865 links, 0 broken
- [x] Commit and push (chaimantec/cairn-kb-cs224r): b4168dc; check_math --github-published 2962 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 7 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 07 Offline RL — video lRDaXnPIzks — `07_cs224r_offline_rl_2025.pdf`

### Findings

- **Lecturer:** the recording does not name the speaker, and the deck prints no name. The site's
  schedule lists no guest for this lecture (sources.md), so the wiki writes "Finn" as for Lectures 1–6.
- **Licence:** none stated on the site or in the deck (text layer searched). Slides 25–28 print
  "Slide adapted from Sergey Levine". Citations: Peters et al. (REPS) and Rawlik et al. ("psi-learning")
  on 16; Peng, Kumar, Zhang, Levine, Advantage-Weighted Regression ('19) on 18–19; Kostrikov, Nair,
  Levine, Implicit Q-Learning (ICLR '22) on 21–23, the only credit on slide 22's two charts; Prabhakar,
  Yuan, Yang, Sun, Muralidharan ('22) on 31's diagram and table; Hansen-Estruch et al., IDQL (2023) on 32.
- **Deck:** 33 pages, printed number = page, 1:1. The map script reported pages 5 and 28 unnumbered;
  the raw spans show page 5 prints "5" bottom right (0.92, 0.94), and page 28 prints none (label
  inferred). `--verify` passes.
- **Recording coverage:** slides 3–23 and 30–32, 0:05 to about 1:07:51. The conservative-methods part
  (slides 24–28, CQL) is not taught: the lecture goes from IQL (slide 23) to the summary (≈1:05:55), and
  slide 31's LinkedIn example is passed over as an application "for the algorithm we didn't cover"
  (≈1:06:42). Slides 2 and 33 are not discussed. Much of the lecture is written or drawn live (the
  recap's targets, the formal setting, the car's randomly initialized Q-function, the stitching
  example, the filtered and weighted objectives, AWR's advantages, the histogram and asymmetric loss);
  the wiki reconstructs it from the spoken words and says so.
- **Figure audit:** Opus parent, six 1400px renders (11, 14, 16, 21, 22, 31) against the Sonnet
  transcription. 11, 22 (all ten loss-curve endpoints, five fitted curves) and 31 (every table cell,
  the architecture's arrows) agree. Corrections: on 14 and 16, $s_7$ is drawn beside $s_3$, touching but
  not on it (the lecturer says so at ≈36:29); on 16, the agent's reading that the thicker arrow "stands
  for a larger advantage weight" was an interpretation the slide does not make, and removed; on 16, the
  agent's claim that the advantage box is "cut off" is wrong (the phrase "how much better a_t is" is
  complete); on 21, the second dashed line runs through the last bar, and the blue underline sits under
  $Q(\mathbf{s}', \mathbf{a}')$ only. Slide 6's brace line contradicted itself (items 2–3 or all three);
  the agent's coordinate check says 2–3, and the file now says so.
- **Printed slips (checked on the page):** slides 4, 18 and 19 print the Monte Carlo target as
  $\sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$, index unused (renders of 4, 18, 19), noted in the slide
  file, the wiki and policy-evaluation; slide 5's step 4 prints $\hat{Q}^\pi$ without $\phi$; slide 21
  writes the asymmetric loss $\ell_2^\tau$, slides 22–23 $\ell_2^\lambda$; slide 20 strikes through
  $\mathbf{a}' \sim \pi(\cdot \mid \mathbf{s})$ (conditioned on s, not s') and prints literal asterisks
  around "not"; slides 3–11 write $\mathbb{E}$, $\mathcal{D}$ and slides 16–28 plain $E$, $D$. Recorded from
  the agent's list without a page check, so not claimed as slips: slide 21's AWAC update and slide 23's
  fit-V step without parameter subscripts.
- **Transcript edit:** Sonnet, given the parent's glossary (PPO/SAC, Q-function/Q-value garbles,
  out-of-distribution, KL, Monte Carlo, GAE, AWR, AWAC, expectile, implicit Q-learning, filtered behavior
  cloning, the worked example's digits, the λ value). 88/88 markers; 0 numbers lost; the checker's 18
  gains are [Ed:] note digits and three decided restorations ("+ one" to +1, "e e^ squ" to e^2, ".5" to
  0.5). 1 of 88 paragraphs outside the band (51:15 at 1.21, two [Ed:] notes). 17 student questions, 2
  answers. 8 [Ed:] notes in 7 paragraphs: the captions are much cleaner than Lecture 6's. Agent
  restorations outside the glossary: "an accurate" to "an inaccurate" Q-function (20:16, parent added
  an inline [Ed:] note), "a good off method" to "a good offline method" (28:47).
- **Unclear in the recording, said so on the wiki:** the λ value "0 2 or 3" (≈1:03:37, perhaps 0.2 or
  0.3; the slide says only below 0.5); the worked example (≈51:15), whose returns as captioned (+10, +1,
  −5) average to 2, not the 3 the lecturer uses, with the third advantage captioned "8".
- **Images:** rendered 14 (3, 4, 5, 11, 14, 16, 17, 19, 21, 22, 23, 25, 26, 31), 1.4MB: the ten
  figure-bearing slides and four algorithm and objective slides for their structure (5, 19, 23, 26),
  following Lectures 5–6. Not rendered: 8, 10, 15, 18, 20, 27, 28 (equation slides with transcribed
  underlines, labels, strikethrough, loop arrows and circles); 1, 2, 6, 7, 9, 12, 13, 24, 29, 30, 32, 33
  (text).
- **Embedding:** the script placed 14 under their slide-file headings. Lecture page by hand. It first
  left 3, 17 and 25 (near-duplicates of 4, 16 and 11) to the slide file. The embed dry run reported them
  and verify_kb's distribution table showed 11 of 14, and INDEX tells the chat a cited rendered slide is
  shown in the passage, so they were added. Topic pages by hand with "Lecture 7, slide N" captions:
  offline-rl 11, 14, 26; implicit-policy-constraints 16, 22, 23.
- **Math:** the chat check passed the slide file; the GitHub check found 9 inline formulas opened
  straight after a quotation mark (the transcriber quoted slide labels with math inside, despite the
  rule in AGENTS.md); rewritten with the math outside the quotes.

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 039aea2; check_math --github-published 3290 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 8 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 08 Reward Learning — video PDIxDhA9Z6Y — `08_cs224r_reward_learning_2025.pdf`

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 7e4208d; check_math --github-published 3630 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 9 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 09 RL for LLMs — video XKLGuwvSKvI — `09_cs224r-2025-rlhf.pdf`

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 60d89fa; check_math --github-published 3932 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 10 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 10 RL for LLM Reasoning — video O2VpNnwB4lM — `10_cs224r-rl_for_reasoning_lecture.pdf`

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

## Run 11 — lecture 11 (Model-Based RL)

Run 11, 2026-10-29. Scope: lecture 11 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 11
- [x] raw/transcripts/original/11-model-based-rl.md — verbatim captions (95 paragraphs, ~12,100 words, ~73 min)
- [x] Slide number map: 38 pages, printed 1–38, 1:1. The map script reports 3, 4, 7, 8, 35 unnumbered; raw spans show 4, 7, 8, 35 print theirs bottom right, and only page 3 prints none (label follows the 1:1 sequence, unambiguous)
- [x] raw/slides/11-model-based-rl.md — deck transcribed (Sonnet), `--verify` clean (38 headings); recording-coverage paragraph, video notes from the transcript (5, 6, 8, 34, 36) and audit corrections added by the parent; check_math 114 formulas, 0 errors
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 4, 27, 36 (every value of both series) agree; 34's two charts agree series by series, but PETS's Baoding curve runs behind the legend rather than stopping; 12's backward arrows are on every link but the first; 35's CEM curve rises from about 0.4 to 0.55 (had been flat) and random shooting swings 0.3–0.9 (had been 0.1–0.9). Printed oddities checked on renders: slide 32's author line prints "Kurt Konoglie" (kept as printed); 14's literal asterisks; 34's "Nagabandi et. al" in the legends; 28's and 30's non-bold symbols
- [x] raw/transcripts/11-model-based-rl.md — copy-edit (Sonnet), timestamp / number / ratio checks adjudicated: 95/95 markers identical; 0 lost, gained 1 ×2 (z t plus 1 at 17:02, s t plus 1 at 19:24, both glossary restorations); 0 of 95 paragraphs outside 0.72–1.10 (overall 0.886); 20 student questions and 3 student answers marked; 6 [Ed:] notes. Parent fixes: the 1:04:23 [Ed:] note moved to the second "h" (the first was spoken as h, the second captioned "eight"); a student-question mark added before the finite-differences restatement at 38:50; the header's vague "28:00" deferral removed
- [x] wiki/11-model-based-rl.md — all 17 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing, after fixing a slide-22 `.png` link to `.jpg`); check_quotes 80 quotations, 146 citations, 0 unmatched; check_math 106 formulas, 0 errors
- [x] Topic pages: model-based-rl and planning-with-models (new); offline-and-online (the whole map so far, planning is off-policy), rl-algorithm-families (model-based row, coverage now Lectures 1–11, the recap), reward-learning (a reward model for planning), value-functions (terminal value function), imitation-learning (distilling a planner), mdp-and-notation (notation for model-based RL), policies (a planner as a "policy"), course-logistics (Homework 3 and milestone dates) updated; 01-class-intro, 04-actor-critic and 10-rl-for-reasoning forward links. check_quotes 73 quotations on the new and updated pages, 0 unmatched; check_math wiki 2353 formulas, 0 errors; check_links 1407 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–11 and 190 images, Lecture 11's recording coverage, lecture entry, two new topic entries and eight updated, SEE_ALSO pointer)

### Images
- [x] raw/images/11-model-based-rl/ — 17 slides rendered (1.9MB); embedded in the slide file (17, script), the lecture page (17, by hand, hand-written captions) and topic pages by hand with "Lecture 11, slide N" captions (offline-and-online 4; model-based-rl 8, 27; planning-with-models 13, 22)
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (`--format auto` mixes extensions, so take hand-placed paths from `ls` and run the embed dry run; a glossary restoration aimed at one of several identical words quotes its neighbours); LICENSE.md attribution (Layer 2 now "Lecture 1–11 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 11 of 19, topic pages 29, images 190, figure audit 11 of 11, caveats (Lecture 11's coverage, board work and covered chart; its [Ed:] count)
- [x] BUILD_LOG.md — lecture 11's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: Princeton ROB 345 added (trajectory optimization and model predictive control with known dynamics; repo public, four raw links return 200)
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 190 and 17 of 38 checked; distribution 17 on disk, 17 in wiki); its GitHub math check: 4247 formulas, 0 errors; check_links 1407 links, 0 broken
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 298d25a; check_math --github-published 4247 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 11 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 11 Model-Based RL — video PvqyGnOirgA — `11_cs224r_mbrl_2025.pdf`

### Findings

- **Lecturer:** Chelsea Finn (the title slide names only "CS 224R"). Finn says model-based RL has a
  special place in Finn's heart, having worked on it (≈4:36). Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©,
  Creative, rights). Third-party material: Veo 2 and Sora video stills (6), a maze image and a Sora video
  frame (8), a cliff photograph (18, 20), clip-art (21, 22), and the case-study paper's renderings, charts
  and photographs (33–36): Nagabandi, Konoglie (as printed), Levine and Kumar, "Deep Dynamics Models for
  Learning Dexterous Manipulation", Google Brain, September 2019 as printed on slide 32. No per-slide credit.
- **Deck:** 38 pages, printed 1–38, 1:1, numbers bottom right. The map script reports 3, 4, 7, 8, 35
  unnumbered; raw spans show 4, 7, 8, 35 print theirs, and only page 3 (a section divider) prints none.
  `--verify` passes.
- **Recording coverage:** about 73 minutes, slides 4–25 and 32–36. At ≈1:06:40 Finn says "I think I might
  cover this second approach on Friday" and goes to the case study, so slides 26–30 (model-based policy
  optimization's roll-outs and algorithm, when to use model-based RL, other kinds of models) are not
  discussed; Lecture 12's deck (slides 3–9 by its text layer) opens with its own version of them. Slides 2,
  37 and 38 are not discussed beyond the closing "next time" (≈1:12:52). Board work: a latent-state model
  with an encoder (≈15:23–17:49), the gradient-based planning algorithm with its chain rule and update
  (≈23:17–31:07), sketches of gradient-based and sampling-based optimization over actions (≈31:54–35:44),
  and open versus closed loop (≈51:13–52:52). At ≈32:40 Finn corrects a sign ("There should be a plus sign
  because we're maximizing not minimizing"), apparently in the update written at ≈26:28; the wiki writes
  the corrected update and says so.
- **Figure audit:** Opus parent, six 1400px renders (4, 12, 27, 34, 35, 36) against the Sonnet
  transcription, plus a crop of slide 32's author line. 4, 27 and 36 (every value of both series) agree;
  34's two charts agree series by series. Corrections: slide 12's backward arrows are on every link but the
  first; slide 35's CEM curve rises from about 0.4 to 0.55 (had been flat at 0.4–0.5), and random shooting
  swings 0.3–0.9 (had been 0.1–0.9); slide 34's PETS curve runs behind the legend rather than stopping.
- **Printed oddities (checked on renders):** slide 32 prints "Kurt Konoglie" (kept as printed, not
  corrected); slide 14 prints literal asterisks around "especially"; slide 34's legends print "Nagabandi
  et. al", and a text box covers the lower part of both its charts and its slide number; slides 28 and 30
  print some symbols non-bold. Finn calls the random-shooting baseline yellow (≈1:10:35) where slide 34 draws it
  orange; the wiki says so.
- **Transcript edit:** Sonnet, given the parent's glossary (REINFORCE, PPO, importance sampling, SAC, DQN,
  RL, model-based, dexterous, Veo 2, learned simulator, z t, p of s t plus 1 given s t, backprop, r of s, a,
  stochasticity and Gaussian, cross-entropy method, overparameterized, parallelized, open- and closed-loop,
  warm-start, h for "eight" at 1:04:23, PDDM, MBPO, PETS, soft actor-critic, Baoding balls). 95/95 markers;
  0 lost, gained 1 ×2 (glossary restorations); 0 of 95 paragraphs outside the band (overall 0.886). 20
  student questions and 3 student answers marked. 6 [Ed:] notes in 6 paragraphs: "ps1 given st" (19:24),
  "next action" for the next state (20:10), "RSI" (27:13), "110 steps" as captioned (59:01), "eight" for h
  (1:04:23), "soft actor" (1:09:47). Parent fixes: the 1:04:23 note moved to the second "h" (the agent had
  put it on the first, which was spoken as h); a student-question mark added before the finite-differences
  restatement (38:50); a vague deferral removed from the header. Every restored proper noun is printed in
  the deck except "Gates building", a spoken aside.
- **Unclear in the recording, said so on the wiki:** the replanning horizon "something like 110 steps"
  (≈59:01); which step the sign correction at ≈32:40 refers to.
- **Images:** rendered 17 (4, 6, 7, 8, 12, 13, 17, 18, 19, 20, 21, 22, 27, 33, 34, 35, 36), 1.9MB: every
  figure-bearing slide except slide 5 (an agenda whose video frame shows slide 36's scene), the build steps
  14, 15 and 26, and slide 32's paper title block, transcribed in full. Not rendered also: 10, 16, 23, 25,
  28, 29, 30 (text, equation and algorithm slides) and 1, 2, 3, 9, 11, 24, 31, 37, 38 (title, divider,
  agenda, reminders).
- **Embedding:** the script placed 17 under their slide-file headings; the lecture page by hand (embed dry
  run: nothing missing, once a `slide-22.png` link was corrected to the file's `.jpg`); topic pages by hand
  with "Lecture 11, slide N" captions.
- **SEE_ALSO:** Princeton ROB 345 added for trajectory optimization and receding horizon control (MPC) with
  known dynamics; it does not learn models.

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
- [x] Commit and push (chaimantec/cairn-kb-cs224r): b4744f1; check_math --github-published 4570 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 12 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 12 Multi-Task RL — video qNdsI_4AQJw — `12_cs224r_mtrl_gcrl_2025.pdf`, with its note to revise
  Lecture 11's "not in the recording" pointers when it landed (done: wiki/11-model-based-rl.md, wiki/model-based-rl.md, INDEX.md, kb.json)

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

## Run 13 — lecture 13 (Meta RL)

Run 13, 2026-10-31. Scope: lecture 13 only. Images on, PDFs link-only, commit as chaimantec
<nantapon@chaimantec.com>.

### Lecture 13
- [x] raw/transcripts/original/13-meta-rl.md — verbatim captions (90 paragraphs, ~12,000 words, ~69 min)
- [x] Slide number map: 39 pages, 1:1. The map script found no number (it scans the corners); the raw spans show bottom-centre numbers on 15 pages, each equal to its page index, and none on the other 24, which sit between consecutive numbered pages with no gap
- [x] raw/slides/13-meta-rl.md — deck transcribed (Sonnet), headings 1..39 (`--verify` OK on the 1..N fallback, which is right for this deck); check_math 172 formulas, 0 errors; recording coverage, slide 17's new-task note, slide 20's video note and audit corrections added by the parent
- [x] Figure spot-check of 6 pages from 1400px renders (Opus parent; transcriber was Sonnet): 12, 19, 21, 22, 26, 35, with crops of 21's table and the charts of 22 and 26. 12, 21 (every cell), 22 and 35 agree; 26's series, colours and values agree but its dashed lines had been called the other methods' final performance (the slide does not say; removed); 19's attention network has seven rows of nodes (corrected). Slide 20's images found to be frames of two videos (≈33:08–34:40), which the transcription had not said. Text layer confirms slide 22's printed "ICML 2019", slide 9's "using on", plain n on slide 13
- [x] raw/transcripts/13-meta-rl.md — copy-edit (Sonnet) with the parent's glossary; check_transcript.py: 90/90 markers identical; 0 lost, gained 1 ×2 and 2 (s 1, s 2 from the caption S1, S2) and 19, 26, 8 (slide numbers in [Ed:] notes); 0 of 90 paragraphs outside 0.72–1.10, overall 0.914; 15 student-question and 9 student-answer markers; 4 [Ed:] notes. Restored names all in the deck except GPT-3, Aviral, deep set (spelled so in the captions) and positional encoding
- [x] wiki/13-meta-rl.md — lecture page; all 19 images placed by hand at the argument with hand-written captions (embed dry run: nothing missing); check_quotes 80 quotations, 121 citations, 0 unmatched (after converting the slide file's [colour] tags to prose); check_math 98 formulas, 0 errors
- [x] Topic pages: meta-rl (new, with slides 11, 17, 25, 33 placed by hand as "Lecture 13, slide N"); multi-task-rl (meta-RL against the aggregated MDP), policies (policies that learn in context), mdp-and-notation (the meta-RL POMDP; notation for meta-RL), llm-reasoning (test-time compute as meta-RL), replay-buffers (trajectory buffers for meta-RL), offline-and-online (on- and off-policy meta-training), rl-algorithm-families (coverage now Lectures 1–13), course-logistics (Lecture 13's reminders, evening office hours, part of HW4), 12-multi-task-rl, 01-class-intro and 10-rl-for-reasoning forward links. check_quotes 309 quotations, 621 citations, 0 unmatched; check_math wiki 2698 formulas, 0 errors; check_links 1664 links, 0 broken
- [x] INDEX.md table of contents and coverage (coverage note now Lectures 1–13 and 218 images, Lecture 13's recording coverage, lecture entry, one new topic entry and eight updated)

### Images
- [x] raw/images/13-meta-rl/ — 19 slides rendered (2.3MB) by `extract_slide_images.py` (its 1..N fallback map is right for this deck); embedded in the slide file (19, script), the lecture page (19, by hand, hand-written captions) and meta-rl (11, 17, 25, 33, by hand)
- [x] INDEX.md images paragraph; AGENTS.md table row and two general lines (a video's frame may carry no player chrome; colours in prose, never inline tags); LICENSE.md attribution (Layer 2 now "Lecture 1–13 decks"; the deck's text layer has no licence notice)

### Publish
- [x] kb.json — coverage 13 of 19, topic pages 32, images 218, figure audit 13 of 13, decks transcribed 13, caveats (Lecture 13's coverage, board work, video frames, slide 22's printed venue, [Ed:] notes)
- [x] BUILD_LOG.md — lecture 13's record (the Findings below move with this section)
- [x] sources.md "In this KB" column
- [x] SEE_ALSO.md: no change; no sibling KB covers meta-RL
- [x] verify_kb.py clean, review section read (no ranking claims; numeric claims 218 and 19 of 39 checked; distribution 19 on disk, 19 in wiki); its GitHub math check: 4891 formulas, 0 errors; slide_number_map --verify OK after the image insertions
- [x] Commit and push (chaimantec/cairn-kb-cs224r): 1e042e5; check_math --github-published 4896 formulas, 0 errors; kbUrl already set from run 1, so no PATCH
- [x] Move this run's section to BUILD_LOG.md; tick lecture 13 in the backlog; TODO.md keeps only open work
- [x] Backlog line, moved from TODO.md: 13 Meta RL — video wSiyEpvoGkA — `13_cs224r_metarl_2025.pdf`

### Findings

- **Lecturer:** Chelsea Finn (the title slide names only "CS 224R"). Pronouns not stated; the wiki writes "Finn".
- **Licence:** none stated on the site or in the deck (text layer searched for licen, copyright, ©, Creative,
  rights). Third-party material: an espresso photograph (6), Braque and Cezanne paintings (8–9), maze diagrams
  "adapted from Duan et al. '17" (10–11), a legged robot, a manipulation-task grid and clip-art (12), the RL²,
  SNAIL and PEARL architecture figures (19), SNAIL's maze video frames (20, Mishra et al., ICLR 2018), DeepSeek
  R1 excerpts and the MRT chart (22, Qu et al., venue printed as ICML 2019), robot and walker pictures (23),
  PEARL's charts and stills (26, Rakelly et al., ICML 2019), hallway, robot and document pictures (30), kitchen
  photographs (32), emoji (33), PEARL's trajectory panels (35), a trajectory panel beside MAME and MetaCURE
  citations (36).
- **Deck:** 39 pages, printed 1–39, 1:1. The map script found no number (it scans the corners); the raw spans
  show bottom-centre numbers on 15 pages (1, 2, 5, 8, 13–18, 20–22, 26, 27, 38, 39), each equal to its page
  index, and none on the other 24, which sit between consecutive numbered pages; the renders show none either.
  `--verify` passes on the 1..N fallback, which is right here, and the renderer's map is right too.
- **Recording coverage:** about 69 minutes, slides 3–37, ending at about 1:09:10 on slide 37's summary. Not
  discussed: slide 2's reminders, 38–39. Board work: the black-box algorithm (≈24:34–26:55), its meta-test
  (≈30:03–30:49), a discount factor (≈36:10). Slide 20's two maze videos play at ≈33:08–34:40 and are replayed
  ("watching that again. Whoops").
- **Figure audit:** Opus parent, six 1400px renders (12, 19, 21, 22, 26, 35), with crops of 21's table and the
  charts of 22 and 26. Agree: 12, 21 (every cell), 22 (series, colours, arrows, values), 35, and 26's series,
  colours and values. Corrections: 26's dashed lines had been called "the final performance of the other
  methods", which the slide does not say (removed, and each line's colour and level given); 19's attention
  network has seven node rows (red, two orange, green, two orange, green from the bottom), not the eight
  given. The transcriber reported no video frames; the transcript shows slide 20's images are video frames
  (added). The text layer confirms 22's "ICML 2019", 9's "using on", plain n on 13.
- **Printed oddities:** slide 9's "using on a few examples"; slide 22's venue "ICML 2019" for the MRT paper;
  slides 17–18 print $a$, $s$ non-bold; slide 29 says "w.r.t. Reward" where 31 and 34 say "w.r.t. Task
  Reward"; slide 26's first and third panels have clipped tick labels; the deck writes the training set as
  $\mathscr{D}_ {\text{tr}}$, $D_{\text{train}}$ and $\mathcal{D}_ {\text{train}}$ on different slides, and the
  episode count as N where Finn says n.
- **Slide file:** the transcriber wrote colours as inline tags ([red]…[/red]); seven wiki quotations of those
  phrases failed check_quotes until the tags were rewritten as prose (lesson in AGENTS.md).
- **Transcript edit:** Sonnet, given the parent's glossary (multi-task, meta-RL, meta-learning at 6:59, 7:45,
  9:16, meta-training, zero-shot transfer, test data points at 8:31, T i and T j, n, positional encoding, s 1,
  s 2, z i and z, parallelize, DeepSeek R1 and R1 at 39:58, test-time compute and training, goal-conditioned
  and k-shot at 43:03, off-/on-policy, end-to-end, hyperparameters, training process at 50:46, RL squared kept,
  decouple at 1:03:59). 4 [Ed:] notes: "test video points" (8:31), a word after PPO (49:13), RL² (50:46),
  "get couple of" (1:03:59).
- **Unclear in the recording, said so on the wiki:** the word after "PPO" at ≈49:13; "decouple" at ≈1:03:59;
  the exact form of the board's objective (reconstructed from the spoken description and labelled so).
- **Images:** rendered 19 (3, 6, 8, 9, 11, 12, 16, 17, 19, 20, 22, 23, 25, 26, 30, 32, 33, 35, 36), 2.3MB.
  Not rendered: build steps 10 (its mazes are on 11) and 24 (slide 16's chain with a question, answered by 25);
  21's table, transcribed cell by cell; 14 (input-output diagram, transcribed) and 27, 31, 34 (repeats of 16's
  or 17's chain beside text); 4, 7, 13, 18, 29, 37 (equation and text); 1, 2, 5, 15, 28, 38, 39 (title,
  reminders, agenda, divider).
