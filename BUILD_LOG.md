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
