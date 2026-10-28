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
