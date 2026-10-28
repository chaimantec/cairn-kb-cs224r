# How this knowledge base is organized

This repo is the knowledge base for **Stanford CS224R — Deep Reinforcement Learning (Spring 2025)**. It is read
by Cairn's in-extension AI chat, which starts at `INDEX.md`, fetches files over
raw.githubusercontent.com and follows relative markdown links.

The chat never reads this file. It is for whoever builds the KB, and Claude Code
loads it into every step of every build agent, so keep it to the rules every
lecture's build follows. The per-lecture record (licence notices, audit findings,
printed slips, which slides were rendered and why) goes in
[`BUILD_LOG.md`](BUILD_LOG.md); anything the chat needs goes in `INDEX.md`;
attribution goes in [`LICENSE.md`](LICENSE.md).

## Layout

| Path                | Contents                                                      |
| ------------------- | ------------------------------------------------------------- |
| `INDEX.md`          | Entry point. Course summary + annotated table of contents.    |
| `wiki/`             | Durable pages: one per lecture, plus cross-lecture topics.    |
| `raw/slides/`       | Full text of every slide, headed `## Slide N` with the **printed** number. |
| `raw/transcripts/`  | Copy-edited transcripts with `[MM:SS]` paragraph marks.       |
| `raw/transcripts/original/` | Verbatim auto-captions. **Gitignored**, local reference only. |
| `raw/pdfs/`         | Downloaded decks and handouts. **Gitignored**; cite the canonical URL. |
| `raw/images/`       | Rendered figure slides, `NN-<slug>/slide-N.jpg`. Linked from the pages using them. |
| `sources.md`        | Every course document by lecture, with canonical URLs (maintained by hand). |
| `LICENSE.md`        | Three-layer licence and image attribution.                    |
| `kb.json`           | Machine-readable coverage and provenance.                     |
| `TODO.md`           | Open work only. Finished runs move to `BUILD_LOG.md`.       |
| `BUILD_LOG.md`      | Finished runs, one section each. Not needed to answer questions. |

## Conventions

- **INDEX.md is the front door.** The chat reads it first on every conversation.
  Every wiki page must appear there with a one-line description of what it holds.
  An unindexed page is effectively invisible.
- **Relative links only**, including images. Absolute GitHub URLs break when the
  repo is renamed or forked. Slide decks link to their canonical course-site URL,
  never to `raw/pdfs/`, which is gitignored.
- **Cite everything.** Written material as "slide N" (the printed number),
  spoken material as (≈M:SS). Claims trace back to one or the other.
- **Never rank lectures against each other** in prose; state the measurement for
  the lecture in hand.
- **The lecturer's pronouns are not stated in the material**: write "Finn" or
  rephrase rather than guess.
- **Never invent course content.** If the transcript is unclear at some point,
  say so on the page. Do not fill the gap from outside knowledge — the chat
  presents these pages as authoritative material from this course.
- **Prose over fragments.** The chat quotes these pages to learners; bullet
  fragments quote badly.
- **Math in LaTeX**: `$...$` inline, `$$...$$` displayed. Renders both in
  Cairn's chat and on github.com. Never inside a code fence — that shows the
  source instead of the formula. Define each symbol on first use, and follow the
  course's own notation rather than a tidier one, since the learner is comparing
  the page against the lecturer's board. The slides use bold lower-case
  $\mathbf{s}_ t$, $\mathbf{a}_ t$, $\mathbf{o}_ t$, $\pi_\theta$, $p_\theta(\tau)$; use `\mid`
  for conditioning and `\thinspace` for spacing. Put a space after every underscore that
  follows `}` or `)` (`\mathbf{s}_ t`, `\mathbb{E}_ {…}`), or GitHub's markdown pairs the
  underscores into italics; never end an inline formula with `)` directly before a `)`
  (`…)$)` renders as raw text on github.com). A literal dollar sign in
  prose opens a math span: write "1 million dollars". Run `check_math.mjs --github`
  (the `cairn-kb` skill) before pushing.
- **Transcripts stay verbatim.** Captions spell notation out as speech ("theta
  sub j"); reconstructing it as $\theta_j$ belongs in `wiki/`, not in
  `raw/transcripts/`.

## Course materials

- The site is one page (`spring_2025/index.html`). Decks are at `slides/NN_*.pdf` and
  handouts at `material/`, all on the course host, so the crawler downloads them. Exclude
  `projects/` (student final-project archive) and `spring_2023` when crawling.
- `sources.md` is a per-lecture table maintained by hand from the site's schedule. Update its
  "In this KB" column when a lecture lands. The catalog's playlist order puts the Q-learning TA
  session last (19); the site held it in week 3.
- Some optional-reading links on the site point at the wrong arXiv id. `sources.md` records
  them as the site gives them and says so; do not "fix" them from outside knowledge.
- Confirm a guest-lecture deck against its video by its title slide, not its filename
  (lecture 17's catalog title differs from the site's).
- No licence is stated on the site or in the decks. See `LICENSE.md`.

## Transcripts

- `raw/transcripts/NN-<slug>.md` is the copy-edited transcript the chat reads, edited by a
  Sonnet subagent after the lecture's slide file exists, and checked in the parent with
  `kb-checkers/check_transcript.py`: an identical `[MM:SS]` sequence, the number inventory, and
  per-paragraph word ratios in 0.72–1.10. Adjudicate every gained or lost number.
- Give the copy-edit agent the parent's glossary of decided restorations, and tell it: no
  LaTeX and no `$` in the body; never move words across a marker; "So the question is" opens a
  repeated student question.
- Only the lecturer is audible. Student turns are marked `*[Student question, inaudible]*` /
  `*[Student answer, inaudible]*` before the lecturer's restatement.
- The verbatim captions stay on disk in `raw/transcripts/original/` and are **not
  committed** (no licence; a full caption copy is the weakest thing to publish).
- Recordings can skip slides (Lecture 1's skips its logistics slides). Say so in the slide
  file's header, the transcript header and the wiki page, so the slide file is known to be the
  only source for them.

## Slides

- Run `slide_number_map.py` before transcribing and hand the agent the map. A number the text
  layer finds can still be invisible on the page (covered by an image); the label stands, but
  say so in the slide file.
- Transcription is delegated to a Sonnet subagent, reading contiguous ranges of at most 15
  pages and appending as it goes; `--verify` must pass. Then the Opus parent spot-checks six
  figure slides from the 1400px renders (~1.5k tokens each, cross-model with the transcriber),
  checking structure, arrows, labels and equations. Check every "printed slip" the agent lists
  against the page before recording it.
- When the lecturer corrects a printed slide aloud, note it under that slide with the timestamp,
  and write the corrected form in `wiki/`, saying the slide prints it otherwise.
- A slide holding a video holds one frame. What the video showed comes from the transcript,
  written into the slide entry by the parent with its timestamp.

## Images

- One image per rendered slide at `raw/images/NN-<slug>/slide-N.jpg` (or `.png`), 1400px wide,
  rendered by `extract_slide_images.py` and placed by `embed_slide_images.py`, which puts each
  image under `## Slide N` in the slide file and after the first block that cites slide N in a
  wiki page. Run it once per wiki page, `--dry-run` first. Never cite a rendered slide inside a
  table row or in a sentence that ends in `:` before a displayed equation; the image would land
  there. A range such as "slides 26–28" cites every slide in it, and one 12 or more wide cites its first slide, so keep slide ranges out of coverage notes. The same goes for a sentence ending in `:` before a list.
- Render every figure-bearing slide except title, agenda and text-only slides, photographs of
  course staff, memes and clip-art whose captions are transcribed, logo walls whose names are
  transcribed, and slides whose only images are equations reproduced exactly in the slide file.
- Replace the script's default captions (`Slide N — <title>?.`) with hand-written ones saying
  what to look at; no math or emphasis inside a caption.
- The embed script anchors on "slide N" and cannot tell lectures apart, so on a topic page that
  cites more than one lecture's slides, place the new lecture's images by hand with the lecture
  page's captions ("Lecture N, slide M — …"); run the script only on single-lecture pages.
- `grep -o 'raw/images/[^)]*' wiki/NN-*.md` lists a lecture's wiki images. The slide file's
  `images:` header lists every rendered slide and why the rest were not.
- When a lecture adds images, update `INDEX.md`'s images paragraph, the table below,
  `kb.json` `materials.images`, and `LICENSE.md`.

| Lecture | Images | Notes |
| --- | --- | --- |
| 1 Class Intro | 16 of 45 pages | robot video stills, a Go broadcast, two simulation screenshots, generated-image strips, the chip-placement diagram, the RL graphical models, the policy network |
| 2 Imitation Learning | 15 of 27 pages | the multimodal steering chart, distribution and generative-policy diagrams, success-rate bar charts, robot and driving model stills, the compounding-error and DAgger diagrams, demonstration interfaces, SFV video frames |
| 3 Policy Gradients | 11 of 28 pages | the online-RL loop and its iterations, the trajectory plots, the REINFORCE and reward-weighted-imitation slides, the humanoid and jacket-folding examples, the summary and review slides, the off-policy algorithm |

## Rebuilding

Built and updated by the `cairn-kb` skill. To add newly released lectures,
append entries to `TODO.md` and re-run the skill — it only does unchecked work.
When a run is committed and pushed, move its `TODO.md` section verbatim to
`BUILD_LOG.md`, so `TODO.md` keeps only open work. Add here only a lesson
that applies to every lecture, stated in general form.
