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
  (`…)$)` renders as raw text on github.com), and never open inline math straight after a quotation
  mark (`"$\tau^4$: …"`): put the symbol outside the quotes. A literal dollar sign in
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
- A guest deck may revise one already transcribed in a sibling KB (Lecture 9's is CS224N's
  post-training deck). List the sibling in `SEE_ALSO.md`, but never borrow its slide labels or
  descriptions: the revision renumbers and changes slides.
- No licence is stated on the site or in the decks. See `LICENSE.md`.

## Transcripts

- `raw/transcripts/NN-<slug>.md` is the copy-edited transcript the chat reads, edited by a
  Sonnet subagent after the lecture's slide file exists, and checked in the parent with
  `kb-checkers/check_transcript.py`: an identical `[MM:SS]` sequence, the number inventory, and
  per-paragraph word ratios in 0.72–1.10. Adjudicate every gained or lost number.
- Give the copy-edit agent the parent's glossary of decided restorations, and tell it: no
  LaTeX and no `$` in the body; never move words across a marker; "So the question is" opens a
  repeated student question. A restoration that depends on context (a common word that is
  also a benchmark or a name) applies only at the timestamps the glossary gives: check the
  agent's other uses of it. When the target is one of several identical words in a paragraph, quote the
  words around it, and check where the agent put it. Put the KB's standing spellings (multi-task, meta-RL,
  epsilon-greedy) in every glossary: an agent leaves an unlisted variant as captioned, and a wiki quotation
  of that passage then fails.
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
- The map script reads only one corner, and these decks also print the number at bottom centre or
  bottom right. When it reports pages unnumbered, read the raw spans of those pages before
  calling a label inferred. It can also take a digit in a page's content for the page number and
  report gaps that do not exist. `extract_slide_images.py` resolves slides through the same map, so
  when the spans show the map is wrong, render by page index (its `render()` takes a 1-based page)
  and check the printed number on one render.
- Transcription is delegated to a Sonnet subagent, reading contiguous ranges of at most 15
  pages and appending as it goes; `--verify` must pass. Then the Opus parent spot-checks six
  figure slides from the 1400px renders (~1.5k tokens each, cross-model with the transcriber),
  checking structure, arrows, labels and equations. Check every "printed slip" the agent lists
  against the page before recording it.
- When the lecturer corrects a printed slide aloud, note it under that slide with the timestamp,
  and write the corrected form in `wiki/`, saying the slide prints it otherwise.
- Describe a photographed person by the name the slide prints, never by apparent gender; tell the
  transcriber so.
- A slide holding a video holds one frame. What the video showed comes from the transcript,
  written into the slide entry by the parent with its timestamp. A video's frame may carry no player chrome, so a
  transcriber's "no video frames" is checked against the transcript ("watching that again", "this is what it
  looks like") at each photograph or screenshot slide, and a transcriber's "may be a video frame" is checked
  the same way, then kept or removed. A slide that launches a live demo (a web game) holds none of it:
  describe the demo under that slide from the transcript, as for a video.
- PyMuPDF is in the user site-packages, which `python3 -I` disables, and this Python has no `-P`: run PyMuPDF
  scripts without `-I` from a directory holding no `.py` files, and tell the transcriber so.
- When a chart prints no axis values, have the transcriber give its printed labels (gaps, percentages) and the
  bars' relative heights, not fractions of the panel, which mean nothing.
- Tell the transcriber to give colours in prose ("in blue"), never as inline tags (`[blue]…[/blue]`):
  tags inside a printed phrase break every verbatim quotation of it.
- An animated chart exports as its first frame: a few points on empty axes. Say so in the slide
  entry and the image caption, and take the result from what the lecturer says.

## Images

- One image per rendered slide at `raw/images/NN-<slug>/slide-N.jpg` (or `.png`), 1400px wide,
  rendered by `extract_slide_images.py` and placed by `embed_slide_images.py`, which puts each
  image under `## Slide N` in the slide file and after the first block that cites slide N in a
  wiki page. Run it once per wiki page, `--dry-run` first. Never cite a rendered slide inside a
  table row or in a sentence that runs on into a displayed equation, with or without a `:`; the
  image would land there. A range such as "slides 26–28" cites every slide in it, and one 12 or more wide cites its first slide, so keep slide ranges out of coverage notes. The same goes for a sentence ending in `:` before a list.
- Render every figure-bearing slide except title, agenda and text-only slides, photographs of
  course staff, memes and clip-art whose captions are transcribed, logo walls whose names are
  transcribed, and slides whose only images are equations reproduced exactly in the slide file.
- Replace the script's default captions (`Slide N — <title>?.`) with hand-written ones saying
  what to look at; no math or emphasis inside a caption.
- The embed script anchors on "slide N" and cannot tell lectures apart, so on a topic page that
  cites more than one lecture's slides, place the new lecture's images by hand with the lecture
  page's captions ("Lecture N, slide M — …"); run the script only on single-lecture pages.
- `INDEX.md` tells the chat that a rendered slide is shown wherever a wiki page cites it, so a
  lecture page shows every rendered slide it cites. When the embed dry run reports images to add,
  place them (or drop the citation if the passage did not come from that slide), rather than
  leaving a near-duplicate out.
- `--format auto` keeps whichever of `.jpg` and `.png` is smaller, so one lecture mixes both. When placing
  images by hand, take each path from `ls`, and run the embed dry run: a mistyped extension shows up as an
  image still to add.
- `grep -o 'raw/images/[^)]*' wiki/NN-*.md` lists a lecture's wiki images. The slide file's
  `images:` header lists every rendered slide and why the rest were not.
- When a lecture adds images, update `INDEX.md`'s images paragraph, the table below,
  `kb.json` `materials.images`, and `LICENSE.md`.

| Lecture | Images | Notes |
| --- | --- | --- |
| 1 Class Intro | 16 of 45 pages | robot video stills, a Go broadcast, two simulation screenshots, generated-image strips, the chip-placement diagram, the RL graphical models, the policy network |
| 2 Imitation Learning | 15 of 27 pages | the multimodal steering chart, distribution and generative-policy diagrams, success-rate bar charts, robot and driving model stills, the compounding-error and DAgger diagrams, demonstration interfaces, SFV video frames |
| 3 Policy Gradients | 11 of 28 pages | the online-RL loop and its iterations, the trajectory plots, the REINFORCE and reward-weighted-imitation slides, the humanoid and jacket-folding examples, the summary and review slides, the off-policy algorithm |
| 4 Actor-Critic Methods | 21 of 37 pages | the recap, the V/Q trajectory fans and drum example, the dissatisfying-PG stills, the loop with its fit-a-model step, the value-network and Monte Carlo/bootstrap/n-step slides, the discounted MDP, the full algorithm and review, the off-policy slides (KL/PPO ideas, replay buffer, fixes), the SAC-versus-PPO chart |
| 5 Off-Policy Actor Critic | 15 of 30 pages | the recap diagrams (online loops, V/Q trajectory fans), version 1's surrogate objective with the KL and clipping ideas, the PPO algorithm with its Half-Cheetah chart, the replay-buffer timeline and annotated broken/fixed algorithm steps, the Q-network slide, the SAC-versus-PPO chart, the SAC and PPO robot video frames and the HIL-SERL table |
| 6 Q-Learning | 12 of 29 pages | the recap diagrams (V/Q trajectory fans, the faded loop, the Q-network), the 2D-navigation thought exercise, the greedy-policy loop, the Q-learning algorithm with its two Bellman equations, the full algorithm with its replay-buffer diagram, the DQN algorithm, the Atari Q-value and DQN-versus-double-DQN charts, the n-step fan, the Atari bar chart and grasping table |
| 7 Offline RL | 14 of 33 pages | the recap loops and value-function targets, the full off-policy actor-critic algorithm, the randomly initialized Q-function plot with its data support, the two-trajectory stitching diagram, the advantage-weighted imitation slides with their s3 weights, the full AWR algorithm, the AWAC target and value histogram, the expectile-loss and fitted-expectile charts, the IQL algorithm, the CQL penalty and objective, the LinkedIn notification diagram and A/B table |
| 8 Reward Learning | 18 of 34 pages | the IQL recap with its Q-function plot, the CQL penalty, objective and log-sum-exp computation, the LinkedIn diagram and A/B table, the reward-source pictures, the goal-classifier photos and algorithm, the cloth-task frames, the GAN examples, the trajectory-preference sketches, the preference objective and algorithm, the backflip and driving-weight figures, the omelette RLHF slides, the Constitutional AI chart, the asymmetric self-play diagram |
| 9 RL for LLMs | 24 of 57 pages | the scale chart and AI 2027 screenshot, the assistant start screens, the Khan Academy graph, the pretrain/finetune diagrams, the FLAN-T5 diagram and Flan-T5 table, the crossed-out LM outputs, the InstructGPT pipeline (twice), the REINFORCE, Bradley-Terry, KL-penalty and DPO equation slides, the RLHF summary chart, the PPO-RLHF pipeline, the DPO win-rate bars, the Open LLM Leaderboard, the boat-race still, the over-optimization chart, the STaR loop, the PRISM and FSPO figures |
| 10 RL for LLM Reasoning | 27 of 41 pages | the next-token training diagram, the data-stock chart and robot datasets, the reasoning MDP, the three data axes, the first frames of the animated result charts (10, 19, 23), the RFT-scaling chart, the spurious-steps picture, the credit, rollout, Q-value and advantage slides, the filtering and DPO-pair diagrams, the online-RL and PAV slides, the dense-reward and hard-question charts, the compute-constrained objective, the meta-strategy and Kimi length charts, the test-time scaling diagram, the leaderboards, the open questions |
| 11 Model-Based RL | 17 of 38 pages | the recap map of offline and online algorithms, the Veo 2 and Sora stills, the notional algorithm with its maze and Sora frame, the backpropagation and sampling planning chains, the gradient-versus-sampling plots with CEM, the cliff photographs and the fixed loop, the open/closed-loop diagram, the MPC slide with its car, the MBPO roll-out diagram, the case study's setup, simulated charts, ablations and real-hand chart |
| 12 Multi-Task RL | 9 of 29 pages | the model-based policy optimization roll-outs, the task-distribution examples, the multi-task RL loop, the BC-Z and OpenVLA architectures, the two π0.5 video frames, the hockey relabeling example, the HER task renderings and chart |
| 13 Meta RL | 19 of 39 pages | the multi-task fan recap, the espresso photograph, the Braque and Cezanne paintings and the few-shot framing, the maze meta-train and meta-test panels, the task-distribution examples, the black-box chain and its unrolled algorithm, the three architectures, the SNAIL maze video frames, the test-time compute chart, the multi-task digression, the inferred-task chain, the PEARL charts, the hallway and kitchen photographs, the coupling diagram, the posterior-sampling and alternative-exploration panels |
| 14 Exploration | 18 of 34 pages | the Breakout and Montezuma frames, the Mao cards, the tractability spectrum, the regret and bandit-strategy slides, the kitchen and reasoning examples, the coupling cycle, the alternative strategies' trajectory panel, the Idea 3.0 boxes, the DREAM diagrams and objective, the sample-complexity and navigation charts, the navigation and video frames, the Bounce and Breakout frames and the grading results |
| 15 Hierarchical RL and IL | 20 of 46 pages | the recap policies, the two-level policy and the hierarchy/flat/chain-of-thought comparison, the think-pair-share, the four robot systems and their architectures, the segmented demonstrations, the ALOHA architecture and the language-DAgger diagram, the correction and long-horizon video frames, the hierarchy and DAgger charts, the image-subgoal architecture and SuSIE frames, the chain-of-thought figure, the hierarchical-RL mazes, the skill-discovery frames |

## Rebuilding

Built and updated by the `cairn-kb` skill. To add newly released lectures,
append entries to `TODO.md` and re-run the skill — it only does unchecked work.
When a run is committed and pushed, move its `TODO.md` section verbatim to
`BUILD_LOG.md`, so `TODO.md` keeps only open work. Add here only a lesson
that applies to every lecture, stated in general form.
