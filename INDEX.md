# Stanford CS224R — Deep Reinforcement Learning (Spring 2025)

A graduate course on deep reinforcement learning taught by **Chelsea Finn** at Stanford, with
guest lectures by Archit Sharma, Aviral Kumar and Ashish Kumar. The course website is
<https://cs224r.stanford.edu/spring_2025/>. It covers imitation learning, policy gradients,
actor-critic and Q-learning methods, offline RL, reward learning, RL for large language models
(preference optimization and reasoning), model-based RL, multi-task, goal-conditioned and meta-RL,
exploration, hierarchical RL, and RL for robots. The emphasis is applied and on methods that scale
to deep neural networks, with use-cases in robotics and language modeling; the site describes it as
complementary to the more theoretical CS234.

> **Coverage: Lectures 1–3 only, of the 19 recordings in the catalog** (18 lectures and one
> Q-learning TA session). This knowledge base holds Lecture 1 (course intro and MDPs), Lecture 2
> (imitation learning) and Lecture 3 (policy gradients): their edited transcripts, their slide
> decks as text, 42 of their slides as images, and the topic pages they start. **It cannot answer
> questions about the technical content of Lectures 4–19** (actor-critic methods onward). For those, point the learner to
> the lecture's slide deck, listed with every other course document in [`sources.md`](sources.md).
> Build progress is in [TODO.md](TODO.md).

## How to read this knowledge base

- Start with the **wiki** pages below. They explain, cite the slide ("slide N", the printed
  number) or the transcript time (≈M:SS) for every claim, and link to each other.
- The **transcripts** in `raw/transcripts/` are copy-edited from the YouTube captions:
  punctuated, filler removed, mis-heard terms restored, student turns marked. Read these, not
  the captions. Only the lecturer is audible, so students' questions appear only as the lecturer
  repeats them. `[Ed: …]` notes mark restorations and unclear passages.
- The **slide files** in `raw/slides/` hold the full text of every slide, numbered, with every
  figure described in prose. Prefer them for anything written down: equations, numbers,
  citations. Lecture 1's recording leaves out the course-logistics slides (5–6 and 10–15); their
  content is only in the slide file and on [course logistics](wiki/course-logistics.md).
  Lecture 2's recording ends after slide 21, so its slides on collecting demonstrations, its
  summary and its closing reminders (22–27) are only in the slide file and on the wiki pages that
  say so. Lecture 3's recording does not discuss slide 2's reminders, and it shows the KL
  constraint on slide 25 but skips it for time.

### Images

Lectures 1, 2 and 3 have images: 16 slides of Lecture 1, 15 of Lecture 2 and 11 of Lecture 3
rendered as pictures, each shown under its `## Slide N` heading in the lecture's slide file
([`raw/slides/01-class-intro.md`](raw/slides/01-class-intro.md),
[`raw/slides/02-imitation-learning.md`](raw/slides/02-imitation-learning.md),
[`raw/slides/03-policy-gradients.md`](raw/slides/03-policy-gradients.md)) and, where a wiki page
cites that slide, in the passage that cites it. **No other lecture has images yet**, and not every
slide of these three has one: the slide file's header lists which do.

- Use an image path you have actually read in a file. Never construct one from the pattern, and
  never assume a slide has an image because a neighbouring one does.
- Image links are relative. To show one, read its path; what comes back is the URL to display.
- Prefer the slide file's transcription for numbers and equations, and show one image at a time,
  keeping the slide citation with it.

## Lecture pages

- [Lecture 1 — Class Intro (course intro and MDPs)](wiki/01-class-intro.md): what deep RL means
  and how it differs from supervised learning (indirect feedback, non-i.i.d. data); four reasons to
  study it, with legged robots, robot manipulation, AlphaGo's "Move 37", LLM post-training,
  traffic control, image models trained to follow prompts, chip placement, Finn's PR2 learning
  story, the open research problems, and the Yevgen reset video; representing experience as data
  (states, observations, actions, trajectories, rewards); the Markov property; the towel-robot and
  chatbot worked examples and the think-pair-share; the policy as a neural network, roll-outs and
  memory; the RL objective, its two sources of randomness, the horizon and the discount factor;
  why policies are stochastic; value and Q-functions; the five algorithm families and why RL has so
  many; and the MDP/POMDP recap.

- [Lecture 2 — Imitation Learning](wiki/02-imitation-learning.md): learning a policy from expert
  demonstrations; version 0 (regression to the expert's actions) and why it fails on multimodal
  data such as drivers who merge or stay straight; policies that output distributions, from
  categorical and Gaussian to Gaussian mixtures, discretized autoregressive models and diffusion;
  training by maximum likelihood and cross-entropy; network versus distribution expressivity;
  version 1 and its evidence (transport and shirt-hanging success rates); expressive policies in
  π0, Gr00t N1, Helix, OpenVLA, Waymo EMMA and Wayve LINGO-2; offline versus online; compounding
  errors and covariate shift; DAgger and human-gated DAgger, with safety drivers; and, from the
  slides only, how to collect demonstrations, the embodiment gap and the summary.

- [Lecture 3 — Policy Gradients](wiki/03-policy-gradients.md): the online RL loop; evaluating the
  objective from samples; the derivation of the policy gradient (the integral, the log-gradient
  identity, and why the dynamics drop out); the REINFORCE algorithm; the gradient as imitation
  weighted by reward; three worked examples (a humanoid learning to walk, every reward positive,
  folding a jacket) and the student questions on them; causality (sums of future rewards) and
  baselines (why subtracting the average reward is unbiased); the surrogate objective and its
  printed typo; on-policy versus off-policy; importance sampling and the off-policy policy
  gradient's common final form; and, from the slides only, the KL constraint.

## Topic pages

- [MDPs and RL notation](wiki/mdp-and-notation.md): a table of the course's symbols (state,
  observation, action, trajectory, reward, initial state distribution, dynamics, policy), what
  each means in the towel and chatbot examples, the Markov property and why observations break
  it (and why a fixed horizon needs the time remaining in the state), when to treat something as
  a state or an observation, and the extra notation for demonstrations and for policy gradients.
- [Policies](wiki/policies.md): the policy $\pi_\theta(\mathbf{a} \mid \mathbf{s})$ as a neural
  network, the roll-out loop and the words "roll-out" and "episode", policies with memory for
  observations, and the two reasons policies are stochastic.
- [The RL objective](wiki/rl-objective.md): from the sum of rewards to the expected sum, the two
  sources of randomness, the trajectory distribution $p_\theta(\tau)$, why the policy enters only
  through that distribution, the horizon $T$, the discount factor, and how Lecture 3 estimates the
  objective from samples and differentiates it.
- [Value functions](wiki/value-functions.md): the definitions of $V^\pi(\mathbf{s})$ and
  $Q^\pi(\mathbf{s}, \mathbf{a})$ as Lecture 1 gives them.
- [Imitation learning](wiki/imitation-learning.md): the problem (demonstrations from an unknown
  expert policy), behavior cloning versions 0 and 1, compounding errors and covariate shift,
  corrective data with DAgger and human-gated DAgger, a table of their trade-offs, how
  demonstrations are collected, and how the policy gradient relates to it (an imitation gradient
  weighted by reward; imitation learning as initialization).
- [Expressive policy distributions](wiki/expressive-policy-distributions.md): why a policy needs
  more than a mean; categorical, Gaussian, Gaussian-mixture, discretized autoregressive and
  diffusion policies, what each outputs and how each is trained; why a bigger network does not
  help; and the evidence for expressive policies.
- [Offline and online algorithms](wiki/offline-and-online.md): the offline/online definitions
  Lecture 2 introduces, why online data can be expensive or unsafe, which imitation methods are
  which, and Lecture 3's on-policy/off-policy distinction among online algorithms.
- [Policy gradients](wiki/policy-gradients.md): the method as a whole, from the online loop and
  the derivation to REINFORCE, the reading as reward-weighted imitation, its three weaknesses
  (high variance, sensitivity to reward scale, need for dense rewards and large batches) and the
  two fixes (causality and baselines), the surrogate objective, and the off-policy version.
- [Importance sampling](wiki/importance-sampling.md): estimating an expectation under one
  distribution with samples from a proposal distribution, its catches (support, similarity), and
  how it makes the policy gradient off-policy, including why per-trajectory ratios explode.
- [RL algorithm families](wiki/rl-algorithm-families.md): imitation learning, policy gradients,
  actor-critic, value-based and model-based methods, which lecture of the schedule covers each,
  how the first two relate, and the trade-offs (data cost, supervision, stability, action space, dynamics) that explain why RL
  has so many algorithms.

## Course pages

- [Course logistics](wiki/course-logistics.md): staff and guest lecturers, prerequisites, the four
  homeworks and their topics, grading, the default and custom final projects, late days, the AI
  tools policy, and the first week's steps, from Lecture 1's slides and the course site.

## Raw materials

- [`raw/transcripts/`](raw/transcripts/) — copy-edited lecture transcripts with `[MM:SS]`
  paragraph marks. The verbatim captions are not committed; each transcript says how to
  regenerate them.
- [`raw/slides/`](raw/slides/) — full text of every slide, numbered, with figures described.
- [`raw/images/`](raw/images/) — rendered figure slides, linked from the pages that use them.
- [`sources.md`](sources.md) — every course document by lecture, with canonical URLs: slide
  decks, homeworks, project guidelines and the site's optional readings.
- [`LICENSE.md`](LICENSE.md) — what may be reused, and the attribution of every image.

Slide and handout PDFs are **not** committed to this repo; [`sources.md`](sources.md) has the
canonical URL of each.
