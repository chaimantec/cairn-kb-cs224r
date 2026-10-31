# Stanford CS224R — Deep Reinforcement Learning (Spring 2025)

A graduate course on deep reinforcement learning taught by **Chelsea Finn** at Stanford, with
guest lectures by Archit Sharma, Aviral Kumar and Ashish Kumar. The course website is
<https://cs224r.stanford.edu/spring_2025/>. It covers imitation learning, policy gradients,
actor-critic and Q-learning methods, offline RL, reward learning, RL for large language models
(preference optimization and reasoning), model-based RL, multi-task, goal-conditioned and meta-RL,
exploration, hierarchical RL, and RL for robots. The emphasis is applied and on methods that scale
to deep neural networks, with use-cases in robotics and language modeling; the site describes it as
complementary to the more theoretical CS234.

> **Coverage: all 19 recordings in the catalog** (18 lectures and one Q-learning TA session). This knowledge base holds Lecture 1 (course intro and MDPs), Lecture 2
> (imitation learning), Lecture 3 (policy gradients), Lecture 4 (actor-critic methods), Lecture 5
> (off-policy actor-critic: PPO and soft actor-critic), Lecture 6 (Q-learning: DQN, double
> Q-learning, n-step returns), Lecture 7 (offline RL: filtered behavior cloning, AWR, AWAC and
> implicit Q-learning), Lecture 8 (conservative Q-learning, and reward learning from goal
> classifiers and from human preferences, including RL from human feedback), Lecture 9 (a guest
> lecture on RL for language models: instruction finetuning, RLHF and direct preference
> optimization), Lecture 10 (a guest lecture on RL for LLM reasoning: rejection finetuning,
> per-step credit assignment, offline and online RL, and thinking models), Lecture 11 (model-based
> RL: learning dynamics models, planning by backpropagation and by sampling, model predictive control,
> and a dexterous-manipulation case study), Lecture 12 (model-based policy optimization and when to
> use a model; multi-task imitation and RL, goal-conditioned RL and hindsight relabeling), Lecture 13
> (meta-RL: the problem statement, black-box methods and the challenge of learning to explore),
> Lecture 14 (exploration: regret and bandit strategies, exploration in large MDPs, and DREAM's decoupled
> exploration and execution in meta-RL), Lecture 15 (hierarchical imitation and RL: high-level and
> low-level policies, goal representations, supervising each level, when to move on to the next goal, and
> systems with language and image subgoals), Lecture 16 (RL for robots without human resets: defining
> and evaluating autonomous RL, forward-backward RL, MEDAL, practising cycles of tasks, and single-life
> RL), Lecture 17 (a guest lecture on sim-to-real transfer: Rapid Motor Adaptation, legged locomotion
> from egocentric depth without maps, in-hand rotation and drones, and what is left for humanoids and
> manipulation), Lecture 18 (the last lecture: open problems in deep RL, from rewards, prior data, world
> models and scaling to safety, calibration, reliability and evaluation, and Finn's advice on how to do
> research), and Lecture 19 (the Q-learning TA session, by Anikait Singh: Q-value iteration on a grid world,
> the optimal value, Q-value, policy and advantage, Monte Carlo, TD and n-step targets derived on the board,
> stitching, semi-gradients and target networks, gradient clipping and the Huber loss, replay buffers, and
> overestimation with critic ensembles): their edited transcripts, their slide decks as text, 356 of their
> slides as images, and the topic pages they start.
> Build progress is in [TODO.md](TODO.md).

## How to read this knowledge base

- Start with the **wiki** pages below. They explain, cite the slide ("slide N", the printed
  number) or the transcript time (≈M:SS) for every claim, and link to each other.
- The **transcripts** in `raw/transcripts/` are copy-edited from the YouTube captions:
  punctuated, filler removed, mis-heard terms restored, student turns marked. Read these, not
  the captions. In Lectures 1–18 only the lecturer is audible, so students' questions appear only as the
  lecturer repeats them; in Lecture 19 three students' questions are captioned and kept, in quotation
  marks after `*[Student question]*`. `[Ed: …]` notes mark restorations and unclear passages.
- The **slide files** in `raw/slides/` hold the full text of every slide, numbered, with every
  figure described in prose. Prefer them for anything written down: equations, numbers,
  citations. Lecture 1's recording leaves out the course-logistics slides (5–6 and 10–15); their
  content is only in the slide file and on [course logistics](wiki/course-logistics.md).
  Lecture 2's recording ends after slide 21, so its slides on collecting demonstrations, its
  summary and its closing reminders (22–27) are only in the slide file and on the wiki pages that
  say so. Lecture 3's recording does not discuss slide 2's reminders, and it shows the KL
  constraint on slide 25 but skips it for time. Lecture 4's recording ends on the review slide
  (23), so its off-policy actor-critic section, comparison chart and closing slides (24–37) are
  only in its slide file; Lecture 5 teaches that material aloud from a revised deck. Lecture 5's
  recording covers slides 3–28; slide 2's reminders and the closing plan and next-time slides
  (29–30) are only in the slide file. Lecture 6's recording covers slides 3–29 but does not discuss
  slide 2's reminders or slide 29's next-week topics, which are only in the slide file. Lecture 7's
  recording covers slides 3–23 and 30–32: it does not teach conservative Q-learning (CQL, slides
  24–28) or slide 31's application, which Lecture 8 teaches aloud from the same slides; slides 2 and
  33 are only in the slide file. Lecture 8's recording covers slides 3–32; its reminders (slides 2
  and 34, which announce the two guest lectures) and its repeated plan (slide 33) are only in the
  slide file. Lecture 9's deck prints slide numbers 1–99 on 57 pages, because hidden slides were not
  exported, so cite the printed number, never the page. Its recording follows the deck but does not
  discuss slides 81, 93, 97 and 98, which are only in the slide file. For what came before
  instruction finetuning in the deck it is based on (prompting, chain-of-thought), see the CS224N
  knowledge base listed in [`SEE_ALSO.md`](SEE_ALSO.md). Lecture 10's recording covers slides 2–27
  and then slide 32; slides 28–31 and 33–41 (thinking models: the formulation, meta strategies,
  response length, test-time scaling, leaderboard results, open questions) are only in the slide
  file. Its result charts on slides 10, 19 and 23 hold only the first frame of an animation, so
  their results come from what the lecturer says. Lecture 11's recording covers slides 4–25 and
  32–36: Finn leaves model-based policy optimization for the next lecture, so slides 26–30 (its
  augmented roll-outs and full algorithm, when to use model-based RL, other kinds of models) and the
  reminders (slides 2, 37–38) are only in Lecture 11's slide file; Lecture 12 teaches that material
  aloud from its own version of those slides. Lecture 12's recording covers slides 3–25 and the start
  of slide 26, then ends at about 1:10:21: the rest of goal-conditioned relabeling (relabeling with the
  last state, slide 26), the simulated-manipulation results (slide 27), the closing plan and next time
  (28–29) and slide 2's reminders are only in the slide file. Its slide 24 prints "task 2 ID" where
  Finn says the passing task, task 1, is meant. Lecture 13's recording covers slides 3–37 and ends on
  slide 37's summary; slide 2's reminders and the closing plan, next time and reminders (38–39) are only
  in the slide file. Its slide 20 holds frames of two maze videos, whose content comes from what Finn
  says, and Finn writes the black-box algorithm on a board the recording does not show. Lecture 14's
  recording covers slides 3–32 and ends on slide 32's grading results; slide 2's reminders and the
  closing plan and reminders (33–34) are only in the slide file. Its slide 12 prints only "Exploration
  game": the drug-trial game the class plays runs on a website, described from the transcript on the
  lecture page. Finn's regret curves and the bottleneck and DREAM derivations are drawn where the
  recording does not show them, and slides 21–23 print the unit Gaussian as N(0, 1) where Finn says
  the identity is meant. Slides 27, 30 and 31 hold frames of videos. Lecture 15's recording covers slides
3–44 and closes on slide 46's "next time"; slide 45's summary outline and the reminders on slides 2 and 46
are only in the slide file. Finn draws the hierarchical policy, the roll-out procedure and a pasta-making
example's sets of states on a board the recording does not show, so the lecture page describes them from
the transcript. Slides 31–33, 35, 36 and 39 hold frames of videos, and slide 43's video is not in the PDF.
Lecture 16's recording covers slides 3–45 and ends on slide 45's summary; the reminders and "next time" on
slides 2 and 46 are only in the slide file. Its deck prints a slide number on only three pages, but every
slide's number is its page number. Finn writes the evaluation formulas and the forward-backward and
task-cycle algorithms on the slides where the recording does not show it, so the lecture page describes them
from the transcript and the slides that follow. Slides 8, 26, 30, 31, 33, 39 and 42–44 hold frames of videos
(37–38 the first frame of 39's), and slide 32 most likely the first frame of one.
Lecture 17, a guest lecture by Ashish Kumar, is about 50 minutes; its recording covers slides 2–78 and ends
on slide 78's question, and seven student questions (inaudible; Kumar restates most of them) come between
slides 41 and 42. Its deck prints a number only on slides 62–67, but every slide's number is its page
number. Slides 5–11, 22–27, 30, 31, 33–36, 45, 58, 62, 67, 69, 70, 73 and 74 hold frames of videos, and
most likely 55 and 56; what they showed comes from the transcript. Slide 46 is blank in the PDF.
Lecture 18, the last lecture, is about 71 minutes; its recording opens on the plan for both parts and ends on
slide 52's closing thoughts. Slide 2's reminders are not mentioned, so they are only in the slide file and on
the wiki pages, and slide 53's thank-you is not read aloud. Its deck prints no number on slides 40, 52 and 53,
but every slide's number is its page number. Five student questions (inaudible; Finn restates them) are
answered where they arise, and slide 39's two image panels are frames of videos Finn describes.
Lecture 19, the Q-learning TA session given by Anikait Singh, is about 51 minutes and **ends on slide 31**,
during the discussion of critic ensembles: slides 32–35, the walkthrough of the Homework 2 algorithms with
DQN and soft actor-critic pseudocode, are only in the slide file and on the lecture page. Its deck prints a
number only on its title page, so every label is the slide's page position. Two passages are worked on a
whiteboard the recording does not show (the Monte Carlo, TD and n-step derivation, whose lines slide 20
prints, and the TD loss). Three student questions are captioned (≈7:47, 10:09, 12:26) and five are inaudible (two of them at ≈41:19).

### Images

Every lecture, 1 to 19, has images: 16 slides of Lecture 1, 15 of Lecture 2, 11 of Lecture
3, 21 of Lecture 4, 15 of Lecture 5, 12 of Lecture 6, 14 of Lecture 7, 18 of Lecture 8, 24 of Lecture 9, 27 of
Lecture 10, 17 of Lecture 11, 9 of Lecture 12, 19 of Lecture 13, 18 of Lecture 14, 20 of Lecture 15, 25 of Lecture 16, 39 of Lecture 17, 15 of Lecture 18 and 21 of Lecture 19 rendered as pictures, each shown under its
`## Slide N` heading in the lecture's slide file ([`raw/slides/01-class-intro.md`](raw/slides/01-class-intro.md),
[`raw/slides/02-imitation-learning.md`](raw/slides/02-imitation-learning.md),
[`raw/slides/03-policy-gradients.md`](raw/slides/03-policy-gradients.md),
[`raw/slides/04-actor-critic.md`](raw/slides/04-actor-critic.md),
[`raw/slides/05-off-policy-actor-critic.md`](raw/slides/05-off-policy-actor-critic.md),
[`raw/slides/06-q-learning.md`](raw/slides/06-q-learning.md),
[`raw/slides/07-offline-rl.md`](raw/slides/07-offline-rl.md),
[`raw/slides/08-reward-learning.md`](raw/slides/08-reward-learning.md),
[`raw/slides/09-rl-for-llms.md`](raw/slides/09-rl-for-llms.md),
[`raw/slides/10-rl-for-reasoning.md`](raw/slides/10-rl-for-reasoning.md),
[`raw/slides/11-model-based-rl.md`](raw/slides/11-model-based-rl.md),
[`raw/slides/12-multi-task-rl.md`](raw/slides/12-multi-task-rl.md),
[`raw/slides/13-meta-rl.md`](raw/slides/13-meta-rl.md),
[`raw/slides/14-exploration.md`](raw/slides/14-exploration.md),
[`raw/slides/15-hierarchical-rl.md`](raw/slides/15-hierarchical-rl.md),
[`raw/slides/16-autonomous-rl.md`](raw/slides/16-autonomous-rl.md),
[`raw/slides/17-sim-to-real.md`](raw/slides/17-sim-to-real.md),
[`raw/slides/18-frontiers.md`](raw/slides/18-frontiers.md),
[`raw/slides/19-q-learning-tutorial.md`](raw/slides/19-q-learning-tutorial.md)) and, where a wiki page cites that
slide, in the passage that cites it. Not every slide has one: each slide file's header lists which do.

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

- [Lecture 4 — Actor-Critic Methods](wiki/04-actor-critic.md): the value, Q and advantage
  functions and the drum-practice exercise; why policy gradients waste data (the falling humanoid,
  the jacket); replacing the reward to go with Q and the baseline with V, giving the advantage;
  the online loop's third step (fit a model of the expected return); why only $V^\pi$ is fitted;
  Monte Carlo estimation, bootstrapping (temporal difference learning) and the think-pair-share
  comparing them; bias and variance; n-step returns; the discount factor as a probability of dying;
  the full actor-critic algorithm and the names actor and critic; and, from the slides only,
  off-policy actor-critic (importance weights, the KL and PPO ideas, replay buffers, Q-function
  critics, soft actor-critic, and a SAC-versus-PPO learning-curve comparison, all taught aloud in
  Lecture 5).

- [Lecture 5 — Off-Policy Actor Critic](wiki/05-off-policy-actor-critic.md): the recap with an
  AlphaGo value-function example; taking multiple gradient steps per batch with importance weights;
  the surrogate objective and why too many steps overfit stale advantages (learning-curve and
  grid-world sketches); keeping the policy close with a KL penalty (with $\beta$, memory and caching)
  or by clipping the importance weights; PPO's three tricks (clipping, the minimum as a lower bound
  with the $-1000$ worked example, generalized advantage estimation), its algorithm, example
  hyperparameters and Half-Cheetah chart; replay buffers; the think-pair-share on which policy a
  buffer-fitted value function belongs to; fitting a Q-function with actions from the current
  policy, and action coverage; the fixed policy update; soft actor-critic; SAC versus PPO; and
  robot examples (walking and HIL-SERL assembly with SAC, sim-to-real and a Rubik's cube with PPO).

- [Lecture 6 — Q-Learning](wiki/06-q-learning.md): the recap of fitting a Q-function from a replay
  buffer; the 2D-navigation thought exercise on the greedy policy a Q-function defines (better, not
  optimal) and its follow-ups (iterating, discounting, dense rewards); why the greedy policy is at
  least as good, so the policy gradient can go; continuous actions and sparse rewards (backups);
  policy evaluation, improvement and policy iteration; the max target ("Q-values for new policy");
  the Bellman equation and Bellman optimality equation, and why Q rather than V; convergence (tables
  yes, in general no); epsilon-greedy and Boltzmann exploration; full Q-learning with a replay buffer,
  the restaurant example and buffer management; target networks and DQN; Q-values over training on
  Atari and the rising-loss tip; overestimation and double Q-learning with the current and target
  networks; n-step returns and why they are only correct on-policy (with the slide's printed
  exponent slip); Atari and robot-grasping results; and Finn's advice on choosing PPO, DQN or SAC.

- [Lecture 7 — Offline RL](wiki/07-offline-rl.md): learning from a fixed dataset collected by an
  unknown behavior policy; the recap of Monte Carlo, bootstrapped and Q-function targets and action
  coverage; why offline RL (existing data, safety, reuse) and its relation to off-policy RL; the
  formal setting and distribution shift; why soft actor-critic fails on fixed data (the car's
  out-of-distribution actions and overestimated Q-values); offline RL versus imitation learning and
  trajectory stitching; filtered behavior cloning; advantage-weighted imitation and its KL-constrained
  reading; advantage-weighted regression (AWR) with the deterministic-data think-pair-share and the
  worked advantage example; advantage-weighted actor-critic (AWAC); expectile regression and implicit
  Q-learning (IQL), with the questions on choosing lambda; the summary and which algorithm to use
  (with IDQL); and, from the slides only, conservative Q-learning (CQL) and the LinkedIn application,
  which Lecture 8 teaches aloud.

- [Lecture 8 — Reward Learning](wiki/08-reward-learning.md): the end of offline RL and the start of
  reward learning. The recap of offline RL and implicit Q-learning, and why IQL's asymmetric loss is
  for the value function and not the Q-function (the luck argument, with a worked example);
  conservative Q-learning (CQL) taught aloud: pushing down on large Q-values, pushing back up on the
  data's, the lower bounds, the role of alpha and of mu, the student questions, the full algorithm, the
  log-sum-exp form, and the LinkedIn notification application; why be conservative; the offline RL
  summary. Then where rewards come from (games, pouring water, dialog, driving, proxies) and why
  imitation is not enough; goal classifiers, how RL exploits them, adding visited states as negatives,
  regularization, 50/50 balancing and the false-negative argument; the self-improving robot (26% to
  62%); the GAN connection; learning from human preferences (relative preferences, the sigmoid
  objective, partial roll-outs, the complete algorithm, in the loop of RL); the backflip and driving
  examples; RL from human feedback for language models (the omelette prompt, the three-stage pipeline,
  PPO); AI feedback (RLAIF, Constitutional AI); the summary and other forms of feedback; agents that
  set their own goals; and, from the slides only, the reminders announcing the two guest lectures.

- [Lecture 9 — RL for LLMs: Preference Optimization](wiki/09-rl-for-llms.md) (guest lecture by
  Archit Sharma): how a pretrained language model becomes an assistant. Growing compute, what
  pretraining learns and language models as world models; why language modeling is not assisting
  users (GPT-3 and the moon landing); instruction finetuning, Flan-T5's bigger-model-bigger-gain
  table and the limitations (cost, no right answer, equal penalties, a mismatched objective), with
  the format-versus-knowledge question; optimizing human preferences, the InstructGPT pipeline, the
  policy-gradient recap with the log-derivative trick, reward models from pairwise comparisons
  (Bradley-Terry, calibration, scale), why RL is needed when the reward model is differentiable
  (discrete tokens, Gumbel-softmax), the KL penalty, RLHF beating human summaries, and reward-model
  accuracy and intransitive preferences; direct preference optimization (DPO), its derivation from
  the closed-form optimal policy with the partition function cancelled on the board, the questions
  on beta, reward models and sampling, and its win rates; InstructGPT, ChatGPT, DPO in open models
  and the stylistic changes; engagement and whose preferences; sycophancy, reward hacking and
  reward-model over-optimization; and what's next (verifiable rewards, personalization).

- [Lecture 10 — RL for LLM Reasoning](wiki/10-rl-for-reasoning.md) (guest lecture by Aviral
  Kumar): training language models to solve math problems with RL. Why next-token prediction on
  reference solutions scales slowly (the data-similarity bound, running out of text by 2028, an
  o1 proof that asserts its way out); reasoning as a sparse-reward MDP with deterministic dynamics;
  the three data axes (SFT, rejection finetuning, RL on incorrect solutions); rejection finetuning
  on the board and the questions on it; the fitted exponents and RFT's 2x efficiency; why too much
  on-policy imitation hurts, deduplication, and spurious steps (the 100 times 2 example, causal
  confusion); credit for steps by rollouts from each prefix; the Q-value of a step under a rollout
  policy, which rollout policy to use, and the advantage as one Q-value minus the previous one with
  slide 17's worked example; process reward models; option 1, filtering steps by advantage (versus
  AWR's exponentiated weights), with the questions on training single steps and the Markov
  property; option 2, per-step DPO pairs, the rollout policy's quality versus diversity, and how to
  split a solution into steps; the 8x data efficiency of offline RL, $\text{pass@}k$ and the
  temperature
  question; online RL with a 0/1 reward and GRPO; process advantage verifiers and their 5–6x sample
  efficiency, with the generalization question; what changed for thinking models (same objective,
  macro actions, a larger token budget); and, from the slides only, solving hard questions, a
  thinking trace, the compute-constrained objective, meta strategies, longer responses, test-time
  scaling, the leaderboards and the open questions.

- [Lecture 11 — Model-Based RL](wiki/11-model-based-rl.md): the recap that maps every algorithm so
  far (offline and online, on- and off-policy, actor, critic or both); the key idea of learning a
  simulator (physics, action-conditioned video prediction with Veo 2 and Sora, finance, games); the
  notional learned-simulator algorithm and what goes wrong (data coverage and the maze, the Sora
  cube-stacking video, model inaccuracies); learning a dynamics model when it is known, partly known
  or unknown, in image space or a latent space, with a reward model; planning by backpropagating
  rewards into actions, with its chain rule, the corrected sign of its update and the questions on
  it; gradient-based against sampling-based optimization, random sampling and the cross-entropy
  method, with the questions on finite differences, the "policy" in quotes and planning latency;
  random shooting and CEM as planners; the cliff and the data distribution mismatch, fixed by
  refitting on executed data; open loop against closed loop and model predictive control, with
  warm-starting and how often to replan; the pros and cons of planning; distilling a planner and
  planning with a terminal value function; model-based policy optimization, when to use a model and
  other kinds of models (slides only); and the PDDM case study on a 24-joint robot hand (ensembles,
  a modified CEM, simulated results and ablations, Baoding balls on a real hand in about four hours).
- [Lecture 12 — Multi-Task RL](wiki/12-multi-task-rl.md): the end of model-based RL and the start of
  multi-task learning. The recap of planning; model-based policy optimization, which trains a policy on
  short synthetic roll-outs branching off every real state, with the students' ideas for augmenting a
  trajectory, the full algorithm and the questions on convergence, rewards, compute, stability,
  hyperparameters and offline versions; when to use model-based RL (the maze against pouring water,
  robotics with compact states against images); other kinds of models (inverse, multi-step inverse,
  action-free prediction). Then what multi-task RL is and why (data efficiency, shared structure,
  generalists); a task as an MDP, with recommendation, character-animation, garment and multi-robot
  examples; task identifiers and the task as part of the state; goal-conditioned RL and its
  negative-distance reward; multi-task imitation learning with stratified sampling; conditioning
  architectures (BC-Z's language and video encoders, OpenVLA's prompt); the towel and kitchen videos
  and the questions on per-task policies, hierarchy, long horizons and vision tasks; multi-task RL and
  sharing data across tasks; hindsight relabeling with the hockey example (and slide 24's typo), the
  algorithm, which task to relabel for, and when relabeling applies (same dynamics, a computable reward,
  an off-policy learner, with Finn's correction on importance weights); and goal-conditioned
  relabeling, with the last state as goal and the HER results (slides only past ≈1:10:21).
- [Lecture 13 — Meta RL](wiki/13-meta-rl.md): learning a new task from a few episodes of experience.
  The recap of multi-task and goal-conditioned RL; why meta-RL (the espresso machine, not starting from
  scratch); forward, multi-task and meta-learning transfer, zero-shot against few-shot; few-shot learning
  outside RL (Braque or Cezanne, in-context learning in GPT-3); the maze-navigation example, the
  exploration-exploitation trade-off and the key assumption of a shared task distribution; task
  distributions (mazes, terrains, manipulation, dialog users); the formal setting, a task as an MDP, and
  the episodic and online variants with exploration and task policies; black-box meta-RL, a sequence
  model fed states, rewards and actions with memory across episodes, and how it differs from a recurrent
  policy; the algorithm and its meta-test, with the questions on new episodes and initial states; RNN,
  attention-and-convolution and feedforward-and-average architectures; SNAIL's maze videos and table; meta-RL
  for test-time compute (DeepSeek R1 on 2+2, MRT against GRPO on Math 500); experience as a task
  identifier and meta-RL as a POMDP; off-policy meta-training (PEARL against RL², ProMP and MAML) and
  per-task trajectory buffers; the black-box pros and cons; and exploration: end-to-end training, the
  hallway and kitchen examples, the coupling problem, posterior sampling and when it fails, intrinsic
  rewards, and task dynamics and reward prediction.
- [Lecture 14 — Exploration](wiki/14-exploration.md): exploration and learning to explore. Why
  Montezuma's Revenge is hard and Breakout is not; the card game Mao; exploration against exploitation
  (restaurants, ads, oil drilling); the tractable-to-intractable range of problems and multi-armed
  bandits; regret, its formula and Finn's regret curves (linear against sublinear), with the questions on
  them; optimism under uncertainty (UCB) and probability matching (posterior sampling); the drug-trial
  exploration game and its comparison of greedy, epsilon-greedy, UCB and posterior sampling; why
  exploration from scratch in large MDPs is intractable and demonstrations, base models and shaped
  rewards stand in for it; exploring at test time in meta-RL, the coupling problem and the alternative
  strategies recapped; explore-to-identify-the-task (Idea 3.0) and its meta-test; DREAM's bottlenecked
  task representation, the variational information bottleneck and its questions, the execution
  objective (with Finn's correction of N(0, 1) to the identity), the per-step information-gain reward
  for exploration, its sample-complexity analysis against RL² and the 3D navigation results; the three
  ways to learn to explore compared; and exploring students' Bounce and Breakout programs to find bugs,
  with CS106A's 44% faster and 6% more accurate grading.
- [Lecture 15 — Hierarchical RL and IL](wiki/15-hierarchical-rl.md): stringing subtasks together into
  long-horizon tasks. Why long-horizon tasks are hard (many states, mistakes, getting stuck); the main idea,
  a high-level policy that predicts intermediate goals and a low-level policy that pursues them at a higher
  frequency, with the cheesecake decomposition and the roll-out procedure; why hierarchy may help
  (supervision, knowledge sharing, structured exploration, compute) and the questions on it; hierarchy
  against a flat policy and chain of thought; the goal-representation think-pair-share (cooking, biking,
  legal briefs, vacations) and what makes a good representation; supervising each level, with the pasta
  example, why end-to-end training gives a flat policy, gaps between skills, adapting the levels to each
  other and LLMs as high-level policies; when to move on (when done, or after a fixed number of steps);
  π0.5, Gr00t N1, Helix and Gemini Robotics; hierarchical imitation with language subgoals (segmented
  demonstrations, DAgger on the high level in language space, the correction videos, a 34% gain over a flat
  policy); image subgoals from an image-editing model (SuSIE, human videos) and the questions on them; a
  chain-of-thought version and RL fine-tuning as an open problem; hierarchical RL with state and language
  goals and hindsight relabeling; and discovering skills without supervision.
- [Lecture 16 — RL for Robots: Autonomous Learning](wiki/16-autonomous-rl.md): learning on real robots
  without a person resetting the scene. Why robots are not autonomous already (resets exist only in
  simulation; the basketball-court, mug and bookshelf examples; videos of people resetting robots); the
  definition of autonomous RL; deployed against continuing (average-reward) policy evaluation, with the Mars
  rover and the robot chef and the question on the expectation; SAC on the Fish task with longer and longer
  episodes; the two failures, drift and state distribution collapse; forward-backward RL, with roll-out
  lengths and the question on sharing data; three alternatives; MEDAL's backward reward learned from a few
  demonstrations, and the questions on it; the EARL results (tabletop and door closing), forward and backward
  tasks of different difficulty, and how demonstrations are collected; time-lapses on real robots with rewards
  from models; practising a cycle of tasks (the coffee graph, the algorithm, the dexterous hand, the legged
  robot, the wrench task); choosing tasks with a vision-language model (SOAR) and how a task proposer differs
  from a high-level policy; repeatable tasks (napkin folding); single-life RL (the key-in-the-lock video, the
  three settings, the walker that gets stuck), guiding towards familiar states and adapting in skill space
  (RMA, a VLM-guided legged robot); and the questions on safety and common sense.
- [Lecture 17 — Advancing Robot Intelligence with Reinforcement Learning](wiki/17-sim-to-real.md): a guest
  lecture by Ashish Kumar on sim-to-real transfer. Imitation and RL that work (off-policy data and generalists
  against on-policy data and specialists); what RL has achieved and its two ingredients, well-specified rewards
  and scalable evaluation; simulation for robots; the real-world videos of one blind walking policy; Rapid
  Motor Adaptation (the base policy and extrinsics trained with PPO, the ten-term reward function, the
  adaptation module trained with DAgger from 50 steps of history, 10 Hz and 100 Hz at test time); the indoor
  tests (oil, payloads, planks), switching adaptation off, the gait, torque and extrinsics plots, and the
  comparison table against domain randomization, system identification and fine-tuning; seven student
  questions (real-world improvement, training phase 2 directly, history and simulator exploits, the size of
  the extrinsics, balancing rewards, the previous action, DAgger); vision without terrain maps (two phases,
  scandots, egocentric depth, RNNs), emergent hip abduction and footstep planning on stairs, live demos and
  the comparison with blind and map-based policies; in-hand rotation with contact detection, and drones of
  different sizes; and what is left: humanoids at Tesla, why manipulation lags (simulation and rewards), the
  bitter lesson, a general reward model, and exceeding human capability.
- [Lecture 18 — Frontiers and Open Problems in Deep RL, and How to Do Research](wiki/18-frontiers.md): the
  course's last lecture, by Finn, in two parts. Open problems: domains without rewards (what human preferences
  reward, personalization and polarization, scoring robot shirt folds, hand-tuned recommendation weights,
  scientific reasoning, optimizing for learning); prior data and pre-trained weights that constrain learning
  (RLPD's buffer-only initialization); video generation models as world models and out-of-distribution
  actions; scaling RL with accurate value functions and batch online RL, where diffusion policies matter, and
  the question on DPO; safety without unsafe data, and exploring safely; hallucinations, the human-AI
  interface (the doctors study), RLHF's loss of calibration and the question on why, and reliability
  (HIL-SERL); and evaluating generalist systems with no offline metric (the rolled-back GPT-4o update). How to
  do research: Finn's backstory and the realities (few ideas last, research is incremental, simple ideas
  scale); what to work on (an important problem with a plan, excitement, brutal honesty; idea- against
  problem-driven research; bottlenecks; the video-prediction internship story; not boxing yourself in, not
  being a perfectionist); handling risk (front-loading core unknowns, with the sinusoid and gripper examples);
  getting things to work; deciding when to pivot, and the question on speed against evaluation; sharing
  research and writer's block; mentorship and confidence; and why now is a good time to do research.

- [Lecture 19 — Tutorial Session: Review of Q-Learning](wiki/19-q-learning-tutorial.md): the Q-learning
  TA session, by Anikait Singh. The MDP and the value, Q and advantage functions; the Bellman equation and
  Q-value iteration run iteration by iteration on a grid world, with the students' questions on the
  difference between the two and on initializing at zero, and the dynamic-programming heuristic; the optimal
  value, Q-value, policy and advantage of the grid world, why learn more than a policy, and a suboptimal
  policy's Q-function; value and Q networks for discrete and continuous actions; Monte Carlo, TD and n-step
  targets derived from the value function's definition on the board, the backup diagrams and the
  bias-variance table; stitching; semi-gradients, target networks and hard or Polyak updates, and the TD loss;
  gradient clipping and the Huber loss; replay buffers; the overestimation argument with zero-mean noise; and
  critic ensembles and double Q-learning. From the slides only: the DQN and soft actor-critic pseudocode of
  the Homework 2 walkthrough.

## Topic pages

- [MDPs and RL notation](wiki/mdp-and-notation.md): a table of the course's symbols (state,
  observation, action, trajectory, reward, initial state distribution, dynamics, policy), what
  each means in the towel and chatbot examples, the Markov property and why observations break
  it (and why a fixed horizon needs the time remaining in the state), when to treat something as
  a state or an observation, and the extra notation for demonstrations, for policy gradients, for
  actor-critic methods, for off-policy actor-critic (surrogate objectives, clipping, GAE), for
  Q-learning (optimal policy, target networks, double Q-learning, exploration) and for offline RL
  (behavior policy, filtered datasets, the expectile loss, CQL) and for reward learning (goal sets,
  learned rewards, preference pairs, the sigmoid, prompts and responses) and for preference
  optimization (prompts and completions, the reward model, the pretrained and RL models, winning and
  losing completions, the KL weight and the partition function), and for RL for LLM reasoning (the
  reasoning MDP, rollout policies, the Q-value and advantage of a step, the compute budget), and for
  model-based RL (dynamics models, planning horizons, sampled action sequences, model-based policy
  optimization), and for multi-task RL (tasks as MDPs, task identifiers, the original state with a
  bar, goal-conditioned rewards, relabeled data), and for meta-RL (exploration and task policies,
  training sets of experience, inferred task variables), with Lecture 13's POMDP in which only the task is
  hidden, and for exploration (bandits, regret, UCB and posterior sampling, task identifiers, bottlenecked
  task representations and exploration rewards, Lecture 14), and for hierarchy (high-level and low-level
  policies, goals, the re-query interval, Lecture 15), and for autonomous RL (state distributions, forward
  and backward policies and rewards, proposed tasks, the lifetime, Lecture 16), and for sim-to-real (state,
  previous action, extrinsics, scandots, depth and the vision policy's estimates, Lecture 17); and the Q-learning tutorial's plain notation (Lecture 19).
- [Policies](wiki/policies.md): the policy $\pi_\theta(\mathbf{a} \mid \mathbf{s})$ as a neural
  network, the roll-out loop and the words "roll-out" and "episode", policies with memory for
  observations, the two reasons policies are stochastic, and Lecture 6's deterministic greedy
  policy with the epsilon-greedy and Boltzmann exploration policies built around it, and Lecture 11's
  planner as a "policy" in quotes, and Lecture 12's task-conditioned policies and their architectures,
  and Lecture 13's sequence-model policies that learn in context, with memory across episodes, and Lecture
  15's two-level hierarchical policies, and Lecture 17's policy that adapts from its recent history of states
  and actions, with when a longer history lets it exploit the simulator; and Lecture 19's grid-world policy as a map of actions, the deterministic optimal policy and a stochastic suboptimal one.
- [The RL objective](wiki/rl-objective.md): from the sum of rewards to the expected sum, the two
  sources of randomness, the trajectory distribution $p_\theta(\tau)$, why the policy enters only
  through that distribution, the horizon $T$, the discount factor (and Lecture 4's reading of it as
  a probability of the agent dying), how Lecture 3 estimates the objective from samples and
  differentiates it, Lecture 8's question of where the reward comes from, and Lecture 9's
  KL-penalized objective for language models and its closed-form optimum, and Lecture 16's deployed and
  continuing (average-reward) evaluation of an agent that never resets.
- [Value functions](wiki/value-functions.md): the value function $V^\pi(\mathbf{s})$, the
  Q-function $Q^\pi(\mathbf{s}, \mathbf{a})$ and the advantage $A^\pi(\mathbf{s}, \mathbf{a})$,
  how the three relate (V as the expectation of Q; Q as the reward plus the next state's value, and
  Lecture 5's recursive form in terms of the next Q-value), why the advantage can be estimated from V
  alone, why a policy comes out of Q but not out of V, the drum-practice and Go examples, and Lecture
  7's offline use of the behavior policy's advantage, and Lecture 10's Q-value and advantage of a
  step in a language model's solution, and Lecture 11's terminal value function for planning, and Lecture 18's open question of value functions accurate enough to optimize policies at scale; and Lecture 19's grid world, with its optimal value, Q-function, policy and advantage and a suboptimal policy's Q-function.
- [Actor-critic methods](wiki/actor-critic.md): the method as a whole: from the policy gradient's
  reward to go to the advantage, the critic as a learned value function, the five-step algorithm and
  the names actor and critic, and the off-policy versions Lecture 5 teaches (multiple gradient steps
  leading to PPO, replay buffers leading to soft actor-critic, and the comparison of the two),
  how dropping the actor leads to Q-learning, and why the off-policy version breaks on a fixed
  offline dataset; and Lecture 19's critic of the current policy for one step of improvement, and the homework's soft actor-critic pseudocode.
- [Proximal policy optimization (PPO)](wiki/ppo.md): why many gradient steps on one batch overfit
  stale advantages, the surrogate objective, the KL-penalty and clipping remedies, the clipped and
  minimum objective as a lower bound, generalized advantage estimation, the four-step algorithm
  and example hyperparameters, and where PPO is the right trade (simulation, language models, and
  Lecture 6's advice), and its use for the RL step of RL from human feedback (Lectures 8 and 9),
  with the complexity of a PPO-based RLHF pipeline that motivates DPO; and GRPO, which Lecture 10
  describes as PPO with rollout-based advantages; and Lecture 17's walking policies trained with PPO in simulation; and Lecture 18's note that PPO's value functions only reduce gradient variance, and GPT-4's loss of calibration after PPO post-training.
- [Replay buffers and soft actor-critic](wiki/replay-buffers.md): reusing all past data, why a
  value function fitted on a buffer belongs to a mixture of past policies, the recursive Q-function
  and targets with actions from the current policy, action coverage, the fixed policy update, the
  remaining state-distribution problem, soft actor-critic, and its efficiency against PPO's
  stability; and the buffer in Q-learning, with how it is managed (first in, first out,
  pre-populating, asynchronous collection); and what goes wrong when the buffer is a fixed offline
  dataset; and Lecture 12's per-task buffers and buffers holding relabeled data; and Lecture 13's buffers of
  whole trajectories per task for off-policy meta-RL; and Lecture 16's SAC without resets and the buffer's state distribution collapsing around the goal; and Lecture 18's prior data loaded into the buffer without pre-trained weights; and Lecture 19's reasons for storing transitions (temporal correlations, recency bias), the buffer's maximum size, and the homework's two-critic soft actor-critic.
- [Policy evaluation](wiki/policy-evaluation.md): fitting a value network by Monte Carlo
  estimation, bootstrapping (temporal difference learning) or n-step returns; the shared
  regression; the worked example where they disagree; bias versus variance; discounted targets;
  Lecture 5's generalized advantage estimation and Q-function fitting from a replay buffer; and
  Lecture 6's policy iteration and n-step Q-learning targets; and Lecture 7's offline versions
  (Monte Carlo for the behavior policy, bootstrapping on data actions, expectile regression); and Lecture 19's derivation of the Monte Carlo, TD and n-step targets from the value function's definition, with the backup diagrams and the bias-variance table.
- [Imitation learning](wiki/imitation-learning.md): the problem (demonstrations from an unknown
  expert policy), behavior cloning versions 0 and 1, compounding errors and covariate shift,
  corrective data with DAgger and human-gated DAgger, a table of their trade-offs, how
  demonstrations are collected, how the policy gradient relates to it (an imitation gradient
  weighted by reward; imitation learning as initialization), imitation as an offline RL baseline
  (filtered behavior cloning, advantage weighting, why it cannot stitch), Lecture 8's case for
  learning a reward instead, and instruction finetuning as the supervised stage of training a chat
  model (Lecture 9), and rejection finetuning, spurious steps and causal confusion (Lecture 10), and
  distilling a planner's actions into a policy (Lecture 11), and multi-task imitation learning with
  stratified sampling (Lecture 12), and hierarchical imitation learning with DAgger on a language-space
  high level (Lecture 15), and demonstrations as the target of MEDAL's backward policy, with how they are
  collected (Lecture 16), and DAgger distilling a privileged teacher in simulation into a deployable student
  (Lecture 17).
- [Expressive policy distributions](wiki/expressive-policy-distributions.md): why a policy needs
  more than a mean; categorical, Gaussian, Gaussian-mixture, discretized autoregressive and
  diffusion policies, what each outputs and how each is trained; why a bigger network does not
  help; the evidence for expressive policies; IDQL's diffusion policies for offline RL, and Lecture 18's diffusion policies in batch online RL.
- [Offline and online algorithms](wiki/offline-and-online.md): the offline/online definitions
  Lecture 2 introduces, why online data can be expensive or unsafe, which imitation methods are
  which, Lecture 3's on-policy/off-policy distinction among online algorithms, and Lecture 5's
  spectrum of increasingly off-policy actor-critic methods, from PPO to replay buffers, and why
  Q-learning is off-policy while its n-step targets are not; and Lecture 7's offline RL, why it is
  worth it and why off-policy methods fail on fixed data; and Lecture 9's contrast between RLHF's
  online sampling and DPO's fixed preference data; and Lecture 10's imitation, offline and online RL
  for reasoning, compared by data efficiency; and Lecture 11's map of every algorithm so far, and why
  planning with a model is off-policy; and Lecture 12's offline and online versions of model-based policy
  optimization, and why relabeled data is off-policy; and Lecture 13's off-policy against on-policy
  meta-training; and Lecture 17's contrast between off-policy imitation and on-policy RL; and Lecture 18's batch online RL and prior data loaded into the buffer rather than the weights.
- [Policy gradients](wiki/policy-gradients.md): the method as a whole, from the online loop and
  the derivation to REINFORCE, the reading as reward-weighted imitation, its three weaknesses
  (high variance, sensitivity to reward scale, need for dense rewards and large batches) and the
  two fixes (causality and baselines), the surrogate objective, the off-policy version, what
  actor-critic methods change, and Lecture 9's recap for language models (why RL is needed even with a
  differentiable reward model), and Lecture 10's online RL for reasoning with a 0/1 reward.
- [Importance sampling](wiki/importance-sampling.md): estimating an expectation under one
  distribution with samples from a proposal distribution, its catches (support, similarity), and
  how it makes the policy gradient off-policy, including why per-trajectory ratios explode, and
  its use in off-policy actor-critic: the surrogate objective, why the ratio invites overfitting,
  and PPO's clipping; and Lecture 6's mention of it as a fix for off-policy n-step targets; and Lecture
  12's remark on importance weights for relabeled multi-task data.
- [RL algorithm families](wiki/rl-algorithm-families.md): imitation learning, policy gradients,
  actor-critic (with PPO and SAC), value-based (Q-learning, DQN) and model-based methods, which
  lecture of the schedule covers each, how the first four relate, Finn's advice on when to use PPO,
  DQN or SAC and on which offline RL algorithm to use, and the trade-offs (data cost, supervision, stability, action space, dynamics) that
  explain why RL has so many algorithms, with pointers to Lectures 9 and 10's RL for language models,
  and Lecture 11's map of the families so far and its answer to when to use a model, and Lecture 12's
  model-based policy optimization, and Lecture 13's meta-RL, which inherits its sample efficiency from
  its RL optimizer, and Lecture 14's exploration, and Lecture 15's hierarchy, and Lecture 16's learning
  without resets, and Lecture 17's sim-to-real transfer, and Lecture 18's open problems across the families.
- [Q-learning](wiki/q-learning.md): the value-based method as a whole: the greedy policy a
  Q-function defines and why it is at least as good, policy iteration and backups, the max target,
  convergence, epsilon-greedy and Boltzmann exploration, the full algorithm with a replay buffer,
  the three tricks (target networks and DQN, double Q-learning against overestimation, n-step
  returns and their on-policy catch), continuous actions, when to use it, why its coverage
  fails offline, and the conservative Q-learning remedy, with Lecture 14's bandit game on why
  epsilon-greedy's epsilon is lowered; and Lecture 19's tutorial: Q-value iteration on a grid world, semi-gradients and hard or Polyak target updates, gradient clipping and the Huber loss, the overestimation argument and critic ensembles.
- [Bellman equations](wiki/bellman-equations.md): the Bellman equation for any policy's Q-function
  and how it lets a Q-function be fitted from other policies' data, the Bellman optimality equation
  for the optimal policy and how Q-learning uses it, the names, the value-function versions, why
  the course builds on Q, and Lecture 7's offline target for the behavior policy; and Lecture 19's Q-value iteration, the Bellman equation iterated on a grid world, with the dynamic-programming heuristic.
- [Offline reinforcement learning](wiki/offline-rl.md): learning from a fixed dataset with no further
  interaction; the behavior policy and distribution shift; why off-policy methods overestimate Q-values
  on out-of-distribution actions; why imitation is not enough (rewards, trajectory stitching); the three
  approaches (filtered or weighted imitation, implicit policy constraints, conservative objectives);
  conservative Q-learning (CQL) in brief, with its LinkedIn application; which algorithm to use;
  and Lecture 10's offline RL for language reasoning; and Lecture 18's open question of offline metrics for evaluating policies; and Lecture 19's stitching example and larger critic ensembles (SAC-N, EDAC) offline.
- [Implicit policy constraint methods](wiki/implicit-policy-constraints.md): filtered behavior cloning,
  advantage-weighted imitation and its KL-constrained reading, advantage-weighted regression (AWR),
  advantage-weighted actor-critic (AWAC), expectile regression and implicit Q-learning (IQL), each
  fixing the previous one's weakness, with the deterministic-data special case, the choice of
  the expectile, and Lecture 8's explanation of why only the value function gets the asymmetric loss;
  and Lecture 10's filtering of reasoning steps by advantage.
- [Conservative Q-learning (CQL)](wiki/conservative-q-learning.md): the offline RL method that
  penalizes Q-values, as Lecture 8 teaches it from Lecture 7's slides: pushing down on large Q-values
  with a search distribution mu, pushing back up on the data's, the two lower bounds, what alpha and mu
  do, the entropy regularizer and log-sum-exp that remove mu, the algorithm for discrete and continuous
  actions, why be conservative rather than clone, the LinkedIn application, and the parallel with
  reward classifiers.
- [Reward learning](wiki/reward-learning.md): why rewards cannot be taken for granted (games against
  pouring water, dialog and driving; proxies), why imitation is not enough, goal classifiers, how RL
  exploits a fixed classifier, adding visited states as negatives with regularization and 50/50
  balancing (and why mislabelled successes are tolerable), the GAN connection and full reward
  functions, the self-improving robot, the trade-offs, other forms of feedback, agents that set
  their own goals, Lecture 9's over-optimized reward models, and Lecture 10's process reward models
  and verifiers, and Lecture 11's reward model for planning, and Lecture 12's rewards for synthetic and
  relabeled data, and Lecture 16's discriminator reward for returning to demonstrated states and rewards from
  models for learning without resets, and Lecture 17's hand-designed ten-term locomotion reward, why
  manipulation rewards do not scale, and the hope of a general reward model, and Lecture 18's domains with no reward to verify (chatbot preferences, robot shirt folds, hand-tuned recommendation weights, scientific reasoning, optimizing for learning).
- [Learning rewards from preferences](wiki/preference-learning.md): why comparing trajectories is
  easier than scoring them, the reward model and its sigmoid log-likelihood objective, partial
  roll-outs and the missing margin, the complete algorithm over ranked batches and in the loop of RL,
  the backflip and driving examples, RL from human feedback for language models (reward model,
  three-stage pipeline, PPO), AI feedback (RLAIF, Constitutional AI), the trade-offs, and how Lecture 9
  names the objective Bradley-Terry and extends it to rankings (Plackett-Luce), and Lecture 18's evidence that people prefer confident answers that agree with them over true ones.
- [Instruction finetuning](wiki/instruction-finetuning.md): why a pretrained language model is not
  an assistant, finetuning on (instruction, output) pairs across many tasks, Flan-T5's result that
  bigger models gain more, format versus knowledge, the limitations that motivate learning from
  preferences, and its place in InstructGPT and ChatGPT (Lecture 9), and Lecture 12's reading of
  supervised fine-tuning as multi-task imitation learning with prompts as tasks.
- [RL from human feedback (RLHF)](wiki/rlhf.md): optimizing a language model for human preferences,
  the three-step pipeline, the Bradley-Terry reward model and why comparisons beat scores, reward-model
  accuracy, the RL step and why it is RL (discrete tokens), the KL penalty, what it achieves, and its
  costs and limitations (complexity, reward hacking and over-optimization, sycophancy, hallucination,
  data) (Lectures 8 and 9), and Lecture 17's use of over-long RLHF tuning as a warning for general reward models, and Lecture 18's open problems: what human preferences reward, personalization against polarization, RLHF's loss of calibration and why, and the rolled-back GPT-4o update.
- [Direct preference optimization (DPO)](wiki/direct-preference-optimization.md): training a language
  model on preferences with a classification loss, its notation, the five-step derivation from the
  KL-constrained objective's closed-form optimum (with the partition function cancelling), what the
  loss does and the role of beta, DPO against RLHF, and its use in open models (Lecture 9); and
  per-step preference pairs built from rollouts for reasoning (Lecture 10).
- [RL for LLM reasoning](wiki/llm-reasoning.md): training language models to solve math problems
  with RL (Lecture 10): why imitating reference solutions scales slowly, reasoning as a sparse-reward
  MDP with deterministic dynamics, the three data axes, rejection finetuning and why it stops
  working (spurious steps), the three uses of per-step advantages (filtering, per-step DPO, process
  advantage verifiers as a dense reward), online RL with an outcome reward and GRPO, and how
  thinking models such as DeepSeek-R1 are trained, with test-time scaling from the slides, and Lecture
  13's view of optimizing test-time compute as a meta-RL problem, and Lecture 17's account of reasoning as one
  of RL's successes, resting on a rule-based reward, and Lecture 18's contrast between verifiable math and scientific reasoning, and its very online data.
- [Process rewards and per-step credit](wiki/process-rewards.md): the Q-value of a step of a
  solution under a rollout policy, estimated by rollouts from each prefix; the advantage as one
  Q-value minus the previous one, with a worked example; choosing the rollout policy (a different
  policy, quality against diversity); rollouts against a learned model (process reward models,
  process advantage verifiers); its three uses; and its relation to advantage-weighted regression,
  stitching and RLHF's reward models (Lecture 10).
- [Model-based RL](wiki/model-based-rl.md): the fifth algorithm family as a whole: learning a
  simulator and what goes wrong (coverage, domain dependence, model errors), learning a dynamics
  model (known, partly known or unknown; image or latent space; a reward model; ensembles), the two
  ways to use one (planning, and model-based policy optimization's augmented roll-outs, with its
  algorithm and trade-offs as Lecture 12 teaches them), when a model is worth having (Lecture 12's maze
  and pouring examples, and robotics), other kinds of models, and how it relates to the other families (Lectures 11 and 12), and Lecture 18's video generation models as world models and why they mispredict unfamiliar actions.
- [Planning with a learned model](wiki/planning-with-models.md): the planning objective and the
  planner as a "policy"; planning by backpropagation, with the gradient-ascent update; random shooting
  and the cross-entropy method; gradients against samples; refitting the model on the planner's own
  data (the cliff); open loop, closed loop and model predictive control, with warm-starting,
  replanning frequency and latency; the limits of planning, distillation and a terminal value
  function; and the case study's modified CEM (Lecture 11).
- [Multi-task RL](wiki/multi-task-rl.md): one policy for many tasks: why (data efficiency, shared
  structure, generalists), a task as an MDP and what varies across tasks, task identifiers, the task as
  part of the state and goal-conditioned RL, multi-task imitation learning with stratified sampling,
  feeding the task to a network (BC-Z, OpenVLA, prompts), one policy against many, and conditioning the
  policy and Q-function in multi-task RL (Lecture 12), and how meta-RL differs, with experience as the
  task identifier and no aggregated MDP (Lecture 13), and DREAM's ID-conditioned execution policy,
  trained as multi-task RL (Lecture 14), and the goal-conditioned policy as the low level of a hierarchy
  (Lecture 15), and practising a cycle of tasks to learn without resets (Lecture 16).
- [Hindsight relabeling](wiki/hindsight-relabeling.md): sharing data across tasks by storing experience
  again under another task's identifier and reward (hindsight experience replay, HER): the hockey
  example, the algorithm, which task to relabel for, the conditions (same state and action spaces and
  dynamics, a computable reward, an off-policy learner) and importance weights, and goal-conditioned
  relabeling with the last state as goal, with the HER paper's results and Homework 4 (Lecture 12), and
  relabeling a high level's goal actions, or in language space, in hierarchical RL (Lecture 15).
- [Meta-RL](wiki/meta-rl.md): learning a new task from a few episodes by training across a task
  distribution (Lecture 13): the three kinds of transfer and few-shot learning, the key assumption, the
  formal setting with episodic and online variants, black-box meta-RL (the sequence-model policy, its
  four-step algorithm, architectures and why off-policy meta-training is faster), experience as a task
  identifier, meta-RL as a POMDP, the maze, test-time compute and continuous-control examples, and
  exploration: why end-to-end training is hard (the coupling problem) and the alternatives (posterior
  sampling, intrinsic rewards, task dynamics and reward prediction), and DREAM, which decouples
  exploration from execution with a bottlenecked task representation, with its theory and results
  (Lecture 14), and Lecture 17's RMA as adaptation in context.
- [Exploration](wiki/exploration.md): why exploration is hard (Montezuma's Revenge, Mao) and the
  exploration-exploitation trade-off; epsilon-greedy and Boltzmann exploration (Lecture 6); bandits,
  regret, UCB and posterior sampling, with the drug-trial game's comparison; why large MDPs explore from
  demonstrations, base models and shaped rewards instead of from scratch; and learning to explore in
  meta-RL, from the coupling problem to DREAM (Lectures 13–14); and, from Lecture 15, the structured
  exploration a high-level policy gives in goal space; and, from Lecture 16, exploration without resets, which can drift far from the good states; and, from Lecture 18, optimizing for learning and exploring safely as open problems.
- [Hierarchical RL and imitation](wiki/hierarchical-rl.md): why long-horizon tasks are hard; the two-level
  policy (a high-level policy predicting goals, a low-level policy pursuing them at a higher frequency) and
  how it is rolled out; why hierarchy may help, and the chain-of-thought alternative; goal representations;
  supervising each level and adapting the levels to each other; when to move on to the next goal; and the
  example systems (language and image subgoals, DAgger in language space, hierarchical RL with relabeling,
  skill discovery) (Lecture 15), and how a task proposer for practice differs from a high-level policy
  (Lecture 16).
- [Autonomous and single-life RL](wiki/autonomous-rl.md): learning without resets (Lecture 16). The reset
  problem on real robots; the two ways to evaluate (deployed policy evaluation and continuing, average-reward
  evaluation); why standard RL fails with long episodes (drift, state distribution collapse);
  forward-backward RL; MEDAL's learned reward for returning to the demonstrations' states and its EARL
  results; practising a cycle of tasks with a task-conditioned policy, and choosing tasks; and single-life RL,
  adapting within one episode by steering towards familiar states and adapting in skill space.
- [Sim-to-real transfer](wiki/sim-to-real.md): training a robot's policy in simulation and deploying it on
  the real robot (Lectures 1, 5, 16 and 17). Why simulation supplies RL's two ingredients; domain
  randomization and telling the policy the parameters (extrinsics); inferring them online from history with an
  adaptation module trained by DAgger, as in-context adaptation; the comparison with system identification and
  real-world fine-tuning; vision without terrain maps, through a privileged teacher and a depth student; what a
  policy can exploit in simulation; hands and drones; and why manipulation still falls short.

## Course pages

- [Course logistics](wiki/course-logistics.md): staff and guest lecturers, prerequisites, the four
  homeworks and their topics, grading, the default and custom final projects, late days, the AI
  tools policy, and the first week's steps, from Lecture 1's slides and the course site, with
  Lecture 11's, 12's and 13's Homework 3 and milestone dates, hindsight relabeling in Homework 4,
  Lecture 13's evening office hours and meta-RL in Homework 4, and Lecture 14's reminders (Homework 3
  due, Homework 4 out) and DREAM in Homework 4, and Lecture 15's reminders (milestone due Friday, Homework 4
  due next Friday, tips on Ed for the default project), and Lecture 16's reminders (milestone due that night,
  Homework 4 due next Friday, a guest lecture from Ashish Kumar next), and Lecture 17, that guest lecture, and Lecture 18's reminders (Homework 4 due that day, the poster session next Wednesday, the final report the following Monday with no late days) and advice on doing research, and Lecture 19, the TA section on Q-learning and its preview of the Homework 2 algorithms.

## Raw materials

- [`raw/transcripts/`](raw/transcripts/) — copy-edited lecture transcripts with `[MM:SS]`
  paragraph marks. The verbatim captions are not committed; each transcript says how to
  regenerate them.
- [`raw/slides/`](raw/slides/) — full text of every slide, numbered, with figures described.
- [`raw/images/`](raw/images/) — rendered figure slides, linked from the pages that use them.
- [`sources.md`](sources.md) — every course document by lecture, with canonical URLs: slide
  decks, homeworks, project guidelines and the site's optional readings.
- [`LICENSE.md`](LICENSE.md) — what may be reused, and the attribution of every image.
- [`SEE_ALSO.md`](SEE_ALSO.md) — related knowledge bases (CS224N, whose post-training lecture is an
  earlier version of Lecture 9's deck; CS336, whose post-training lectures cover RLHF and RL for
  reasoning in practice; CS329A, on test-time compute, verifiers and RL for reasoning models; and
  Princeton ROB 345, on trajectory optimization and model predictive control with known dynamics),
  with the repo URL to pass as `kb`.

Slide and handout PDFs are **not** committed to this repo; [`sources.md`](sources.md) has the
canonical URL of each.
