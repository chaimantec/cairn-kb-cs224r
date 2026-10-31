# Autonomous and single-life RL

Reinforcement learning practises by trial and error, but almost every algorithm in the course assumes that
after each attempt the agent is returned to its initial state distribution. In simulation that is
`env.reset()`; on a real robot it is a person resetting the scene, thousands of times. **Autonomous RL**,
also called reset-free or non-episodic RL, removes that assumption: the agent is initialized once and then
learns with no resets, or only very rare ones. [Lecture 16](16-autonomous-rl.md) teaches it as the first of
two lectures on RL for robots: why robots are not autonomous already, how to define and evaluate autonomous
RL, algorithms that learn without human intervention, and **single-life RL**, where a trained agent must
adapt within one episode of deployment. [Lecture 1](01-class-intro.md) lists reset-free RL among the course's
topics, with a video of a person resetting a hockey-puck task after every attempt (Lecture 1, slide 30).

Notation, as on the slides: $\rho_0$ is the initial state distribution, $\rho_g$ a region of goal states with
high reward, $\rho^\ast$ the distribution of states visited by the optimal policy on its way from $\rho_0$ to
$\rho_g$, and $H$ the number of steps in an episode or roll-out.

## The reset problem

Rolling out a policy is "fully autonomous" (Lecture 16, ≈3:07): the agent samples its own actions. The gap
is between attempts. After a failed attempt the agent is somewhere else, and to try again a human typically
"is actually physically moving the robot or physically moving the objects in the scene back to some set of
initial conditions" (≈3:54). In simulation the reset is a function call that exists "Only in simulation!"
(Lecture 16, slide 6). The real world is continual (slide 7): a robot that makes a wrong turn on the way to
the basketball court is really somewhere else, and a mug knocked over has to be put back within reach. And RL
needs many attempts, "Several thousands of trials!" (slide 7), each needing an intervention. Slide 8's videos
show people resetting a hockey-puck task, closing doors for a robot learning to open them, and resetting a
towel for a robot learning to fold it (≈6:58–7:44).

The aim (Lecture 16, slide 11) is to initialize the scene once at the beginning and let the agent interact
autonomously with no environment resets; the slide's footnote adds that one "can relax this constraint to
reset occasionally, or at low frequency". Why it matters (slide 14): we want robots to operate,
and to *train*, with minimal human supervision. RL on real robots is a promising path to high success rates
and reliability, and autonomy unlocks more data, which may unlock greater generalization (≈15:29–16:15).

## Two ways to evaluate

An autonomous agent can be judged in two ways (Lecture 16, slides 12–13, ≈9:16–14:40).

**Deployed policy evaluation** measures the quality of the policy learned: after learning, start it from
the initial state distribution and measure its return. With $p(\mathbf{s}_ 1)$ the initial state distribution,
$\gamma$ the discount and $r$ the reward, it is the course's usual [RL objective](rl-objective.md):

$$J(\pi) = \mathbb{E}_ {\mathbf{s}_ 1 \sim p(\mathbf{s}_ 1), \mathbf{a}_ t \sim \pi} \left[ \sum_ t \gamma^t r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

This is what matters when the point is a good policy: a robot chef learning to cook is judged on the meals it
cooks for customers afterwards, not during training.

**Continuing policy evaluation**, or average-reward RL, measures the reward accumulated over the agent's
lifetime $h$, in the limit as the lifetime grows:

$$\lim_{h \to \infty} \mathbb{E} \left[ \frac{1}{h} \sum_{t=0}^{h} r(s_t, a_t) \right]$$

This is what matters for a Mars rover collecting scientific data, which is judged on how well it does while it
is deployed and learning. Its expectation "isn't under a single policy. It's over the learning process
itself" (≈14:40), and it averages over runs of the algorithm, since the environment and the learning process
(how batches are sampled, for example) both vary (≈14:40–15:29).

## Why standard RL fails without resets

Making episodes longer should mean fewer resets and less supervision. But soft actor-critic on a simple fish
navigation task does worse with every increase in episode length: from 1000 steps to 100,000, the deployed
policy's reward after 5 million steps falls from about 490 to about 130 (Lecture 16, slide 16,
≈17:00–18:32). Slide 17 gives two reasons:

- **Drift.** Exploration can take the agent far from the good states, $\rho^\ast$, into parts of the state
  space that are very hard to recover from (≈18:32–20:05).
- **State distribution collapse.** If the agent does reach the goal, it may stay there, so that the replay
  buffer's state distribution is "concentrated around the goal rather than capturing the state distribution
  that's around the optimal policy" (≈21:38). With no data from the states between start and goal, it never
  learns what to do there.

Either way, in the slide's words, "The agent never learns a good policy". In episodic learning the agent
"Can always retry the task from initial state distribution", which is exactly what is missing.

![Lecture 16, slide 17 — The Challenge of Learning Autonomously](../raw/images/16-autonomous-rl/slide-17.jpg)

*Lecture 16, slide 17 — episodic learning (left) can always retry from the start; non-episodic learning (right) drifts away from the good band (challenge 1) or loops around the goal so its data collapses there (challenge 2), and never learns a good policy. [Deck](https://cs224r.stanford.edu/spring_2025/slides/16_cs224r_autonomy_2025.pdf)*

## Learning to reset: forward-backward RL

Since the agent learns well when it can retry from the initial state distribution, learn a policy that takes
it back there (Lecture 16, slides 18–19, ≈21:38–27:51). **Forward-backward RL** learns a forward policy
$\pi_f$ for the task, with the task's reward $r_f(\mathbf{s}, \mathbf{a})$, and a backward policy $\pi_b$
that undoes it, with a reward $r_b(\mathbf{s}, \mathbf{a})$ for reaching $\rho_0$. It alternates: roll out
$\pi_f$ for $H$ steps and update it with $r_f$, then roll out $\pi_b$ for $H$ steps and update it with
$r_b$, with no reset or a very infrequent one. Any RL algorithm can do the updates. At test time $\pi_b$ is
discarded and $\pi_f$ deployed. The method is simple, but the backward policy is "ultimately discarded"
(slide 19), which seems wasteful.

![Lecture 16, slide 19 — Algorithm: Forward-Backward RL](../raw/images/16-autonomous-rl/slide-19.jpg)

*Lecture 16, slide 19 — forward-backward RL: a forward policy (pink) heads for the goal, a backward policy (blue) rewarded for reaching the initial states brings the agent back, alternating with no resets; at test time only the forward policy is kept. [Deck](https://cs224r.stanford.edu/spring_2025/slides/16_cs224r_autonomy_2025.pdf)*

Practical points from the lecture: a fixed roll-out length is simplest, though early on, before the forward
policy moves anywhere, the backward policy has nothing to do (≈25:33–26:20); the two policies could share
replay-buffer data with relabeled rewards, or an image encoder, but in practice often share little, since they
learn very different things (≈26:20–27:05); the forward and backward tasks can differ a lot in difficulty, as
closing a door against opening it, which makes them learn at different rates (≈38:37–39:23); and when neither
policy is good yet, demonstrations of both tasks can bootstrap them (≈40:55–41:42).

## Resetting to the expert's states: MEDAL

The backward policy need not go all the way back to the start. **MEDAL** (Matching Expert Distributions for
Autonomous Learning; Lecture 16, slides 21–23, ≈28:39–34:46) is forward-backward RL with two changes. It is
given a small set of demonstrations, and its backward reward rewards reaching the states those demonstrations
visit, a "learned reward for matching state distribution in demos" (slide 22). The reward is a discriminator
trained to tell the policy's states from the demonstrations' states, as in
[reward learning](reward-learning.md), and the backward policy tries to fool it. Because the discriminator sees
every state of the demonstrations, a backward policy that only returned to the goal would be recognized as the
policy and not the expert: formally, the objective is a divergence between the backward policy's state
distribution and the expert's (≈34:01–34:46).

This addresses both challenges at once (slide 23): the agent does not drift away, and the backward policy
avoids collapse of the state distribution. The forward policy also practises from a wide set of starting
states, easy and hard, which improves sample efficiency even in episodic settings (slide 23 cites Kakade and
Langford, 2002). The demonstrations can also be added to the replay buffers (≈33:15). On the EARL benchmark,
where an agent is reset only every 200,000 steps, the reset-free methods all far outperform SAC with no
backward policy, and on the harder door-closing task forward-backward RL and MEDAL do best, MEDAL matching an
oracle with frequent resets (slides 24–25, ≈34:46–37:06). On real robots both have run with rewards from
models rather than humans: a vision-language model's for forward-backward RL, learned ones for MEDAL
(slide 26, ≈39:23–40:55).

## Practising a cycle of tasks

With several tasks, the agent can practise whichever task makes sense from where it is (Lecture 16,
slides 27–31, ≈41:42–47:55). A coffee-making robot that drops the cup should next practise picking it up,
which then acts as a reset for putting the cup in the machine (slide 28). The algorithm (slide 29) learns a
task-conditioned [multi-task policy](multi-task-rl.md) $\pi(\cdot \mid \mathbf{s}, \mathbf{z})$: repeatedly
propose a task $\mathbf{z}_ i$ based on the state $\mathbf{s}$, roll out
$\pi(\cdot \mid \mathbf{s}, \mathbf{z}_ i)$ for $H$ steps, and update with $r(\mathbf{s}, \mathbf{a}, \mathbf{z})$,
which needs a reward for every task. At test time the whole multi-task policy is deployed. Examples include a dexterous hand that
re-centres, lifts, flips and rotates an object, a quadruped walking forward and backward and getting up, and a
seven-stage wrench task (slides 30–31).

Which task to practise is "an open research question" (slide 32). The graph of tasks can be predefined, or a
model can propose tasks: SOAR asks a vision-language model what can be done in the current scene ("take the
banana out of the silver pot") and turns the proposal into a goal image to practise reaching (≈48:42–51:06).
Such a proposer differs from a [hierarchical](hierarchical-rl.md) high-level policy in purpose: it should
suggest many different things to practise, whereas a high-level policy is "much more goal-directed in terms
of accomplishing some sort of long-term objective" (≈50:19). A third option is a task that can simply be
repeated, such as folding napkin after napkin (slide 33).

## Single-life RL

Learning without resets improves a policy before deployment. **Single-life RL** (Lecture 16, slides 35–44,
≈53:27–1:02:00) asks what happens after: given prior training data, the agent has one "life" to complete a
task in a novel, out-of-distribution scenario, and must adapt "during a single episode without resets"
(≈55:45), as a person who fumbles a key in a lock adapts in a split second (slides 37–39). Slide 40 sets it
beside episodic RL (many episodes, then deploy) and reset-free RL (one long training run, then deploy).

![Lecture 16, slide 40 — Single-Life Reinforcement Learning (SLRL)](../raw/images/16-autonomous-rl/slide-40.jpg)

*Lecture 16, slide 40 — episodic RL trains in many episodes, reset-free RL in one long one, and single-life RL gets prior data and a single life in a new situation, here a walker trained on flat ground meeting a block. [Deck](https://cs224r.stanford.edu/spring_2025/slides/16_cs224r_autonomy_2025.pdf)*

Plain RL fine-tuning at test time, with TD updates or policy gradients, gets a walker over an unseen obstacle
but then leaves it stuck on its back, unable to recover without interventions (slide 42). Three ideas help
(slide 43): guide learning towards good, familiar states from the prior data rather than towards the goal,
which got the walker back on its feet and to the goal (≈58:06, ≈1:00:25); adapt in a space of pre-trained
skills rather than of actions, so the agent changes strategy rather than weights; and use pre-trained language
models' common sense to reason about how to adapt. Slide 44 contrasts adaptation in low-level action space
(RMA, which handles small changes such as new terrain) with adaptation in high-level skill space (a legged
robot that tries strategies such as moving left or backwards, with a vision-language model's guidance,
≈1:01:13–1:02:00, ≈1:04:20).

Safety is a related open problem (≈1:02:00–1:04:20): learned recovery policies that return to less risky
states, and constraints imposed on the controller itself, help, but learning safety needs examples of
failing. All of these methods are, in Finn's words, still in the research stages: "This is not something that
robot companies are using today." (Lecture 16, ≈41:42)

## Where this appears

- [Lecture 16 — RL for Robots: Autonomous Learning](16-autonomous-rl.md): the whole topic, with the videos,
  the EARL results and the questions.
- [Lecture 1 — Class Intro](01-class-intro.md): reset-free RL among the course's topics, and a person
  resetting a hockey-puck task (slide 30).
- [The RL objective](rl-objective.md#evaluating-an-agent-that-never-resets): deployed and continuing policy
  evaluation beside the episodic objective.
- [Reward learning](reward-learning.md#rewards-for-learning-without-resets): MEDAL's discriminator reward, and
  rewards from models so that learning runs unattended.
- [Replay buffers and soft actor-critic](replay-buffers.md#without-resets): SAC as the baseline, and the replay
  buffer whose state distribution collapses.
- [Multi-task RL](multi-task-rl.md#practising-tasks-without-resets): the task-conditioned policy that practises
  a cycle of tasks.
- [Hierarchical RL and imitation](hierarchical-rl.md#a-task-proposer-is-not-a-high-level-policy): how a task proposer differs from a
  high-level policy.
- [Imitation learning](imitation-learning.md#demonstrations-for-learning-without-resets): the demonstrations
  MEDAL matches.
- [MDPs and notation](mdp-and-notation.md#notation-for-autonomous-rl): the symbols used here.
