# Multi-task RL

Training one policy to do many tasks rather than one: a generalist LLM assistant, a legged robot that
walks, runs, dances and crouches, a recommender personalized to many users. [Lecture 12](12-multi-task-rl.md)
teaches it. This page collects what a task is, how a policy is told which task to do, multi-task
imitation learning and RL, and how the task is fed to a network in practice. Sharing data between tasks
by relabeling, and goal-conditioned RL with hindsight relabeling, have their own page,
[hindsight relabeling](hindsight-relabeling.md).

## Why train on many tasks

A topic recurring through the course is data efficiency: how much data a method needs (Lecture 12,
slide 12, ≈29:17–31:35). If the goal is a policy that does many tasks, training it across tasks can
amortize the cost of learning any one of them, because tasks typically share structure: grammar and
how websites work, for an LLM assistant; staying balanced, for a legged robot that walks forward and
backward; similarities between users, for a personalized recommender. Slide 12 adds "a deeper
motivation": "Generalist ML systems are often more reliable, performant than specialists", which Finn
calls a pretty cool finding that has empirically been true in recent years in AI. Lecture 1 lists
multi-task RL and meta-RL, which transfer from other tasks and goals, as the course's answer to how an
agent can generalize to many scenarios (Lecture 1, slide 29).

## A task is an MDP

What makes something a different task is not well defined. One way to make it precise is that
different tasks are different Markov decision processes (Lecture 12, slide 13, ≈31:35–32:21):

$$\mathcal{T}_ i \triangleq \lbrace \mathcal{S}_ i, \mathcal{A}_ i, p_i(\mathbf{s}_ 1), p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}), r_i(\mathbf{s}, \mathbf{a}) \rbrace$$

with task $i$'s own state space $\mathcal{S}_ i$, action space $\mathcal{A}_ i$, initial state distribution
$p_i(\mathbf{s}_ 1)$, dynamics $p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$ and reward
$r_i(\mathbf{s}, \mathbf{a})$. This "allows for much more variety than the semantic meaning of 'task'":
walking forward on grass and walking forward on flat terrain could be two tasks, with the same reward
and different dynamics (≈32:21–33:07). See [MDPs and notation](mdp-and-notation.md).

Which pieces vary depends on the problem (slide 14, ≈33:07–37:01). Across users of a recommender, the
reward (one person loves rock, another hip-hop) and the dynamics (how often each listens) vary. In
character animation, a backflip, a run and a cartwheel vary the reward while the simulator's physics
stays the same; putting on different garments varies the dynamics and the initial state distribution.
Controlling robot arms with four, six or seven degrees of freedom varies the state and action spaces,
their dimensions included, and the dynamics.

![Lecture 12, slide 14 — A Few Examples Task Distributions](../raw/images/12-multi-task-rl/slide-14.jpg)

*Lecture 12, slide 14 — three kinds of task distribution: character animation varies the reward across maneuvers, or the initial state and dynamics across garments, and multi-robot RL varies everything but the reward. [Deck](https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf)*

## Task identifiers

The policy is told which task to do by a **task identifier** $\mathbf{z}_ i$: a task index (task 0, task
1, …), what one resorts to without a better way to identify a task; a language description; or a video
illustration of what to do (Lecture 12, slide 15, ≈37:01–37:46). A one-hot task ID is fairly common in
RL methods, but there are better ways, such as a language description of a user's preferences or, as
next week's material covers, examples of what they like and do not like, which comes closer to
in-context learning (≈43:08–44:40).

## The task as part of the state

An alternative view puts the identifier into the state, $\mathbf{s} = (\bar{\mathbf{s}}, \mathbf{z}_ i)$,
where $\bar{\mathbf{s}}$ is the original state (slide 16). The collection of task MDPs then becomes one
MDP covering all the tasks, whose reward and dynamics are the task's own: reconstructed from Finn's spoken
board work, $r(\mathbf{s}, \mathbf{a}) = r_i(\bar{\mathbf{s}}, \mathbf{a})$ and
$p(\mathbf{s}' \mid \bar{\mathbf{s}}, \mathbf{z}_ i, \mathbf{a}) = p_i(\mathbf{s}' \mid \bar{\mathbf{s}}, \mathbf{a})$
(≈38:31–40:48). Tasks with different state or action spaces make this more complicated, since one then
has to decide how to represent the combined state.

So multi-task RL is "the same as before, except: a task identifier is part of the state" (slide 17,
≈40:48–43:08). When the identifier is a desired goal state, $\mathbf{z}_ i = \mathbf{s}_ g$, this is
**goal-conditioned RL**, and the reward is often a negative distance to the goal,
$r(\bar{\mathbf{s}}, \mathbf{s}_ g) = -d(\bar{\mathbf{s}}, \mathbf{s}_ g)$, with $d$ for example Euclidean
$\ell_2$ or a sparse 0/1. Since the result is still a standard MDP, standard RL algorithms can be
applied to it, and in many cases that is a great approach; but "You can do better in some cases", by
using the known structure between the tasks, and goal-conditioned RL is one of them (≈42:22–43:08).
See [hindsight relabeling](hindsight-relabeling.md).

## Multi-task imitation learning

With demonstrations $\mathcal{D}_ i$ for each task, multi-task imitation learning minimizes the
single-task imitation loss summed over tasks, with one set of policy parameters $\theta$ (Lecture 12,
slide 18, ≈44:40–46:11):

$$\min_ \theta \sum_ {i=1}^{T} \mathcal{L}(\theta, \mathcal{D}_ i), \qquad \mathcal{L}(\theta, \mathcal{D}) = -E_ {(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}} \log \pi_\theta(\mathbf{a} \mid \mathbf{s})$$

The slide's sum runs to $T$, the number of tasks; Finn speaks of $n$ tasks. Two practical tricks help
(≈46:11–47:44). **Stratified sampling** builds every minibatch with data from each task, $1/n$ of it
from each for a uniform weighting, because minibatches dominated by very different tasks, going
forwards in one and lying down in the next, make the gradient swing from step to step; stratifying
gives lower-variance gradients. And, not on the slides, per-task normalization puts states and actions
on similar scales across tasks. See [imitation learning](imitation-learning.md).

The usual trade-off between imitation and RL carries over: imitation cannot use a reward to improve on
the expert data but is simple and scales easily to large datasets, while RL can do better than a
provided dataset when there is reward information to optimize (≈52:26–53:11).

## Feeding the task to a network

A policy that takes an image and a language string or one-hot vector has inherently multimodal input,
and its architecture has to handle that (≈47:44–48:32). Lecture 12 shows two designs.

**BC-Z** (Jang et al., CoRL 2021) trains a robot on simple manipulation tasks by imitation, with two
kinds of task descriptor: a language instruction ("Place bottle in the ceramic bowl" on the slide) through a
language encoder, or a video of a person doing the task through a video encoder. The resulting
embedding conditions several layers of the vision network that processes the image ("FiLM-Conditioned
Task Embedding"), and training samples the descriptor from either encoder and trains everything,
encoders included, with a behavior cloning loss (slide 19, ≈48:32–50:51).

![Lecture 12, slide 19 — How to condition on the task in practice?](../raw/images/12-multi-task-rl/slide-19.jpg)

*Lecture 12, slide 19 — BC-Z: a human demo video and a language instruction are encoded into one embedding, which conditions the vision network's layers. [Deck](https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf)*

**The prompt as task identifier.** For a language model the natural task descriptor is the prompt, so a
language model's supervised fine-tuning can be seen as multi-task imitation learning with prompts as
tasks (≈50:51). Modern robot policies such as **OpenVLA** (Kim et al., CoRL 2024) do the same: image
encoders are projected into a language model's input space, the instruction is passed in as tokens, and
the model outputs actions; the task identifier is "passed as prompt into a (fine-tuned) LLM" (slide 20,
≈50:51–51:37). See [instruction finetuning](instruction-finetuning.md).

![Lecture 12, slide 20 — How to condition on the task in practice?](../raw/images/12-multi-task-rl/slide-20.jpg)

*Lecture 12, slide 20 — OpenVLA: image features are projected into Llama 2's token space beside the tokenized instruction, and output tokens are decoded into a robot action. [Deck](https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf)*

A recent example, $\pi_{0.5}$ (Pi Team et al., 2025), follows instructions from "flatten the towel" and
"hang the towel" to more than ten in one kitchen, at different levels of abstraction, from "drive base
backward" to "throw the napkin away" (slides 21–22, ≈53:11–54:44). One policy for all tasks usually
beats separate policies, which train each set of weights on less data and cost memory and switching
time, though fitting many tasks into one model can raise optimization challenges (≈54:44–55:30). For
high-level, long-horizon tasks like "clean the kitchen", slide 22 points to the lecture on hierarchy,
[Lecture 15](15-hierarchical-rl.md), where a task- or goal-conditioned policy like these becomes the low level
of a two-level policy whose high level chooses the goals (see
[hierarchical RL and imitation](hierarchical-rl.md)).

## Multi-task RL

Multi-task RL conditions both the policy and the critic on the task (Lecture 12, slide 23,
≈57:02–58:34):

$$\pi_\theta(\mathbf{a} \mid \bar{\mathbf{s}}) \rightarrow \pi_\theta(\mathbf{a} \mid \bar{\mathbf{s}}, \mathbf{z}_ i), \qquad Q_\phi(\bar{\mathbf{s}}, \mathbf{a}) \rightarrow Q_\phi(\bar{\mathbf{s}}, \mathbf{a}, \mathbf{z}_ i)$$

With a discrete set of tasks, a replay buffer per task lets each batch be stratified across tasks. What RL
adds over imitation is the chance to **share data across tasks**: data collected while trying one task
may be useful for another, but reusing it "Would require relabeling the data" with the other task's
identifier and reward. That is [hindsight relabeling](hindsight-relabeling.md).

## Meta-RL: experience as the task identifier

Lecture 13 opens by recalling this lecture's aggregated MDP, with the warning "This is going to be a bit
different for meta-RL!" (Lecture 13, slide 3, ≈0:50). A meta-RL policy is not told its task; it reads its
own experience, which can be viewed as a task identifier, so meta-RL looks like multi-task learning in
which the data is the identifier, though that identifier changes within a task (slide 23, ≈41:31–42:16).
Where a multi-task policy with a structured prompt can transfer zero-shot, a meta-RL policy adapts k-shot,
from k episodes or time steps (slides 7 and 23, ≈6:59, ≈43:03). An index would not do as the identifier,
since a policy told "task 100" at test time cannot generalize to it, whereas experience can (≈44:34–46:07).
And because the task is not in the state, there is no aggregated MDP: meta-RL is a partially observed MDP
in which only the task is hidden (slide 25, ≈46:52–48:25). See [meta-RL](meta-rl.md).

Lecture 14 brings multi-task RL back inside meta-RL. Its DREAM algorithm trains an execution policy that
is given each training task's identifier, $\pi^{\mathrm{exec}}(\mathbf{a} \mid \mathbf{s}, \mu_i)$, which "is just
multi-task learning, or multi-task reinforcement learning" and can be trained by optimizing reward
without relying on exploration (Lecture 14, slide 19, ≈41:09–41:55). A separately trained exploration
policy then learns to collect the experience that identifies the task, through a bottlenecked task
representation that generalizes better than a one-hot index (slides 20–23). See
[exploration](exploration.md#learning-to-explore-meta-rl).

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md): multi-task RL and meta-RL among the course's topics and
  open problems.
- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md): the whole topic, with the examples and questions.
- [Hindsight relabeling](hindsight-relabeling.md): sharing data across tasks, and goal-conditioned RL.
- [Lecture 13 — Meta RL](13-meta-rl.md#experience-as-a-task-identifier): meta-RL against multi-task RL,
  and the recap of this lecture.
- [MDPs and notation](mdp-and-notation.md#notation-for-multi-task-rl): the notation for tasks and task
  identifiers.
- [Lecture 14 — Exploration](14-exploration.md#solution-30-explore-to-identify-the-task): DREAM's
  ID-conditioned execution policy, trained as multi-task RL.
- [Lecture 15 — Hierarchical RL and IL](15-hierarchical-rl.md): a multi-task, goal-conditioned policy as the
  low level of a hierarchy, and why it shares knowledge across similar subtasks.
