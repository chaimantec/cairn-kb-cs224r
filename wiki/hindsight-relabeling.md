# Hindsight relabeling

In multi-task RL, experience collected while trying one task can be stored again under another task it
turned out to be good for, with that task's identifier and reward. Because the new label is chosen after
seeing what happened, this is **hindsight relabeling**, also called hindsight experience replay (HER).
With goals as tasks, relabeling a trajectory with the state it actually reached turns a failure into a
success. [Lecture 12](12-multi-task-rl.md) teaches it, and students implement it in Homework 4. For
tasks, task identifiers and multi-task RL in general, see [multi-task RL](multi-task-rl.md).

## Why relabel

A multi-task policy $\pi_\theta(\mathbf{a} \mid \bar{\mathbf{s}}, \mathbf{z}_ i)$ is told its task by an
identifier $\mathbf{z}_ i$, and with a discrete set of tasks it may keep a replay buffer per task
(Lecture 12, slide 23). Suppose the policy was told to pick up the sponge and, exploring, did something
else. That experience might be useful for other tasks, but with one buffer per task they never learn
from it. Reusing it requires relabeling: if the tasks have different reward functions but the same
dynamics, the reward label must be changed to the other task's reward (≈57:47–59:21).

Lecture 12's example is hockey (slide 24, ≈59:21–1:00:54). One task is passing, another is shooting a
goal. Trying to shoot, the player misses and accidentally makes a pretty good pass to a teammate. That
data is probably useful for getting better at passing, so it is stored as normal under the shooting task,
and also relabeled with the passing task's ID and reward and stored for that task. The slide prints
"task 2 ID" for the relabel; Finn calls that a typo, since passing is task 1 (≈1:00:08).

![Lecture 12, slide 24 — An example](../raw/images/12-multi-task-rl/slide-24.jpg)

*Lecture 12, slide 24 — a failed shot that happens to be a good pass is stored twice: under the shooting task and relabeled for passing. The slide's "task 2 ID" should read task 1, as Finn says. [Deck](https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf)*

## The algorithm

Slide 25 gives multi-task RL with relabeling, with one replay buffer $\mathcal{D}$ (Finn's board version
keeps one per task; either works, ≈1:00:54–1:04:44):

1. Collect data $\mathcal{D}_ k = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{z}_ i, r_{1:T}) \rbrace$ using some policy.
2. Store it, $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_ k$.
3. Relabel the experience in $\mathcal{D}_ k$ for a task $\mathcal{T}_ j$,
   $\mathcal{D}'_ k = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{z}_ j, r'_ {1:T}) \rbrace$ with
   $r'_ t = r_j(\mathbf{s}_ t)$, and store it, $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}'_ k$.
4. Update the policy using $\mathcal{D}$, and repeat.

Which task $j$ to relabel for is "somewhat of a heuristic process", because it is hard to tell whether
data will help learn a task (≈1:02:26–1:03:57). Slide 25 suggests choosing "randomly", or the
"task(s) in which the trajectory gets high reward": if $r_j$ was high, something good was done for task
$j$ by accident, and it can be the trajectory, not just one transition, that earns the high reward
(≈1:04:44). Nothing here assumes discrete states (≈1:03:12).

## When relabeling applies

Relabeling restricts both the tasks and the learning algorithm (Lecture 12, slide 25,
≈1:04:44–1:09:21):

- **The same state and action spaces**, with matching dimensions if there is one buffer, and a similar
  initial state distribution, or sharing data will not be as useful.
- **The same dynamics.** Relabeling changes the reward, not the dynamics, so this is really important:
  if the dynamics differ across tasks, the relabeled data teaches the wrong dynamics.
- **A computable reward.** The new task's reward must be computed, so its functional form must be known
  ("reward function form is known, evaluatable"), or the reward function must have been learned. It helps
  for rewards to be on the same scale. Relabeling is most useful when the reward functions differ;
  otherwise the data is already in the buffer with the right reward.
- **An off-policy algorithm.** The relabeled action was taken by the policy conditioned on
  $\mathbf{z}_ i$; conditioned on $\mathbf{z}_ j$ it might have acted differently, so relabeled data is
  off-policy data. On importance weights, Finn first agrees that one would need to store which policy
  collected the data, then reconsiders and says one would use $\mathbf{z}_ j$, not $\mathbf{z}_ i$, as
  the old policy, since the relabeled data is as if collected with $\mathbf{z}_ j$, so "you don't need
  to store which policy it came from" (≈1:08:35–1:09:21). The slide says only "using an off-policy
  algorithm*", with an asterisk the page does not explain. See
  [offline and online algorithms](offline-and-online.md) and [importance sampling](importance-sampling.md).

Slide 25 cites Kaelbling, "Learning to Achieve Goals" (IJCAI '93), and Andrychowicz et al., "Hindsight
Experience Replay" (NeurIPS '17).

## Goal-conditioned RL with hindsight relabeling

In goal-conditioned RL the task identifier is a goal state $\mathbf{s}_ g$ and the reward is a negative
distance to it (Lecture 12, slide 17). For continuous states the distance could be a mean squared error,
a distance in an embedding space, or a zero-one loss within some epsilon ball of the goal state
(≈1:09:21–1:10:21). The recording ends there; the rest of slide 26 is only in the
[slide file](../raw/slides/12-multi-task-rl.md). It relabels with **the last state of the trajectory as
the goal**:

$$\mathcal{D}'_ k = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{s}_ T, r'_ {1:T}) \rbrace, \qquad r'_ t = -d(\mathbf{s}_ t, \mathbf{s}_ T)$$

so whatever state the trajectory reached becomes a goal it succeeded at. As "Other relabeling
strategies?", the slide offers to "use any state" from the trajectory, and its result is "exploration
challenges alleviated".

Slide 27, not discussed in the recording, shows the Hindsight Experience Replay paper's simulated robot
arm pushing, sliding and picking-and-placing objects to a goal. Plain DDPG's success rate stays near 0% on
all three, and count-based exploration reaches only about 22% on sliding; with HER, pushing reaches about
94–95% by epoch 200, sliding about 87% for the paper's Section 4.5 variant and 42% for plain DDPG+HER,
and pick-and-place about 95% by epoch 60 (values read from the chart by eye).

![Lecture 12, slide 27 — Hindsight relabeling for goal-conditioned RL](../raw/images/12-multi-task-rl/slide-27.jpg)

*Lecture 12, slide 27 — without relabeling, DDPG's red dashed line stays at zero on all three tasks; the blue and red HER curves are what learn. [Deck](https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf)*

"You will implement hindsight relabeling in homework 4!" (slide 26). Homework 4's handout implements
Hindsight Experience Replay on top of a goal-conditioned DQN; see [course logistics](course-logistics.md).

## Where this appears

- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md#hindsight-relabeling): the hockey example, the
  algorithm, its conditions and the goal-conditioned version.
- [Multi-task RL](multi-task-rl.md): tasks, task identifiers and multi-task RL.
- [Replay buffers and soft actor-critic](replay-buffers.md#relabeled-data-for-other-tasks): replay
  buffers that hold relabeled data.
- [Offline and online algorithms](offline-and-online.md#relabeled-data-is-off-policy): why relabeling
  needs an off-policy method.
