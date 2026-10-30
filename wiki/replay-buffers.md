# Replay buffers and soft actor-critic

A **replay buffer** keeps all the experience an agent has collected, so that each update can draw on
all past trial-and-error data instead of only the latest batch. Lecture 5 adds one to the
[actor-critic](actor-critic.md) algorithm. That breaks two of its steps, and fixing them gives an
algorithm that is "very, very close to" soft actor-critic (SAC). Symbols are as on
[MDPs and RL notation](mdp-and-notation.md): $\mathbf{s}$ is a state, $\mathbf{a}$ an action, $r$
the reward, $\mathbf{s}'$ the next state, $\gamma$ the discount factor, and $\pi_\theta$ the current
[policy](policies.md). Timestamps (≈M:SS) are into
[Lecture 5's recording](05-off-policy-actor-critic.md).

## How off-policy can actor-critic be?

One batch of policy data can buy one gradient step, which is fully on-policy, or several, which is
"starting to be off-policy" and is what [PPO](ppo.md) does (Lecture 5, slide 14, ≈46:19). Going
further, "Can we reuse data from previous batches, i.e. all of the past trial-and-error data?" Two
ideas make that possible: maintain a buffer of all past data, and adjust the equations to remove
their on-policy assumptions. The buffer is a dataset that keeps growing and from which very old data
may eventually be removed, called a replay buffer "because you're sort of replaying your past
experiences" (≈47:05). Students first met one in Homework 1.

With a buffer, the algorithm can run much more online: get one time step of data, update, get
another, update again (slide 15, ≈47:50–49:23). Each transition
$(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$ goes into the buffer $\mathcal{R}$, and each update uses
a minibatch sampled from all previous data.

## Why the naive version is broken

Putting the buffer into the algorithm unchanged gives an algorithm that "is currently broken"
(slide 16, ≈49:23–53:18).

**The value function.** A bootstrapped value target $r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$,
fitted on transitions from every past policy, is not a value function for the current policy.
Lecture 5's think-pair-share asks which policy it is for (≈50:10). The buffer holds almost no data
from the current policy, and the past policies' actions differ, so their next states and rewards
differ too. Finn's answer is "some sort of mixture of all of your past policies", not trivial to
write down: roughly an average of the past policies, but weighted differently from state to state,
depending on which policies visited each state (≈50:57–52:30).

**The policy update.** The action $\mathbf{a}_ i$ from the buffer, used in the gradient and in the
advantage, is "not the action $\pi_\theta$ would have taken!"

![Lecture 5, slide 16 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-16.jpg)

*Lecture 5, slide 16 — the broken algorithm: the circled value target is not the right one, the circled buffer action is not one the current policy would have taken, and the teal question asks which policy a value function fitted on the buffer belongs to. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## Fix 1: fit a Q-function, with the next action from the current policy

Fit $Q$ instead of $V$ (Lecture 5, slides 17–19, ≈53:18–1:04:50). The [Q-function](value-functions.md)
takes the action as input, so it does not matter that the buffer's action came from a past policy.
And it can be written recursively, as the reward now plus the expected Q-value at the next state
under the next action the policy would take:

$$Q^{\pi_\theta}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a}), \bar{\mathbf{a}}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ Q^{\pi_\theta}(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

where $p$ is the dynamics. This holds "For *any* $(\mathbf{s}, \mathbf{a})$", whichever policy chose
the action. The buffer's data points also carry the future reward of the trajectory they came from,
but it "depends on past policy's actions", so it is not used.

The target therefore takes the buffer's state, action, reward and next state, but samples the next
action $\bar{\mathbf{a}}'_ i$ fresh from the *current* policy (slide 18):

$$Q^{\pi_\theta}(\mathbf{s}_ i, \mathbf{a}_ i) \approx r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma Q^{\pi_\theta}(\mathbf{s}'_ i, \bar{\mathbf{a}}'_ i) = y_i$$

A network $\hat{Q}^\pi_\phi$, with parameters $\phi$, takes the state and action as input and is
regressed onto $y_i$ with a squared error. Because the next action comes from the current policy and
the dynamics are the same whoever acted, the estimate is of the current policy's Q-function.

![Lecture 5, slide 18 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-18.jpg)

*Lecture 5, slide 18 — fitting Q from the buffer: the target y is the reward plus the Q-value of the next state with an action sampled from the current policy, regressed by a network that takes s and a as input. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

**Coverage is the catch.** "To be accurate, targets $y_i$ need sufficient action coverage" (slide 18).
The data need not include every state-action pair, but it must cover actions similar to the ones the
current policy would take; no data from a state is a problem, and so is data from it only for very
different actions. Neural networks interpolate reasonably well, which helps. So the buffer does not
give an accurate Q-function for free, but it gives "somewhat of a means to use old data"
(≈1:00:15–1:02:32). The single sampled next action estimates an expectation, and some methods use
several samples (≈1:02:32).

## Fix 2: sample the policy update's action from the current policy

The policy update gets two changes (slide 20, ≈1:04:50–1:06:23). It weights by $\hat{Q}^\pi$ instead
of the advantage, which drops the baseline: higher variance, but acceptable because it now uses all
the data in the buffer. And because the current policy's actions are likely better than past
policies' actions, it samples a fresh action rather than using the buffer's:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \thinspace \hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i), \qquad \mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$$

It is the same trick as in the Q-function's target, at the current state instead of the next one.

## The algorithm, and what remains

The online actor-critic algorithm with a replay buffer (Lecture 5, slide 21):

1. Take an action $\mathbf{a} \sim \pi_\theta(\mathbf{a} \mid \mathbf{s})$, get
   $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, and store it in $\mathcal{R}$.
2. Sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i \rbrace$ from
   $\mathcal{R}$.
3. Update $\hat{Q}^\pi_\phi$ using targets
   $y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$, with
   $\mathbf{a}'_ i$ sampled from the current policy.
4. Compute the gradient above with $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$.
5. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$, with step size $\alpha$.

One problem remains: the buffer's states did not come from the current policy's state distribution
$p_\theta(\mathbf{s})$. "nothing we can do here, just accept it", and it is not necessarily bad: the
policy is trained on a broader distribution than the one it would visit (slide 21, ≈1:06:23).

This is essentially **soft actor-critic** (Haarnoja, Zhou, Abbeel, Levine, 2018; slide 22). There
are fancier ways to fit the Q-function, which the next lecture covers (see
[Q-learning](q-learning.md)), and other ways to estimate
the gradient, such as the **reparameterization trick** for a Gaussian policy (slide 22,
≈1:06:23–1:07:08).

## Data efficiency against stability

On two humanoid tasks, SAC, with a replay buffer, learns far faster than [PPO](ppo.md), which has
none (Lecture 5, slide 23, ≈1:07:08–1:07:55). The trade-off, in the slide's words: off-policy
methods with a replay buffer "can be far more data efficient", but "can also generally be a lot
harder to tune hyperparameters, less stable (than e.g. PPO)".

![Lecture 5, slide 23 — More off-policy vs. less off-policy actor critic?](../raw/images/05-off-policy-actor-critic/slide-23.jpg)

*Lecture 5, slide 23 — learning curves on two humanoid tasks: SAC (orange, with a replay buffer) learns far faster than PPO (brown, no replay buffer), most clearly on the right-hand task, at the cost of being harder to tune. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

That efficiency is enough for RL on real robots (slides 24–25, ≈1:07:55–1:08:40). A four-legged
robot learns to walk from scratch in under two hours with SAC. An SAC-based method seeded with
demonstrations (HIL-SERL) reaches 100% success on precise assembly tasks where behaviour cloning
gets much lower success rates, with faster cycle times.

## In Q-learning

Lecture 6's [Q-learning](q-learning.md) uses the same buffer without an actor: collect transitions
with an exploration policy such as epsilon-greedy, sample batches from the buffer uniformly, and
take $K$ gradient steps on the Q-function per round of collection (slide 15, ≈33:23–34:55).
Sampling from the whole buffer rather than from what was just collected decorrelates the batch
(≈34:09). The lecture adds how a buffer is usually managed (≈38:45–40:18). It has a fixed size,
often a large one, and once full it discards the oldest data first, first in, first out. At the
start it is small, so the first iteration sometimes collects extra data to pre-populate it. And in
asynchronous algorithms, where some threads add data while others update the network, managing the
buffer matters especially, so that no data point gets a thousand gradient steps while another gets
one.

Lecture 6 closes the course's online methods with advice on choosing among them (slide 27,
"Chelsea's advice", ≈59:41–1:00:27): SAC and its variants when you care most about data efficiency
and can accept tuning and less stability, PPO when you care about stability and ease of use rather
than data efficiency, and DQN for discrete or low-dimensional continuous actions.

## With no new data

Lecture 7 recalls the action-coverage requirement: because the Q target bootstraps from the critic's own
estimate, the critic must have been trained on actions resembling those it is evaluated on, so it "needs
good coverage over actions" (≈6:17–7:02). An online buffer keeps receiving the current policy's actions.
A fixed offline dataset never does, and running soft actor-critic on one lets the policy exploit the
critic on actions the data does not contain, overestimating their Q-values (slides 10–11). That failure
is the starting point of [offline RL](offline-rl.md).

## Relabeled data for other tasks

In multi-task RL with a discrete set of tasks, it can help to keep a replay buffer per task, especially
to balance the sampling of each batch across tasks (Lecture 12, slide 23, ≈57:47). A buffer can also
hold data that no policy collected for the task it is labeled with: **hindsight relabeling** stores an
experience under its original task and again under another task, with that task's identifier and
reward, either in a shared buffer or in the other task's (≈1:01:40–1:02:26). Relabeled data is off-policy
by construction, so it needs an off-policy learner (≈1:07:50–1:08:35). See
[hindsight relabeling](hindsight-relabeling.md).

## Where this appears

- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md#version-2-replay-buffers):
  the derivation, the think-pair-share, the questions on coverage, and the robot examples.
- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md#not-in-the-recording-off-policy-actor-critic):
  the first version of these slides, which that recording does not reach.
- [Policy evaluation](policy-evaluation.md): fitting value functions by Monte Carlo and
  bootstrapping, which the Q-function target extends.
- [Lecture 6 — Q-Learning](06-q-learning.md#full-q-learning-with-a-replay-buffer): the buffer in
  Q-learning, how it is managed, and the advice on choosing PPO, DQN or SAC.
- [Offline and online algorithms](offline-and-online.md): on-policy and off-policy.
- [Lecture 7 — Offline RL](07-offline-rl.md#can-we-just-use-off-policy-algorithms): the coverage
  requirement recalled, and what goes wrong when the buffer is a fixed dataset.
- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md#multi-task-rl-and-sharing-data-across-tasks):
  per-task replay buffers, and buffers holding relabeled data.
