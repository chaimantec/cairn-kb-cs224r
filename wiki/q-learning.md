# Q-learning

Q-learning is the course's value-based method: it learns a Q-function and acts greedily on it,
without a policy network or a policy gradient. Lecture 1 names the family, methods that "estimate
the value of the optimal policy" and back a good policy out of it (slide 41). Lecture 6 derives the
algorithm, from an actor-critic method to Q-learning, and then shows the three tricks that make it
work in practice: target networks, which give DQN, double Q-learning and n-step returns. Symbols are
as on [MDPs and RL notation](mdp-and-notation.md).

## From a Q-function to a policy

The Q-function $Q^\pi(\mathbf{s}, \mathbf{a})$ is the expected sum of future rewards from taking
action $\mathbf{a}$ in state $\mathbf{s}$ and then following the policy $\pi$ (see
[value functions](value-functions.md)). So
$\arg\max_{\mathbf{a}_ t} Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t)$ is the best action from
$\mathbf{s}_ t$ if $\pi$ is followed afterwards, and it is "*at least* as good as any
$\mathbf{a}_ t \sim \pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$ *regardless* of what
$\pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$ is" (Lecture 6, slide 8, ≈12:32–13:18). The deterministic
greedy policy

$$\pi'(\mathbf{a}_ t \mid \mathbf{s}_ t) = \begin{cases} 1 & \text{if } \mathbf{a}_ t = \arg\max_{\mathbf{a}} Q^\pi(\mathbf{s}_ t, \mathbf{a}) \cr 0 & \text{otherwise} \end{cases}$$

is therefore at least as good as $\pi$, given an accurate $Q^\pi$, and the same as $\pi$ if $\pi$ is
already optimal (Lecture 6, ≈11:46–12:32). Lecture 6's thought exercise shows it is not optimal
after one step: in a 2D world with a reward only at a star, the greedy policy built from a
go-right policy's Q-values learns to go up only in the row just below the star's, because every
Q-value further down is still zero (slide 7, ≈7:08–10:13). Repeating the step spreads the reward
back one row at a time, which is what a **backup** is, and it has to continue for as long as the
horizon of the problem (≈15:40–16:26).

Alternating the two steps is **policy iteration**: fit $Q^\pi$ for the current policy, which is
**policy evaluation**, then define the greedy policy, which is **policy improvement** (Lecture 6,
slide 10, ≈17:12–18:00). No network represents the policy; the only model is the Q-function
(≈14:05). See [policy evaluation](policy-evaluation.md).

## The Q-learning target

Instead of evaluating the old policy and then improving it, put the improvement into the
Q-function's regression target (Lecture 6, slide 11, ≈20:19–21:04). With $\hat{Q}_ \phi$ a
Q-network with parameters $\phi$, $\gamma$ the discount factor, and $(\mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i)$
a transition with reward $r_i$ and next state $\mathbf{s}'_ i$, the target is

$$y_i = r_i + \gamma \max_{\mathbf{a}'} \hat{Q}_ \phi(\mathbf{s}'_ i, \mathbf{a}')$$

which the slide labels "Q-values for new policy!": the targets are already the Q-values of the
greedy policy about to be defined. The second way to see it is that this target is the right-hand
side of the **Bellman optimality equation**, which only the optimal policy's Q-function satisfies,
so fitting the target is trying to make that equation hold (slide 12, ≈21:04–25:42). See
[Bellman equations](bellman-equations.md).

Q rather than V, because a policy comes straight out of Q through the $\arg\max$. Knowing only
$V(\mathbf{s})$, you would have to predict which next state each action leads to, which needs the
dynamics (Lecture 6, ≈23:24–24:56).

**Convergence.** With a table holding a Q-value for every state and action, and sufficient
exploration, so that the data covers every state and action, Q-learning converges. More generally
it is not guaranteed to, and it can diverge even with a linear Q-function, though it "can be made
to work well" (Lecture 6, slide 13, ≈27:15–28:48).

## What data to collect

Q-learning is **off-policy**: the Bellman optimality equation holds for every state and action, so
the data need not come from the current policy (Lecture 6, slides 13–14, ≈28:48). It should in fact
be broader, because the max in the target considers many actions, and an action with no data can
have an erroneously high Q-value that corrupts the target (≈29:34). So data comes from an
exploration policy (slide 14):

- **Epsilon-greedy**: with probability $\epsilon$ take a uniformly random action, otherwise the
  greedy one; written out, the greedy action has probability $1 - \epsilon$ and each of the others
  $\epsilon / (\vert \mathcal{A} \vert - 1)$, with $\vert \mathcal{A} \vert$ the number of actions.
  Training often starts with a larger $\epsilon$ and lowers it (≈30:19–31:51).
- **Boltzmann exploration**: take actions with probability proportional to the exponentiated
  Q-value, $\pi(\mathbf{a}_ t \mid \mathbf{s}_ t) \propto \exp(Q_\phi(\mathbf{s}_ t, \mathbf{a}_ t))$
  (≈31:51–32:37).

Epsilon-greedy is a lot more common with discrete actions, and the Boltzmann form a bit more common
with continuous ones (≈32:37–33:23). See [policies](policies.md).

## The full algorithm

"full Q-learning with replay buffer" (Lecture 6, slide 15, ≈33:23–34:55), with $\mathcal{R}$ the
[replay buffer](replay-buffers.md) and $\alpha$ a step size:

1. collect data $\lbrace (\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i) \rbrace$ using some policy, and add it to $\mathcal{R}$
2. sample a batch $(\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i)$ from $\mathcal{R}$
3. take a gradient step on the squared error between $Q_\phi(\mathbf{s}_ i, \mathbf{a}_ i)$ and the target:

$$\phi \leftarrow \phi - \alpha \sum_i \frac{dQ_\phi}{d\phi}(\mathbf{s}_ i, \mathbf{a}_ i) \left( Q_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - \left[ r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \max_{\mathbf{a}'} Q_\phi(\mathbf{s}'_ i, \mathbf{a}') \right] \right)$$

Steps 2–3 repeat $K$ times per round of collection: "K = 1 is common, though larger K more
efficient". Sampling the batch from the whole buffer rather than from what was just collected
decorrelates it (≈34:09). The result is $Q_\phi$, and the policy deployed at the end is greedy with
respect to it.

![Lecture 6, slide 15 — Putting it together](../raw/images/06-q-learning/slide-15.jpg)

*Lecture 6, slide 15 — the full algorithm: an exploration policy such as epsilon-greedy fills the replay buffer from the world, and off-policy Q-learning samples batches from it, with K gradient steps per round of collection. [Deck](https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf)*

## Making it work: three tricks

**Trick 1: a target network.** The target's max changes every time the parameters do, a "moving
target" that "can lead to unstable optimization" (Lecture 6, slide 17, ≈40:18–41:03). So keep a
frozen copy of the parameters, $\phi'$, use it only in the target, and copy $\phi$ into it
periodically, for example every 100 gradient steps (≈41:03–42:38):

$$y_i = r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}'_ i, \mathbf{a}')$$

Between copies the labels do not change, so the inner loop is supervised learning (slide 18). This
is the **DQN** algorithm, "deep Q network" (Mnih et al., Playing Atari with Deep Reinforcement
Learning, 2013), which in Finn's opinion was basically the first deep RL method: it learned Atari
games from pixels (≈42:38–43:25). Students implement it in homework 2.

![Lecture 6, slide 18 — How to make Q-learning stable?](../raw/images/06-q-learning/slide-18.jpg)

*Lecture 6, slide 18 — DQN: the target uses frozen parameters phi prime, saved in step 0 and refreshed every outer loop, so the inner loops are supervised learning on labels that do not change. [Deck](https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf)*

**Trick 2: double Q-learning.** Learned Q-values rise with the return during training, but they sit
far above the true return of the policy they describe: Q-learning **overestimates** (slides 19–20,
≈44:11–47:16). The cause is the max. Since
$\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}') = Q_{\phi'}(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}'))$,
the same noisy network picks the action and evaluates it, so it picks actions whose values are high
because of noise (slide 21, ≈47:16–48:49). Picking with one network and evaluating with another
decorrelates the noise (slide 22). In practice the current network picks and the target network
evaluates (slide 23, ≈51:55):

$$y = r + \gamma Q_{\phi'}\left(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_\phi(\mathbf{s}', \mathbf{a}')\right)$$

Some overestimation remains, but much less (slide 20, ≈50:22–51:08).

![Lecture 6, slide 20 — Are the Q-values accurate?](../raw/images/06-q-learning/slide-20.jpg)

*Lecture 6, slide 20 — on four Atari games, DQN's value estimates (orange) sit far above the true value of its policy (red line), while double DQN's estimates (blue) lie much closer to theirs (blue line). [Deck](https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf)*

**Trick 3: n-step returns.** Early in training, when the Q-network is bad, the single reward in the
target is the only part that carries information (slide 24, ≈52:42–53:28). Summing the rewards of
the next $N$ steps before bootstrapping, as for value functions in Lecture 4, gives less biased
targets and typically faster learning, especially early on (slides 24–25, ≈54:16–55:47). With $j$
indexing the samples in the batch, the slides print

$$y_{j,t} = \sum_{t'=t}^{t+N-1} \gamma^{t-t'} r_{j,t'} + \gamma^N \max_{\mathbf{a}_ {j,t+N}} Q_{\phi'}(\mathbf{s}_ {j,t+N}, \mathbf{a}_ {j,t+N})$$

where the exponent $t - t'$ appears to be a slip for $t' - t$ (see the
[Lecture 6 page](06-q-learning.md#trick-3-n-step-returns)). The catch is that the summed rewards
come from the policy that collected the data, so the target is "only actually correct when learning
on-policy", which is not an issue when $N = 1$. Most commonly the problem is ignored and $N \gt 1$
used anyway, since it very often helps; alternatively choose $N$ dynamically to use only data that
follows the current policy, or use [importance sampling](importance-sampling.md) (slide 25,
≈56:33–57:19).

## Continuous actions

The max is easiest over a discrete set of actions (Lecture 6, ≈14:05–15:40). With continuous
actions, gradient descent on the action works poorly; sampling many actions and keeping the one
with the highest Q-value can be better if the action space is not very high-dimensional, and other
sampling-based optimizers iteratively sample and refine. The robot-grasping system of Kalashnikov
et al. (2018) does a continuous optimization of this kind over its Q-function's actions and
reaches 96% grasp success (slide 26, ≈58:05–58:55).

## When to use it

On Atari, DQN learned to outperform people on most games (Mnih et al. 2015; Lecture 6, slide 26).
Finn's advice (slide 27, ≈59:41–1:00:27) is to use DQN and its variants when the actions are
discrete or low-dimensional continuous, where it can be fairly data-efficient; PPO when stability
and ease of use matter more than data efficiency; and SAC when data efficiency matters most and
tuning and some instability are acceptable. See [RL algorithm families](rl-algorithm-families.md).

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#types-of-algorithms-and-why-there-are-so-many): the
  value-based family among the five.
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md#fitting-a-q-function-with-data-from-past-policies):
  fitting a Q-function from a replay buffer, the step Q-learning starts from.
- [Lecture 6 — Q-Learning](06-q-learning.md): the whole algorithm, its exploration policies, DQN,
  double Q-learning, n-step returns and the examples.
