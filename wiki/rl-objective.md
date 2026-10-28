# The RL objective

What a reinforcement learning agent is trying to do, as Lecture 1 builds it up. Symbols are as
defined on [MDPs and RL notation](mdp-and-notation.md): $\mathbf{s}_ t$ and $\mathbf{a}_ t$ are
the state and action at time $t$, $T$ is the trajectory's length, $r$ is the reward function and
$\pi_\theta$ is the policy with parameters $\theta$.

## From a sum of rewards to an expected sum

The goal is to maximize reward over the whole trajectory, not at a single time step, which first
suggests maximizing the sum of rewards (Lecture 1, ≈38:32):

$$\max \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$$

But that sum is not a deterministic quantity. Running the same policy twice gives different
trajectories and different rewards.

### Where the randomness comes from

Asked for the sources of variability, the class suggested stochastic dynamics, a stochastic
policy, a stochastic reward and changing goals (≈39:18–40:49). Rewards are typically modeled as
deterministic functions; randomness that seems to live in the reward, such as a goal that cannot
be observed, can be pushed into the state without loss of generality, and goals that change are
usually treated as a different decision process. That leaves **two main sources** (≈40:49–41:36):

1. **the world is stochastic**: the dynamics $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$
   are a distribution, not a deterministic function;
2. **the policy is stochastic**: it may not make the same decision every time.

In driving, other cars behave randomly, and your own car may not make the same decision every
time. The initial state can be random too, like the deal in a card game (≈42:23).

### The trajectory distribution

Putting the initial state, the policy and the dynamics together gives a probability distribution
over trajectories, written $p_\theta(\tau)$ (≈41:36–43:09):

$$p_\theta(\tau) = p(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T) = p(\mathbf{s}_ 1) \prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \thinspace p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

Read left to right, this is the probability of the first state, then of the first action under
the policy, then of the next state under the dynamics, and so on. Slide 37 draws it as a
graphical model in which each state feeds the policy's action and both feed the next state, with
the formula beneath; slide 38 restates the formula. (The lecture says "plus the product" at
≈43:09; the slides multiply.)

![Slide 37 — What is the goal of reinforcement learning?](../raw/images/01-class-intro/slide-37.jpg)

*Slide 37 — the sum of rewards is random: the policy (state to action) and the dynamics (state and action to next state) in the graphical model are the two sources, and the trajectory probability multiplies them. [Deck](https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf)*

### The objective

The goal of RL is to find the policy parameters that maximize the **expected** sum of rewards
under that distribution (slide 38, ≈43:09–43:56):

$$\max_{\theta} \thinspace \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

The policy does not appear inside the sum. It affects the objective only through
$p_\theta(\tau)$, the distribution of trajectories it produces, and that is what makes RL harder
than supervised learning, where the parameters sit directly in the loss. Much of the course is
about how to optimize behavior under this objective (≈44:42). The
[algorithm families](rl-algorithm-families.md) are different ways of doing it.

## Horizon and discounting

The horizon $T$ is the number of steps in a trajectory. It is written as fixed because that is
easier, but it can vary (≈44:42–45:28).

The plain sum weights every reward equally: it cares about immediate rewards exactly as much as
rewards far in the future. That is not always what is wanted. If $T$ were 1,000 years, rewards
1,000 years from now should probably count for less than rewards today. A **discount factor**
handles this by weighting the reward at time $t$ by the discount factor raised to the power $t$.
It is usually greater than 0 and at most 1. A discount factor of 1 recovers the undiscounted
objective, and lower values make the agent greedier, caring more about near-term rewards than
distant ones (≈45:28–47:01).

Lecture 1 introduces the discount factor in speech only: it is on no slide of that deck and is
given no symbol. The lecture says discount factors return in a couple of lectures.

## How good is a particular policy?

The same expectation, started from a given state (and action) instead of the initial state
distribution, defines the [value function and the Q-function](value-functions.md)
(slide 40).

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#the-goal-of-reinforcement-learning): the objective,
  its sources of randomness, the horizon and the discount factor.
