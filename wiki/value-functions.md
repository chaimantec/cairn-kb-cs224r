# Value functions

Improving a policy requires knowing how good it is, and value functions are how CS224R measures
that (Lecture 1, slide 40, ≈47:47–48:32). Symbols are as on
[MDPs and RL notation](mdp-and-notation.md); $\pi$ is a [policy](policies.md).

## The value function $V^\pi$

$V^\pi(\mathbf{s})$ is the future expected reward from starting at state $\mathbf{s}$ and
following the policy $\pi$ from then onward. It is the [RL objective](rl-objective.md), the
expected sum of rewards, but started at $\mathbf{s}$ rather than at a state drawn from the
initial state distribution (≈48:32).

## The Q-function $Q^\pi$

$Q^\pi(\mathbf{s}, \mathbf{a})$ is the future expected reward from starting at $\mathbf{s}$,
taking the action $\mathbf{a}$, and then following $\pi$. The first action $\mathbf{a}$ may differ
from what $\pi$ would have chosen in $\mathbf{s}$, which is what makes the Q-function useful for
comparing actions (≈49:20).

## What Lecture 1 does and does not say

Lecture 1 only introduces the two functions, because the course uses them heavily from then on;
it does not go into depth or give equations for them beyond these definitions. Q-functions start
being used from the week after Lecture 1 (≈49:20). Two of the [algorithm
families](rl-algorithm-families.md) are defined by them: actor-critic methods estimate the value
of the current policy and use it to improve the policy, and value-based methods estimate the
value of the optimal policy and back a good policy out of it (slide 41).

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#how-good-is-a-policy-value-and-q-functions): the
  definitions of $V^\pi$ and $Q^\pi$.
