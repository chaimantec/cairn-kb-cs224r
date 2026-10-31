# Bellman equations

A Bellman equation writes a value as the immediate reward plus the discounted value of what comes
next. CS224R uses two. The **Bellman equation** holds for the Q-function of any policy and is what
lets a Q-function be fitted from other policies' data. The **Bellman optimality equation** holds
only for the optimal policy and is what [Q-learning](q-learning.md) tries to satisfy. Lecture 5
derives the first without naming it; Lecture 6 adds the second and gives both their names (slide
12, ≈26:29–27:15). Symbols are as on [MDPs and RL notation](mdp-and-notation.md).

## The Bellman equation

The Q-function $Q^\pi(\mathbf{s}, \mathbf{a})$ is the expected sum of rewards from taking action
$\mathbf{a}$ in state $\mathbf{s}$ and then following the policy $\pi$ (see
[value functions](value-functions.md)). Splitting the first reward off that sum leaves the rewards
from the next step on, which in expectation over the next state $\mathbf{s}'$, drawn from the
dynamics $p$, and the next action $\bar{\mathbf{a}}'$, drawn from the policy, are the next Q-value
(Lecture 5, slide 17, ≈53:18–57:10). With $r$ the reward and $\gamma$ the discount factor:

$$Q^\pi(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a}), \bar{\mathbf{a}}' \sim \pi(\cdot \mid \mathbf{s}')} \left[ Q^\pi(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

It holds "for *all* $(\mathbf{s}, \mathbf{a})$", and for any policy, including the optimal one
(Lecture 6, slide 12, ≈21:04–21:50). Because it holds whichever policy chose the action
$\mathbf{a}$, a transition $(\mathbf{s}, \mathbf{a}, \mathbf{s}')$ collected by any earlier policy
gives a regression target for the current policy's Q-function, once the next action is sampled from
the current policy: $y = r(\mathbf{s}, \mathbf{a}) + \gamma Q^\pi(\mathbf{s}', \bar{\mathbf{a}}')$.
That is **policy evaluation** from a replay buffer, the step soft actor-critic is built on (Lecture
5, slides 17–18; see [replay buffers and soft actor-critic](replay-buffers.md) and
[policy evaluation](policy-evaluation.md)).

Lecture 7's advantage-weighted actor-critic fits the behavior policy's Q-function from offline data by
taking the target's next action from the data, $r + \gamma E_{\mathbf{a}' \sim D}[\hat{Q}^\pi_ \psi(\mathbf{s}', \mathbf{a}')]$,
so the Q-function is only ever evaluated on actions the data contains (slide 20). See
[implicit policy constraint methods](implicit-policy-constraints.md#awac-bootstrapping-with-the-datas-own-next-actions).

## The Bellman optimality equation

The optimal policy $\pi^\ast$ always takes the action that maximizes future reward, which is the
action with the highest Q-value. So for the optimal policy the expectation over the next action
becomes a maximization (Lecture 6, slide 12, ≈21:50–22:37):

$$Q^{\pi^\ast}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a})} \left[ \max_{\bar{\mathbf{a}}'} Q^{\pi^\ast}(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

The max term is "the action $\pi^\ast$ would take in state $\mathbf{s}'$" (slide 12). The equation
is an if and only if: it holds only for the optimal policy, and a Q-function that satisfies it
belongs to the optimal policy (≈25:42).

Q-learning fits a Q-network $\hat{Q}_ \phi$, with parameters $\phi$, to targets built from the
right-hand side, $y_i = r_i + \gamma \max_{\mathbf{a}'} \hat{Q}_ \phi(\mathbf{s}'_ i, \mathbf{a}')$,
so "When we optimize" that step, "we are trying to make this equation hold!" (Lecture 6, slide 12,
≈24:56–25:42). Since the max does not depend on which policy collected the data, the targets can be
computed from any transition, which is why Q-learning is off-policy (slide 13, ≈28:48).

![Lecture 6, slide 12 — From actor-critic to critic only](../raw/images/06-q-learning/slide-12.jpg)

*Lecture 6, slide 12 — the Q-learning algorithm with its max target (blue), and the two equations it rests on, colour-matched to the terminology box: the Bellman equation (orange-red) and the Bellman optimality equation (magenta). [Deck](https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf)*

## Iterating the Bellman equation

The Q-learning tutorial (Lecture 19) writes the optimal Q-function's Bellman equation with the
expectation over the next state as a sum over the dynamics, and turns it into **Q-value iteration**
(slide 6, ≈4:41–5:28):

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

The slide puts $a_t$ under the max, though the maximization is over the next action $a_{t+1}$.
Starting from $Q_0 = 0$ everywhere, each iteration propagates the terminal rewards one step further
back (≈10:09). Asked how this differs from the Bellman equation, Singh says the equation itself has no
iteration subscript and only states a relation the optimal Q-function satisfies; the subscripts $k$
and $k + 1$ make it an update (≈7:47–8:33). A heuristic for telling dynamic programming from Monte
Carlo: "if you see the Q function on both sides of your update" it is dynamic programming, while a Monte
Carlo update has only rewards on the right (≈8:33).

## Names, and the value-function versions

The first equation is the Bellman equation, the one used for policy evaluation; the second is the
Bellman optimality equation, because it is specifically for the optimal policy. People sometimes
informally call the second a Bellman equation too (Lecture 6, ≈26:29). Equations like these can also
be derived for the value function $V$ instead of the Q-function, and they go by the same names
(≈26:29–27:15); the lecture does not write them out. Lecture 4's relation
$Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = r(\mathbf{s}_ t, \mathbf{a}_ t) + \mathbb{E}_ {\mathbf{s}_ {t+1}} \left[ V^\pi(\mathbf{s}_ {t+1}) \right]$
is the one-step link between the two functions (slide 14; see
[value functions](value-functions.md#how-the-three-relate)).

Q is the version the course builds algorithms on because a policy comes straight out of it: the
$\arg\max$ over actions. A value function alone would need a model of the dynamics to say which
action leads to the better next state (Lecture 6, ≈23:24–24:56).

## Where this appears

- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md#value-q-and-advantage-functions): the
  one-step relation between $Q^\pi$ and $V^\pi$.
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md#fitting-a-q-function-with-data-from-past-policies):
  the recursive form of $Q^\pi$, derived on the board and used to fit a Q-function from a replay
  buffer.
- [Lecture 6 — Q-Learning](06-q-learning.md#why-it-works-the-bellman-optimality-equation): the
  Bellman optimality equation, the two names, and why Q rather than V.
- [Lecture 7 — Offline RL](07-offline-rl.md#advantage-weighted-actor-critic-awac): a Q-function for the
  behavior policy, bootstrapped with next actions from the data.
- [Lecture 19 — Tutorial Session: Review of Q-Learning](19-q-learning-tutorial.md#from-the-bellman-equation-to-q-value-iteration):
  the Bellman equation as a sum over the dynamics, Q-value iteration on a grid world, and the
  dynamic-programming heuristic.
