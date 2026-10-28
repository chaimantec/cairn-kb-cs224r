# RL algorithm families

Every algorithm in CS224R is a way of maximizing the same [objective](rl-objective.md), the
expected sum of rewards
$\mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$.
Lecture 1 sorts them into five families (slide 41, ≈49:20–50:53).

| Family | Idea (Lecture 1, slide 41) | Lecture in the Spring 2025 schedule |
| --- | --- | --- |
| Imitation learning | mimic a policy that achieves high reward | 2; see [imitation learning](imitation-learning.md) |
| Policy gradients | directly differentiate the objective | 3 |
| Actor-critic | estimate the value of the current policy and use it to make the policy better | 4, 5 |
| Value-based | estimate the value of the optimal policy | 6 |
| Model-based | learn to model the dynamics, and use it for planning or policy improvement | 11 |

The last column comes from the [course schedule](../sources.md#lectures), not from Lecture 1;
the schedule also has lectures on offline RL, reward learning, RL for language models,
multi-task and meta-RL, exploration, hierarchy and RL for robots. In the lecture's words, policy
gradients learn directly from practice and experience; value-based methods use the estimated
value of the optimal policy to back out a good policy; and model-based methods learn to predict
the future and use that for planning or to improve the policy (≈50:07). The
[value function and Q-function](value-functions.md) are the "value" these families estimate.

**This knowledge base so far covers Lectures 1 and 2.** Imitation learning has its own page,
[imitation learning](imitation-learning.md); the other four families do not yet. Lecture 2 adds
two points about where imitation learning sits. It is fully **offline** and needs no reward
function, but it may need a great deal of data and gives the policy no way to improve on its own
from practicing; its summary slide notes that "Many successful methods **combine** imitation
learning and reinforcement learning!" (Lecture 2, slide 25). And the offline/online distinction it
introduces is a general way to classify algorithms; see
[offline and online algorithms](offline-and-online.md).

## Why so many algorithms?

Supervised learning has gradient descent, which is simple and works. RL has many algorithms
because they make different trade-offs and thrive under different assumptions (slide 42,
≈50:53–51:39):

- **How easy or cheap is it to collect data with the policy?** Data from a simulator is very
  cheap. Data from interacting with a person who writes responses is expensive.
- **How easy or cheap are different forms of supervision?** Are there demonstrations? Detailed
  rewards?
- **How important are stability and ease of use**, compared with data efficiency? With plenty of
  data, there may be time to tune hyperparameters.
- **What does the action space look like?** How high-dimensional is it? Is it continuous or
  discrete?
- **Is it easy to learn the dynamics model?**

The lecture says the course will return to which algorithm suits which setting in its later
parts.

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#types-of-algorithms-and-why-there-are-so-many):
  the five families and the trade-offs between them.
- [Lecture 2 — Imitation Learning](02-imitation-learning.md): the first family in depth.
