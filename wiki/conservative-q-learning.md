# Conservative Q-learning (CQL)

Conservative Q-learning is an [offline RL](offline-rl.md) method that keeps an ordinary actor-critic
structure, querying the learned policy's actions, and stops the critic from overestimating them by
changing its loss: Q-values that a search distribution finds to be large are pushed down, and the
Q-values of the dataset's own actions are pushed back up. Lecture 7's deck holds it (slides 24–28), but
that lecture's recording skips it. Lecture 8 opens by teaching it aloud from the same slides
(Lecture 8, slides 5–8, ≈6:57–21:39) and gives its LinkedIn application. Citations below are to
Lecture 8 unless they say otherwise; Lecture 7's slides 25–28 are the same four slides.

## The problem it addresses

An off-policy critic's target evaluates the Q-function on actions the learned policy samples, which
may lie outside the data. There, a randomly initialized Q-function is never corrected, the policy
seeks out the actions it happens to overrate, and the Q-values are overestimated (Lecture 8, slide 4,
≈0:51–1:36; the full story is on [offline RL](offline-rl.md)). The
[implicit policy constraint methods](implicit-policy-constraints.md), such as IQL, avoid querying
any action outside the data. CQL instead still estimates the value of the learned policy and queries
its actions, and discourages the RL algorithm from taking actions whose Q-values look high only
because they are out of distribution (≈6:57–7:43). The slides ask how to discourage overestimation
"while still estimating value of the learned policy" (slide 5).

## Step one: push down on large Q-values

The idea is to minimize Q-values outside the data's support, so that the critic is pessimistic, or
conservative, about everything out of distribution (≈7:43–8:29). Add to the usual critic loss a term
that minimizes Q-values on actions drawn from some other distribution $\mu$, for states from the
dataset $D$. The actions worth pushing down are the out-of-distribution ones the policy might
consider, and the policy considers those with high Q-values, so $\mu$ is chosen to maximize the
Q-values (≈8:29–10:50). With $\hat{Q}^\pi$ the learned Q-function, $\pi$ the policy and $\alpha$ a
weight (slide 5):

$$\hat{Q}^\pi = \arg\min_Q \max_\mu \mathbb{E}_ {(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi} [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$$

The first term is the "standard critic update", the second will "push down on large Q-values"
(slide 5). With a large enough $\alpha$ the learned critic is a lower bound on the true Q-function,
$\hat{Q}^\pi \leq Q^\pi$ (slide 5, ≈10:50–11:35).

![Lecture 8, slide 5 — How to mitigate overestimation in offline RL?](../raw/images/08-reward-learning/slide-5.png)

*Lecture 8, slide 5 — the randomly initialized Q-function beside the first step toward CQL, the standard critic update plus a term that pushes down on large Q-values. [Deck](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf)*

## Step two: push back up on the data's actions

That version is too pessimistic: it pushes down on every Q-value, including those of actions inside
the data's support, where there is more data (≈11:35–12:20). CQL subtracts the same term evaluated on
the dataset's own states and actions, so it minimizes Q-values for actions from $\mu$ and maximizes
them for actions in the data (slide 6, ≈12:20–13:06):

$$\hat{Q}^\pi = \arg\min_Q \max_\mu \mathbb{E}_ {(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi} [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})] - \alpha \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim D} [Q(\mathbf{s}, \mathbf{a})]$$

The aim is a Q-function that is accurate inside the support of the data and low outside it, where an
accurate one cannot be learned (≈13:06). Once $\mu$'s search lands inside the distribution, the
push-up term cancels it there (≈14:37).

The pointwise bound is lost, but the policy's expected Q-value is still a lower bound for every state
in the dataset (slide 6):

$$\mathbb{E}_ {\pi(\mathbf{a} \mid \mathbf{s})} [\hat{Q}^\pi(\mathbf{s}, \mathbf{a})] \leq \mathbb{E}_ {\pi(\mathbf{a} \mid \mathbf{s})} [Q^\pi(\mathbf{s}, \mathbf{a})] \quad \text{for all } \mathbf{s} \in D$$

The lecture does not derive it. It needs $\alpha$ large enough and fails at $\alpha = 0$; the CQL paper
has the conditions and the proof (≈16:09–16:57).

![Lecture 8, slide 6 — How to mitigate overestimation in offline RL?](../raw/images/08-reward-learning/slide-6.png)

*Lecture 8, slide 6 — the CQL objective in three coloured parts, the standard critic update, pushing down on large Q-values and pushing up on Q-values for the data's own state-action pairs, with the lower bound it keeps. [Deck](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf)*

**Notation.** The slides minimize over the function $Q$. On the board in Lecture 8, Finn writes the
objective as a minimization over the learned Q-function's parameters, with the critic loss called
$L_\text{critic}$, and notes that "the notation on the slides is slightly different" (≈8:29, ≈13:52).

## The role of alpha

$\alpha$ is a hyperparameter that controls how pessimistic the critic is outside the data. CQL does
hurt generalization outside the support, by design. If you are in a regime where you might generalize
better, a smaller $\alpha$ leans more on the original critic objective and less on the regularization
(Lecture 8, ≈13:06–13:52). Lecture 7's advice slide says CQL has "just one hyperparameter" (Lecture 7,
slide 32).

## What mu is, and why it never has to be built

$\mu$ can be thought of "as sort of a policy", but it is never run: it seeks out actions with large
Q-values, in or outside the distribution, so that their Q-values get pushed down (≈14:37). Maximizing
over $\mu$ means picking one with high Q-values (≈15:23).

Building $\mu$ explicitly would add a moving part, and as written it could collapse onto the single
action with the largest Q-value (Lecture 8, slide 8, ≈17:45–19:18). Adding an entropy regularizer
$R(\mu) = \mathbb{E}_ {\mathbf{s} \sim D}[\mathcal{H}(\mu(\cdot \mid \mathbf{s}))]$ keeps it broad, and
then the optimal $\mu(\mathbf{a} \mid \mathbf{s})$ is proportional to $\exp(Q(\mathbf{s}, \mathbf{a}))$, in
closed form. The push-down term becomes a log-sum-exp (slide 8, ≈19:18–20:52):

$$\mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})] = \log \sum_{\mathbf{a}} \exp \left( Q(\mathbf{s}, \mathbf{a}) \right)$$

So the objective Finn states is the critic loss, plus $\alpha$ times the log-sum-exp of the Q-values,
minus $\alpha$ times the expected Q-values on the data, with no $\mu$ in it (≈20:52). With continuous
actions the log-sum-exp is estimated from samples (≈21:39).

![Lecture 8, slide 8 — How to mitigate overestimation in offline RL?](../raw/images/08-reward-learning/slide-8.jpg)

*Lecture 8, slide 8 — computing the CQL objective, with the circled max over mu, the entropy regularizer, and the log-sum-exp that replaces the push-down term. [Deck](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf)*

## The algorithm

Alternate two steps: update $\hat{Q}^\pi$ with the CQL loss $L_{CQL}$ on the dataset, then update the
policy (Lecture 8, slide 7, ≈16:57–17:45). With discrete actions the policy is greedy, taking
$\arg\max_{\bar{\mathbf{a}}} \hat{Q}^\pi(\mathbf{s}, \bar{\mathbf{a}})$ with probability one. With
continuous actions it takes a gradient step with step size $\eta$ toward actions with high Q-values:

$$\theta \leftarrow \theta + \eta \nabla_\theta \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \pi_\theta(\cdot \mid \mathbf{s})} \left[ \hat{Q}^\pi(\mathbf{s}, \mathbf{a}) \right]$$

Homework 3 implements CQL (slide 8).

## Why be conservative, and why not just behavior cloning

Being conservative goes back to offline RL's premise: with no way to collect more data, a policy
cannot be learned far beyond the data, and assuming it can risks an inaccurate Q-function and a poor
policy. So CQL is conservative specifically outside the data's support (Lecture 8, ≈23:13–23:58).

Behavior cloning also stays inside the data, but CQL, like any RL method, uses the reward to learn a
policy with high Q-values and can do considerably better than behavior cloning (≈15:23–16:09). The
implicit-constraint methods of Lecture 7 also beat behavior cloning; CQL is an alternative to them,
the third approach in both lectures' summaries: a "conservative objective" built by "penalizing
Q-values" (Lecture 8, slide 10; Lecture 7, slide 30).

## An application: LinkedIn notifications

Prabhakar, Yuan, Yang, Sun and Muralidharan (2022) trained a notification policy for LinkedIn offline
and tested it online (Lecture 8, slide 9, ≈21:39–23:13; also Lecture 7, slide 31). The goal is clicks
without spamming users, with weekly active users also in view. Against their existing system, double
DQN alone gave −0.69% weekly active users, +7.72% notification volume and −7.79% click-through rate;
double DQN with the CQL terms added gave +0.24% sessions, +0.18% weekly active users, −1.73% volume and
+2.26% click-through rate.

![Lecture 8, slide 9 — An example application](../raw/images/08-reward-learning/slide-9.jpg)

*Lecture 8, slide 9 — LinkedIn's notification pipeline and the online A/B results for double DQN with and without CQL. [Deck](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf)*

## The same idea for reward classifiers

The fix for a learned goal classifier that RL exploits, adding the states the policy visits as
negatives and retraining, is "very similar to CQL", in Finn's words (Lecture 8, ≈30:58): in both, the
regions the optimizer reaches beyond the data are pushed down. See
[reward learning](reward-learning.md).

## Where this appears

- [Lecture 8 — Reward Learning](08-reward-learning.md): CQL taught aloud, with the student questions,
  the LinkedIn application, and the reward-classifier parallel.
- [Lecture 7 — Offline RL](07-offline-rl.md): the same slides (24–28 and 31), not taught in that
  recording.
- [Offline RL](offline-rl.md): the setting, the overestimation problem, and the other approaches.
- [Implicit policy constraint methods](implicit-policy-constraints.md): IQL and its relatives, which
  avoid querying out-of-distribution actions instead of penalizing them.
- [Q-learning](q-learning.md): the greedy policy and double DQN that CQL modifies in the application.
