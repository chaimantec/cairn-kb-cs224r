# Policy gradients

Policy gradients improve a policy by differentiating the RL objective directly and taking gradient
steps on the policy's parameters. In CS224R they are the first online reinforcement learning
algorithm, and the basis for popular algorithms used to train legged robots and language models
(Lecture 3, ≈2:27). This page collects the method as Lecture 3 develops it. Symbols are as
on [MDPs and RL notation](mdp-and-notation.md): $\mathbf{s}_ t$ and $\mathbf{a}_ t$ are the state
and action at time $t$, $\tau$ is a trajectory of length $T$, $r$ is the reward function, and
$\pi_\theta$ is the policy with parameters $\theta$.

## The idea in one line

Sample trajectories with the current policy, then **increase the likelihood of the actions taken
in high-reward trajectories and decrease the likelihood of those taken in negative-reward
trajectories**: "do more of the good stuff, less of the bad stuff" (Lecture 3, slide 11). Finn
calls it a formalization of trial and error (≈26:42).

## The online loop

Policy gradients follow the outline of every online RL algorithm (Lecture 3, slide 6,
≈2:27–4:47). Initialize the policy (randomly, with imitation learning, or with heuristics), then
repeat: run the policy to collect a batch of data, and improve the policy using that batch. Over
iterations, scattered trajectories with mixed outcomes become a bundle of successes.

![Slide 6 — Online RL Outline](../raw/images/03-policy-gradients/slide-6.jpg)

*Slide 6 — the online RL loop, with three iterations of a simple navigation task: scattered trajectories with mixed crosses and checks become a tight bundle of successes. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

## Deriving the gradient

The objective is the expected sum of rewards, $J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} [r(\tau)]$,
where $r(\tau)$ is shorthand for $\sum_{t=1}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$ and
$p_\theta(\tau)$ is the distribution of trajectories the policy produces (see
[the RL objective](rl-objective.md)). $\theta$ affects $J$ only through which trajectories get
sampled, and that sampling, which also runs through the unknown dynamics, cannot be differentiated
(Lecture 3, ≈7:06). The derivation gets around this in three steps (slides 8–9, ≈7:52–16:34).

1. **Write the expectation as an integral**, $J(\theta) = \int p_\theta(\tau) \thinspace r(\tau) \thinspace d\tau$,
   and move the gradient inside it, since integration is linear.
2. **Use the convenient identity**
   $p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) = \nabla_\theta p_\theta(\tau)$, which
   follows from the derivative of a log and the chain rule. Substituting it turns the integral back
   into an expectation under the policy's own trajectories:
   $\nabla_\theta J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace r(\tau) \right]$.
   Now $\theta$ sits inside the expectation, which can be estimated with samples.
3. **Expand the log-probability of a trajectory.** $\log p_\theta(\tau)$ is
   $\log p(\mathbf{s}_ 1)$ plus a sum over time of $\log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$
   and of the log-dynamics. The initial state distribution and the dynamics do not depend on
   $\theta$, so their gradients vanish, and the unknown dynamics never need to be known.

The result is the **policy gradient**, or **vanilla policy gradient**:

$$\nabla_\theta J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right) \right]$$

Each $\nabla_\theta \log \pi_\theta$ is a backward pass through the log of the network's output for
the action taken, which automatic differentiation computes (≈14:57, ≈21:16). The full derivation,
step by step, is on the [Lecture 3 page](03-policy-gradients.md#deriving-the-policy-gradient).

## REINFORCE

With $N$ sampled trajectories, indexed by $i$, the expectation becomes an average, and the
estimate drives a gradient step with step size $\alpha$ (Lecture 3, slide 10, ≈18:10–20:31):

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) \right) \qquad \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

The algorithm loops over three steps: sample trajectories $\lbrace \tau^i \rbrace$ from
$\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$, estimate the gradient from them, and update
$\theta$. Slide 10 names it the **REINFORCE algorithm**, or vanilla policy gradient. More samples
give a better estimate; only infinitely many would make it exact (≈20:31).

![Slide 10 — Estimating the gradient](../raw/images/03-policy-gradients/slide-10.jpg)

*Slide 10 — the REINFORCE algorithm: the blue-underlined sample average is the "run policy" half of the loop, and the pink-underlined update is the "improve policy" half. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

## Imitation, weighted by reward

The first factor of the policy gradient is exactly the gradient of the imitation-learning
(behavior cloning) objective, which maximizes the log-probability of demonstrated actions. So the
policy gradient is the **imitation gradient weighted by reward**, applied to the policy's own data
rather than an expert's (Lecture 3, slide 11, ≈21:16–22:50). The two differences from
[imitation learning](imitation-learning.md) are that the policy collects its own rollouts, and that
the reward says how good each one was, where imitation learning imitates everything and assumes the
data is good (≈23:36). A negative reward gives a negative gradient that pushes the policy away from
what it did.

![Slide 11 — What does the gradient mean?](../raw/images/03-policy-gradients/slide-11.jpg)

*Slide 11 — the policy gradient beside imitation learning's gradient, with the shared term shaded teal: the policy gradient is the imitation gradient weighted by each trajectory's reward. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

The two can be combined: starting from an imitation-learned policy and then switching to policy
gradients can make sense, since a randomly initialized policy does random things, which is only
workable with a very dense reward (≈24:23–25:09).

## Its weaknesses, and the two fixes

Lecture 3 works through three examples of what the gradient actually does, and they expose three
problems.

**It is noisy, or high-variance** (slide 12, ≈29:02–29:48). A humanoid rewarded for forward
velocity samples five poor trajectories, and only "falls forwards" earns a positive total. The
gradient then encourages falling forward and discourages taking steps at all. A batch that happened
to include some walking would give a very different gradient.

![Slide 12 — What does the gradient do?](../raw/images/03-policy-gradients/slide-12.jpg)

*Slide 12 — the first example: a simulated humanoid rewarded for forward velocity, and five sampled trajectories, of which only "falls forwards" has a positive total. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

**Fix 1: causality.** "Policy behavior at time $t$ does not affect rewards at time $t' \lt t$"
(slide 13), so weight each action only by the **sum of future rewards**, moving the reward sum
inside the sum over time (≈30:34–34:34):

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right)$$

An action can still be blamed for what happens after it, which is right: a step forward might have
caused the fall that followed, and learning those consequences from experience is a large part of
RL (≈34:34–35:20).

**It is sensitive to the scale of the reward** (slide 14, ≈39:16–41:35). When every sampled
trajectory has positive reward (falling, stumbling, walking and running forwards), every one is
reinforced, running most strongly. If every reward were negative, there would be no positive
gradients at all.

![Slide 14 — What does the gradient do?](../raw/images/03-policy-gradients/slide-14.jpg)

*Slide 14 — the second example: four trajectories that all move forward, so every total reward is positive and every one of them is reinforced, running most strongly. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

**Fix 2: a baseline.** Subtract a constant $b$, such as the average reward of the batch, from each
trajectory's reward (slides 15–16, ≈41:35–42:21):

$$\nabla_\theta J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace (r(\tau) - b) \right]$$

Below-average behavior now gets a negative gradient. This is allowed because
$\mathbb{E} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace b \right] = b \thinspace \nabla_\theta \int p_\theta(\tau) \thinspace d\tau = b \thinspace \nabla_\theta 1 = 0$:
a constant baseline is **unbiased**, leaving the gradient unchanged in expectation, and it can
reduce the gradient's **variance** (slide 16, ≈43:08–47:47). The same $b$ is used for a whole batch
and recomputed on each new batch as the policy improves (≈47:01, ≈51:39–52:26). Average reward is
not the optimal baseline, which has additional weighting terms, but it is "pretty good" (≈48:34).
The baseline and causality can be combined, with some care over the baseline, since different time
steps sum different numbers of rewards (≈51:39).

**It needs dense rewards and large batches** (slide 17, ≈49:21–50:52). A robot folding a jacket,
with reward 1, 0.5 or 0, samples three failures and one success. With a baseline, folding is
encouraged, but the three failures get identical gradients, including the trajectory that usefully
folded the sleeves. Had no success been sampled, the gradient would have been zero.

![Slide 17 — What does the gradient do? (baseline edition)](../raw/images/03-policy-gradients/slide-17.jpg)

*Slide 17 — the baseline example: a humanoid folding a jacket, a reward of 1, 0.5 or 0, and four trajectories of which only the last folds it. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

## Implementing it

Computing each $\nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$ on its own
takes $N \times T$ backward passes. Instead, write a **surrogate objective** whose gradient is the
policy gradient and differentiate it once (Lecture 3, slide 18, ≈52:26–53:56):

$$\tilde{J}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

The slide prints the reward sum outside the sum over $t$, a typo Finn corrects in the lecture
(≈53:11). The log-probability term is a cross-entropy loss for a discrete-action policy and
something like a squared error for a Gaussian policy; more sophisticated policies such as diffusion
can be used too. The slide calls the whole thing "weighted maximum likelihood". See
[expressive policy distributions](expressive-policy-distributions.md) for those policy classes.

## On-policy, and an off-policy version

The gradient is an expectation under the **current** policy's trajectories, so after every
gradient step the data must be recollected (Lecture 3, slide 20, ≈54:41–55:27). Vanilla policy
gradient is therefore **on-policy**: its update uses only data from the current policy. An
**off-policy** update can reuse data from other, past policies; see
[offline and online algorithms](offline-and-online.md#on-policy-and-off-policy).

[Importance sampling](importance-sampling.md) gives a policy gradient that is "a little" off-policy
(≈56:14): one that can take several gradient steps on the same batch. Weighting by the ratio of
whole-trajectory probabilities under the new and old policies involves a product of $T$ ratios,
which can become very small or very large for larger $T$ (slide 22). Taking the expectation over
time steps instead, and approximating the ratio of state probabilities as 1, gives the **common
final form** (slides 23–24, ≈1:00:08–1:01:40):

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

where $\pi_{\theta'}$ is the policy being updated and $\pi_\theta$ the one that collected the
samples.

![Slide 24 — Off-policy policy gradient](../raw/images/03-policy-gradients/slide-24.jpg)

*Slide 24 — the off-policy policy gradient's common final form; the orange arrow from the update back to the gradient step is the change that allows several gradient steps per batch. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

If the policy changes a lot before new data is sampled, the data no longer reflects the states the
policy will visit and the estimate becomes less accurate (slide 24). Slide 25 proposes
constraining the policy not to stray too far, with a bound $\delta$ on the expected KL divergence
between new and old policies,
$\mathbb{E}_ {\mathbf{s} \sim \pi_\theta} \left[ D_{KL}\left( \pi_{\theta'}(\cdot \mid \mathbf{s}) \Vert \pi_\theta(\cdot \mid \mathbf{s}) \right) \right] \leq \delta$,
but Lecture 3 skips it for time (≈1:01:40).

## When to use it

The lecture's own summary (slides 19 and 26): even with causality and a baseline the gradient is
noisy, so policy gradients work best with **large batches and dense rewards**. Lecture 3 closes
by saying that the next lecture, on actor-critic methods, builds closely on policy gradients and is
the basis for algorithms such as PPO (slide 28, ≈1:02:26). This knowledge base does not cover that
lecture yet.

![Slide 19 — Summary so far](../raw/images/03-policy-gradients/slide-19.jpg)

*Slide 19 — the summary of the on-policy algorithm, colour-keyed to the two halves of the online loop. [Deck](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf)*

## Where this appears

- [Lecture 3 — Policy Gradients](03-policy-gradients.md): the derivation, the three examples, the
  surrogate objective, and the off-policy version.
- [Lecture 1 — Class Intro](01-class-intro.md#types-of-algorithms-and-why-there-are-so-many):
  policy gradients as the family that differentiates the objective directly; see
  [RL algorithm families](rl-algorithm-families.md).
