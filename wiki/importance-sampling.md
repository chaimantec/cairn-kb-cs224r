# Importance sampling

Importance sampling estimates an expectation under one distribution using samples drawn from
another. Lecture 3 introduces it to make the policy gradient off-policy, and Finn notes that it is
useful well outside reinforcement learning (≈56:14).

## The technique

Suppose we want $\mathbb{E}_ {x \sim p(x)} [f(x)]$, the expected value of some function $f$ under a
distribution $p$, but we can only sample from a different distribution $q(x)$, called the
**proposal distribution** because it proposes the samples (Lecture 3, slide 21, ≈56:14–57:49).
Multiplying inside the integral by $q(x)/q(x)$, which is 1, gives

$$\mathbb{E}_ {x \sim p(x)} \left[ f(x) \right] = \int p(x) f(x) \thinspace dx = \int \frac{q(x)}{q(x)} p(x) f(x) \thinspace dx = \int q(x) \frac{p(x)}{q(x)} f(x) \thinspace dx = \mathbb{E}_ {x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$$

So sample from $q$, weight each sample's $f(x)$ by the ratio $p(x)/q(x)$, and average (≈57:49–58:34).

## The catches

- **Support.** "Important for $q$ to have non-zero support for high probability $p(x)$" (slide 21):
  wherever $p$ puts real probability, $q$ must be able to produce samples. Finn calls this the only
  real catch (≈59:22). What the lecture says about the case where $q(x)$ is zero is unclear in the
  recording.
- **Similarity.** The technique works best when $p$ and $q$ are very similar to each other
  (≈59:22).

## In policy gradients

In RL, $q$ is the old policy whose samples are in hand and $p$ the new policy after a gradient step
(≈57:01). With $\bar{p}(\tau)$ the trajectory distribution of a previous policy $\bar{\pi}$, the
objective becomes (slide 21, ≈58:34)

$$J(\theta) = \mathbb{E}_ {\tau \sim \bar{p}(\tau)} \left[ \frac{p_\theta(\tau)}{\bar{p}(\tau)} r(\tau) \right]$$

and in the ratio the initial-state and dynamics terms cancel, leaving only policy probabilities:

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \frac{\prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)}{\prod_{t=1}^{T} \bar{\pi}(\mathbf{a}_ t \mid \mathbf{s}_ t)}$$

That product of $T$ ratios is the trouble: multiplying many numbers above 1 makes it very large,
and many below 1 drives it to zero quickly, so it "can become very small or very large, for larger
$T$" (slide 22, ≈1:00:08–1:00:54). Taking the expectation over time steps instead gives one ratio
per step. That ratio should be over state-action pairs, which is hard to measure, so the ratio of
state probabilities is "often approximated as 1" (slide 23), an approximation Finn calls incorrect
but the best that can be done (≈1:00:54). The result is a per-step weight
$\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) / \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$
on each term of the policy gradient, where $\pi_{\theta'}$ is the policy being updated and
$\pi_\theta$ the one that collected the data. That is what lets the off-policy policy gradient take
several gradient steps on one batch; see
[policy gradients](policy-gradients.md#on-policy-and-an-off-policy-version).

## In actor-critic methods

Lecture 4 recalls the off-policy policy gradient with a caution: the math "is very simple and very
nice, but it's important to remember that this isn't a silver bullet" (≈2:22). It suits only
policies that are very similar. When they differ, the ratio gets very large or very small, and the
gradient estimate gets worse (slide 4, ≈3:08).

The deck's first off-policy [actor-critic](actor-critic.md) version puts the same per-step ratio in
front of each advantage-weighted term so as to take several gradient steps per batch (Lecture 4,
slides 25–27; the recording does not reach these slides). The deck names the risk of taking too
many steps: the advantages come from the old policy and "will get out-dated". It offers two
remedies. One is a KL constraint on the policy. The other is to bound the importance weights, which
"Doesn't directly constrain policy, but removes incentives" and is the "Key idea behind proximal
policy optimization (PPO)" (slide 27).

Lecture 5 teaches that version aloud (slides 7–10, ≈6:14–31:42). In practice the weighted gradient
is obtained from a **surrogate objective**, the importance-weighted sum of advantages, whose gradient
with respect to the new parameters $\theta'$ is the weighted policy gradient:

$$\tilde{J}(\theta') \approx \sum_{t,i} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$$

The ratio is where the trouble starts. The policy controls only its numerator; the denominator, the
old policy, is fixed, so the policy can drive the weight toward zero or make it very large,
potentially infinite if the old probability was tiny (≈23:15–24:01). Maximizing the surrogate, it
raises the probability of positive-advantage actions the old policy rarely took, moves far from the
old policy, and overfits advantages that were only valid for that policy. PPO's answer is to clip the
weight to between $1 - \epsilon$ and $1 + \epsilon$, typically 0.8 and 1.2, which removes the
incentive to push it further, and to take the minimum with the unclipped term; see
[proximal policy optimization](ppo.md).

## In Q-learning's n-step targets

Lecture 6 lists importance sampling as one way to repair an n-step Q-learning target (slide 25,
≈57:19). The target sums several observed rewards before bootstrapping, and off-policy those rewards
come from the policy that collected the data rather than the one being estimated; the rewards can be
weighted by importance weights to correct for that. The lecture does not develop it, and notes that
most commonly the problem is simply ignored. See [Q-learning](q-learning.md#making-it-work-three-tricks).

## Where this appears

- [Lecture 3 — Policy Gradients](03-policy-gradients.md#importance-sampling): the technique and the
  off-policy policy gradient built from it.
- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md#not-in-the-recording-off-policy-actor-critic):
  the caution that it is not a silver bullet, and importance weights in off-policy actor-critic,
  with the idea behind PPO (slides only).
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md#version-1-multiple-gradient-steps-with-importance-weights):
  importance weights for multiple gradient steps, the surrogate objective, and clipping them in PPO.
- [Lecture 6 — Q-Learning](06-q-learning.md#trick-3-n-step-returns): importance sampling as one fix
  for off-policy n-step targets.
