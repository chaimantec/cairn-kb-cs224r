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

## Where this appears

- [Lecture 3 — Policy Gradients](03-policy-gradients.md#importance-sampling): the technique and the
  off-policy policy gradient built from it.
