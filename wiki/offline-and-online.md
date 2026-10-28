# Offline and online algorithms

Two distinctions that classify RL algorithms by where their data comes from. Lecture 2 introduces
offline versus online; Lecture 3 adds on-policy versus off-policy, which divides the online
algorithms.

Lecture 2 introduces a distinction it calls important across reinforcement learning, as a way to
classify algorithms (slide 16, ≈51:42–52:29):

- **offline**: using only an existing dataset, with no new data from the learned policy;
- **online**: using new data from the learned policy.

## Why it matters

An online algorithm must keep collecting data as it runs, by running the policy it is learning.
That can be **expensive**, and running a policy that is not fully trained can be **unsafe**: in a
medical domain, for example, you may not want to collect data from a policy that does not work yet
(Lecture 2, ≈52:29). An offline algorithm avoids both, but it can only be as good as the data it
was given.

## In imitation learning

Behavior cloning, training a policy to mimic a fixed set of demonstrations, is **fully offline**
(Lecture 2, slide 16): it needs no data from the policy, and no reward function either. Its cost is
that it may need a lot of data for reliable performance. DAgger and human-gated DAgger are
**online**: they run the learned policy and collect an expert's corrections from the states it
visits, which is what lets them address compounding errors, at the price of an interface for the
expert to intervene (slides 20, 21 and 25). See [imitation learning](imitation-learning.md).

## On-policy and off-policy

Lecture 3 adds a second distinction, an attribute of **online** RL algorithms (slide 20,
≈55:27–56:14):

- **on-policy**: the update uses only data from the current policy;
- **off-policy**: the update can reuse data from other, past policies.

The vanilla [policy gradient](policy-gradients.md) is on-policy. Its gradient is an expectation
over trajectories from the current policy $\pi_\theta$, and $\theta$ changes with every update,
so the data must be recollected after every single gradient step, when neural networks are
normally trained with thousands of them (Lecture 3, ≈54:41–55:27). An off-policy version, built with
[importance sampling](importance-sampling.md), can take several gradient steps on one batch; Lecture 3
calls it only "a little" off-policy (≈56:14). Its estimate becomes less accurate the further the
policy moves from the one that collected the data (slide 24).

The course schedule has a later lecture on offline RL (Lecture 7 in this KB's numbering; see
[`sources.md`](../sources.md#lectures)). This knowledge base does not cover it yet.

## Where this appears

- [Lecture 2 — Imitation Learning](02-imitation-learning.md#summary-of-part-1-offline-and-reward-free):
  the definitions, and why behavior cloning is offline.
- [Lecture 3 — Policy Gradients](03-policy-gradients.md#the-trouble-with-on-policy): offline and
  online recalled (slide 4), and the on-policy and off-policy definitions.
