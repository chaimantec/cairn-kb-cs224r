# Offline and online algorithms

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

The course schedule has a later lecture on offline RL (Lecture 7 in this KB's numbering; see
[`sources.md`](../sources.md#lectures)). This knowledge base does not cover it yet.

## Where this appears

- [Lecture 2 — Imitation Learning](02-imitation-learning.md#summary-of-part-1-offline-and-reward-free):
  the definitions, and why behavior cloning is offline.
