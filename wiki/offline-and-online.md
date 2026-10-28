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

Lecture 4 opens by recalling this limit: one gradient step per batch means collecting "a ton and
ton of data", and importance weights are "not a silver bullet", suitable only when the policies are
very similar (slide 4, ≈1:37–3:08). Its deck then makes [actor-critic methods](actor-critic.md)
increasingly off-policy, in slides the recording does not reach (Lecture 4, slides 25–35) and that
Lecture 5 teaches. The
first step takes multiple gradient steps on one batch with importance weights. The second keeps a
**replay buffer** of all past data and trains on minibatches from it, which forces changes to the
equations: a Q-function critic, and actions resampled from the current policy rather than taken
from the buffer. The deck's comparison says off-policy methods with a replay buffer, such as soft
actor-critic, "can be far more data efficient", but "can also generally be harder to tune
hyperparameters, less stable (than e.g. PPO)" (slide 35).

Lecture 5 lays this out as a spectrum (slide 14, ≈46:19). One batch of policy data for one gradient
step is fully on-policy. One batch for several gradient steps is "starting to be off-policy"; that is
[PPO](ppo.md), which keeps the new policy close to the old one so the batch stays useful. Reusing
"all of the past trial-and-error data" from a [replay buffer](replay-buffers.md) is more
off-policy still, and leads to soft actor-critic. The more off-policy method is far more data
efficient but harder to tune and less stable (Lecture 5, slide 23). Even it does not escape the
data entirely: a Q-function fitted from the buffer needs data covering actions similar to the
current policy's (slide 18), and its states come from past policies' state distributions (slide 21).

Lecture 6's [Q-learning](q-learning.md) is off-policy by construction (slides 13–14, ≈28:48).
Its target, $r + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}')$, does not involve the
policy that collected the transition, so the data can and should come from an exploration policy
broader than the current, greedy one, to cover the actions the max considers. It is called
off-policy because its updates use a replay buffer of essentially all past experience; some of that
data is from the current policy, but not only that data (≈36:27–37:14). Even a buffer of size one
is off-policy when the collecting policy differs from the current one (≈38:45). Its n-step version
is the exception that proves the rule: summed rewards come from the policy that collected them, so
an n-step target is "only actually correct when learning on-policy" (slide 25). Lecture 6 ends the
course's online RL methods (slide 29).

The course schedule has a later lecture on offline RL (Lecture 7 in this KB's numbering; see
[`sources.md`](../sources.md#lectures)). This knowledge base does not cover it yet.

## Where this appears

- [Lecture 2 — Imitation Learning](02-imitation-learning.md#summary-of-part-1-offline-and-reward-free):
  the definitions, and why behavior cloning is offline.
- [Lecture 3 — Policy Gradients](03-policy-gradients.md#the-trouble-with-on-policy): offline and
  online recalled (slide 4), and the on-policy and off-policy definitions.
- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md#not-in-the-recording-off-policy-actor-critic):
  the on-policy limit recalled, and off-policy actor-critic with replay buffers (slides only).
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md): the spectrum from one
  gradient step per batch, through PPO, to a replay buffer of all past data.
- [Lecture 6 — Q-Learning](06-q-learning.md#what-data-to-collect-epsilon-greedy-and-boltzmann-exploration):
  why Q-learning is off-policy, the questions on what that means, and why n-step targets are not.
