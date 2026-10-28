# Actor-critic methods

Actor-critic methods are a family of online RL algorithms that build on
[policy gradients](policy-gradients.md). They learn to estimate how good each action was, and then
do more of the actions estimated to be good. Lecture 4 introduces them and says they are "the basis
for algorithms like PPO, which are used to train language models, and that are used to train
humanoids in simulation" (≈3:08). Symbols are as on [MDPs and RL notation](mdp-and-notation.md):
$\mathbf{s}$ is a state, $\mathbf{a}$ an action, $r$ the reward, and $\pi_\theta$ a
[policy](policies.md) with parameters $\theta$.

## The idea in one line

The lecture's review puts it this way (Lecture 4, slide 23, ≈1:02:21). Policy gradients "do more of
the good stuff, less of the bad stuff, based on what you observe". Actor-critic methods "are
learning to estimate what's good and bad, and then doing more of the good stuff according to your
estimate".

## From the policy gradient to the advantage

The policy gradient weights the log-probability of each sampled action by the **reward to go**, the
sum of rewards that followed the action in one sampled trajectory. That is a single-sample estimate
of the action's future rewards, and it is noisy. A humanoid that takes a small step forward and then
falls backwards has a low reward to go after the step, so the gradient pushes down the step forward,
even though the step was making progress. And under a sparse reward, trajectories that make partial
progress, such as folding only a jacket's sleeves, contribute nothing. "Policy gradients doesn't make
efficient use of data!" (Lecture 4, slide 8, ≈15:27–17:46).

Two substitutions fix the weight (Lecture 4, slides 9–10, ≈18:33–22:22). First, replace the
sampled reward to go with its expectation, the [Q-function](value-functions.md)
$Q^\pi(\mathbf{s}, \mathbf{a})$: "this would be way better" (≈19:18). Second, keep a baseline,
but make it the expectation of $Q^\pi$ over the policy's own actions, which is the value function
$V^\pi(\mathbf{s})$. The weight then becomes the **advantage**
$A^\pi(\mathbf{s}, \mathbf{a}) = Q^\pi(\mathbf{s}, \mathbf{a}) - V^\pi(\mathbf{s})$, how much better
the action is than following the policy:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \thinspace A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

Here $J(\theta)$ is the [RL objective](rl-objective.md), $N$ the number of sampled trajectories, $T$
the horizon, and $i$ and $t$ index trajectories and time steps. This still optimizes the same
objective, the expected sum of rewards. With the true advantage it is an unbiased estimate of the
gradient, and "better estimates of $A$ lead to less noisy gradients!" (slide 10; ≈23:55–25:27).

## The critic: a learned value function

The true advantage is unknown, so it has to be estimated. The online loop gains a third step
between running the policy and improving it: fit a model that estimates the expected return
(Lecture 4, slides 11–12, ≈25:27–27:00).

![Lecture 4, slide 12 — Online RL Outline](../raw/images/04-actor-critic/slide-12.png)

*Lecture 4, slide 12 — the online loop with its new third step: a teal box that fits a model to estimate the expected return, between collecting data and improving the policy. [Deck](https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf)*

The lecture fits only the value function, because the advantage can be written in terms of it
(Lecture 4, slide 14, ≈27:00–31:43). The Q-value is the current reward plus the value of the next
state, in expectation over the dynamics. Using the one next state $\mathbf{s}_ {t+1}$ actually
observed in the data gives

$$A^\pi(\mathbf{s}_ t, \mathbf{a}_ t) \approx r(\mathbf{s}_ t, \mathbf{a}_ t) + V^\pi(\mathbf{s}_ {t+1}) - V^\pi(\mathbf{s}_ t)$$

So one network $\hat{V}^\pi_\phi(\mathbf{s})$, with parameters $\phi$, which takes a state and
outputs a single scalar, is enough (≈34:00). Fitting it is **policy evaluation**: Monte Carlo
estimation, bootstrapping or n-step returns, with a discount factor. See
[policy evaluation](policy-evaluation.md).

## The algorithm

The full algorithm has five steps (Lecture 4, slide 22, ≈1:00:04–1:01:36).

1. Sample a batch of trajectories from $\pi_\theta$.
2. Fit $\hat{V}^{\pi_\theta}_ \phi$ to the summed rewards in the data. The lecture's choice is the
   discounted n-step target, with multiple gradient steps on the regression loss.
3. Evaluate the advantage of every action in the data,
   $\hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) = r(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) + \gamma \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t+1,i}) - \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t,i})$,
   where $\gamma$ is the discount factor.
4. Evaluate the policy gradient with $\hat{A}^{\pi_\theta}$ as the weight.
5. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$, with step size $\alpha$, and
   repeat.

There are two networks, and they give the method its name. "This is called an actor-critic
algorithm, because one of the networks is the actor, and one of the networks is the value
function, which can also be called a critic, because it's criticizing how good a state and policy
is" (≈1:01:36).

![Lecture 4, slide 22 — A Full Algorithm Walkthrough](../raw/images/04-actor-critic/slide-22.jpg)

*Lecture 4, slide 22 — the full algorithm in five steps, colour-keyed to the three-step loop, with the two networks labelled actor (the policy) and critic (the value function). [Deck](https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf)*

This answers the falling humanoid. From data in which it steps forward and then falls, the critic can
learn, especially with bootstrapped estimates, that the step forward has a positive advantage
compared with other actions. The algorithm therefore makes much more efficient use of the data it
collects (≈1:02:21).

## Off-policy versions (slides only)

The algorithm above is on-policy: it needs a fresh batch from the current policy for each update.
Lecture 4's deck has a section on making it off-policy, but **the recording ends before it**, so
what follows is what the slides print, with no spoken explanation. See the
[lecture page](04-actor-critic.md#not-in-the-recording-off-policy-actor-critic) and the
[slide file](../raw/slides/04-actor-critic.md).

**Version 1: multiple gradient steps** (Lecture 4, slides 25–27). Weight each term of the policy
gradient by the importance ratio
$\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) / \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$,
as the off-policy policy gradient does (see [importance sampling](importance-sampling.md)), so that
several gradient steps can be taken on one batch. The advantages are still the old policy's and
"will get out-dated". The deck asks what goes wrong with too many gradient steps and offers two
ideas: a KL constraint on the policy, and bounding the importance weights, which is the
"Key idea behind proximal policy optimization (PPO)".

**Version 2: replay buffers** (Lecture 4, slides 28–34). Keep a **replay buffer** of all past data
and train on minibatches drawn from all of it. Done naively, "The algorithm is currently broken".
The value target is not right for the current policy, and the buffer's actions are not the ones
the current policy would have taken. The fixes switch the critic to a Q-function, fitted to targets
$y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$ with the next action
$\mathbf{a}'_ i$ sampled from the current policy. They also compute the policy gradient on a freshly
sampled action $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$, weighted in
practice by $\hat{Q}^\pi$. One problem is accepted rather than fixed: the buffer's states do not
come from the current policy's state distribution. The slides' example of a practical algorithm is
soft actor-critic (SAC).

**More or less off-policy?** (Lecture 4, slide 35.) Learning curves on two humanoid tasks compare
SAC, labelled more off-policy (with a replay buffer), with PPO, labelled less off-policy (no
replay buffer). The slide's verdict is that off-policy methods with a replay buffer "can be far
more data efficient", but "can also generally be harder to tune hyperparameters, less stable (than
e.g. PPO)".

![Lecture 4, slide 35 — More off-policy vs. less off-policy actor critic?](../raw/images/04-actor-critic/slide-35.jpg)

*Lecture 4, slide 35 — learning curves on two humanoid tasks: SAC (orange, with a replay buffer) learns far faster than PPO (brown, no replay buffer) on the second task, at the cost of being harder to tune. [Deck](https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf)*

## Where this appears

- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md): the whole method, from the advantage to
  the full algorithm, with the off-policy slides that the recording does not reach.
- [Lecture 1 — Class Intro](01-class-intro.md#types-of-algorithms-and-why-there-are-so-many):
  actor-critic as the family that estimates the value of the current policy and uses it to improve
  the policy; see [RL algorithm families](rl-algorithm-families.md).
- [Lecture 3 — Policy Gradients](03-policy-gradients.md#review-and-next-time): the method it builds
  on.
