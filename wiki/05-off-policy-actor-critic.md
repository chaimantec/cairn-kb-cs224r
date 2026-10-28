# Lecture 5 — Off-Policy Actor Critic

Stanford CS224R, Spring 2025 (Wednesday, April 16) ·
[video](https://www.youtube.com/watch?v=cRGKc-nAWho) (69 min) ·
[slides (PDF)](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf) ·
[slide-by-slide text](../raw/slides/05-off-policy-actor-critic.md) ·
[transcript](../raw/transcripts/05-off-policy-actor-critic.md)

**What this lecture establishes.** The [actor-critic](actor-critic.md) algorithm of Lecture 4 is
on-policy: each batch of data buys one gradient step. This lecture makes it off-policy in two
stages and arrives at the two algorithms people use in practice. The first stage takes **many
gradient steps per batch** by putting [importance weights](importance-sampling.md) into the
gradient. Too many such steps overfit advantages that were estimated for the old policy, so the
new policy has to be kept close to the old one, either with a KL penalty or by clipping the
importance weights. Clipping, a minimum that makes the objective a lower bound, and generalized
advantage estimation together give **proximal policy optimization (PPO)**. The second stage reuses
**all past data** from a **replay buffer**. That breaks the value function, which would be fitted
for a mixture of past policies, and the policy update, which would use actions the current policy
did not take. Both are fixed by fitting a **Q-function** to targets that use an action sampled from
the current policy, and by sampling the policy update's action the same way. The result is
essentially **soft actor-critic (SAC)**. It is far more data-efficient than PPO but harder to tune
and less stable, and the lecture closes with robots trained with each.

Slide numbers below are the numbers printed on the slides, which equal the PDF page numbers (slide
4 prints none). Timestamps (≈M:SS) point into the video and the
[transcript](../raw/transcripts/05-off-policy-actor-critic.md). **The recording covers slides
3–28.** It opens on the recap and ends at about 1:09:20 on slide 28. It does not discuss slide 2's
reminders or the plan recap and next-time slides (29–30); the
[last section](#not-in-the-recording-reminders-and-next-time) gives them from the
[slide file](../raw/slides/05-off-policy-actor-critic.md).

Several steps are worked on the board, which the recording does not show: a sketch of learning
curves, a grid-world sketch, a KL-penalized objective, the clipping written with max and min, and
the recursive form of the Q-function. Where the transcript says "this" or "right here", this page
reconstructs what was meant from the surrounding words and the slides, and says so.

## Contents

1. [Recap: policy gradients, value functions, actor-critic](#recap-policy-gradients-value-functions-actor-critic)
2. [Plan: PPO and SAC](#plan-ppo-and-sac)
3. [Version 1: multiple gradient steps with importance weights](#version-1-multiple-gradient-steps-with-importance-weights)
4. [The surrogate objective](#the-surrogate-objective)
5. [What goes wrong with too many gradient steps](#what-goes-wrong-with-too-many-gradient-steps)
6. [Idea 1: a KL penalty](#idea-1-a-kl-penalty)
7. [Idea 2: clip the importance weights](#idea-2-clip-the-importance-weights)
8. [PPO, trick 2: take the minimum](#ppo-trick-2-take-the-minimum)
9. [PPO, trick 3: generalized advantage estimation](#ppo-trick-3-generalized-advantage-estimation)
10. [The PPO algorithm and example hyperparameters](#the-ppo-algorithm-and-example-hyperparameters)
11. [Version 2: replay buffers](#version-2-replay-buffers)
12. [Why the replay-buffer algorithm is broken](#why-the-replay-buffer-algorithm-is-broken)
13. [Fitting a Q-function with data from past policies](#fitting-a-q-function-with-data-from-past-policies)
14. [Fixing the policy update](#fixing-the-policy-update)
15. [The remaining problem, and soft actor-critic](#the-remaining-problem-and-soft-actor-critic)
16. [More off-policy versus less off-policy](#more-off-policy-versus-less-off-policy)
17. [Examples: real robots with SAC, simulation with PPO](#examples-real-robots-with-sac-simulation-with-ppo)
18. [Not in the recording: reminders and next time](#not-in-the-recording-reminders-and-next-time)

## Recap: policy gradients, value functions, actor-critic

Online RL runs the policy to collect a batch of data, improves the policy with that batch, and
repeats, so that the approach gets better with experience (slide 3, ≈0:05). The policy gradient
takes $N$ trajectories sampled from the policy $\pi_\theta$, a distribution over actions
$\mathbf{a}$ given states $\mathbf{s}$ with parameters $\theta$, and raises the probability of
actions followed by high reward, minus a baseline $b$, the average reward:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

Here $J(\theta)$ is the [RL objective](rl-objective.md), the expected sum of rewards, $T$ the
horizon and $r$ the reward. This needs data sampled from exactly the policy being updated. The
**importance weight** $\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) / \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$,
the new policy's probability of the action over that of the policy that collected the data, is
what allows several gradient steps without collecting a new batch for each one (slide 3,
≈0:51).

![Slide 3 — Recap: Policy Gradients](../raw/images/05-off-policy-actor-critic/slide-3.png)

*Slide 3 — the online loop beside the policy gradient with its four parts labelled, and below it the off-policy version, whose importance weight (highlighted) is the ratio of the new policy to the old. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

The [value function](value-functions.md) $V^\pi(\mathbf{s})$ is the future expected reward
starting at $\mathbf{s}$ and following $\pi$; the Q-function $Q^\pi(\mathbf{s}, \mathbf{a})$
starts at $\mathbf{s}$, takes $\mathbf{a}$, and then follows $\pi$; and the advantage
$A^\pi(\mathbf{s}, \mathbf{a}) = Q^\pi(\mathbf{s}, \mathbf{a}) - V^\pi(\mathbf{s})$ is how much
better it is to take $\mathbf{a}$ than to follow the policy (slide 4, ≈0:51–3:11). The value
averages the summed rewards of trajectories that run the policy from the state; the Q-function
averages those of trajectories that first take $\mathbf{a}$:

$$V^\pi(\mathbf{s}_ t) = \sum_{t'=t}^{T} \mathbb{E}_ {\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t \right] \qquad Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{T} \mathbb{E}_ {\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right]$$

![Slide 4 — Recap: Some Useful Objects](../raw/images/05-off-policy-actor-critic/slide-4.jpg)

*Slide 4 — the value function averages trajectories that run the policy from s; the Q-function's trajectories first take the action a (green) and then run the policy. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

Finn adds an example from Go, recalling AlphaGo from Lecture 1 (≈3:11–4:43). Let the reward be
the outcome of the game, whether AlphaGo won or lost. Then the value function is the expected
outcome given the state of the board and the policy being run. A better policy has a higher value,
and so does a board state where things look good for that policy: the value depends on both the
policy and the state. A Q-function would consider a particular move first and then run the policy.

Actor-critic methods replace the policy gradient's sum of rewards with an estimate of how good the
action was, the advantage (slide 5, ≈4:43–6:14). Unlike policy gradients, they first estimate which
actions are good and bad and then encourage the policy to do more of what is estimated to be good. The
slide's words are "Estimate what is good and bad" and "do more of the good stuff."

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \thinspace A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

Estimating it is [policy evaluation](policy-evaluation.md): fit a value network
$\hat{V}^\pi_\phi$, with parameters $\phi$, that takes a state and outputs a scalar. Its target is
either the sum of rewards (Monte Carlo), the next reward plus the network's own estimate of the
next value (bootstrapping), or the sum of a short run of rewards plus the value after it (n-step
returns). Slide 5 prints the n-step version, a squared error against targets
$y_{i,t} = \sum_{t'=t}^{t+n-1} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) + \gamma^n \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+n})$,
where $\gamma$ is the discount factor. The advantage is then approximately the reward plus the next
state's value minus the current state's value (≈5:29).

![Slide 5 — Recap: Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-5.jpg)

*Slide 5 — the online loop with its middle step, fitting a model of the expected return, beside the advantage-weighted gradient, the n-step value target and the one-step advantage estimate. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## Plan: PPO and SAC

Lecture 4 mainly considered on-policy actor-critic methods. This lecture gets into two "real
algorithms that people use", PPO and SAC, starting from the actor-critic formulation and making it
off-policy to be more efficient. At the end it compares PPO, SAC and imitation learning
(≈6:14). The plan has two parts (slide 6). The first is taking multiple gradient steps: importance
weights, constraining the step size with a KL penalty or clipping, and the practical PPO
algorithm. The second is an even more off-policy algorithm: fitting Q-value functions with data
from other policies, and the practical SAC algorithm. The one learning goal is "All of the key
concepts for practical algorithms like PPO and SAC."

## Version 1: multiple gradient steps with importance weights

The on-policy actor-critic algorithm has five steps (slide 7, ≈6:14–7:47):

1. Sample a batch of trajectories from $\pi_\theta$.
2. Fit $\hat{V}^{\pi_\theta}_ \phi$ to the summed rewards in the data.
3. Evaluate the advantage for every state and action in the batch,
   $\hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) = r(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) + \gamma \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t+1,i}) - \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t,i})$.
4. Evaluate the gradient $\nabla_\theta J(\theta)$.
5. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$, where $\alpha$ is the step
   size.

After step 5 the parameters differ from those that collected the data in step 1, so a second
gradient step needs the gradient at the new parameters (≈7:00). Call the policy that collected the
data $\pi_\theta$ and the policy after a gradient step $\pi_{\theta'}$. The gradient at $\theta'$
reuses the data from $\pi_\theta$ and weights each term by the importance ratio, while the
advantage estimate stays the one for the old policy $\pi_\theta$ (slide 8, ≈7:47–9:20):

$$\nabla_{\theta'} J(\theta') \approx \sum_{t,i} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {t,i} \mid \mathbf{s}_ {t,i}) \thinspace \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$$

(The slides index the ratio $(i,t)$ and the other terms $(t,i)$, as Lecture 4's did; both mean
time step $t$ of trajectory $i$.) In principle, this allows sampling a batch once, fitting the
advantages, and then taking a gradient step, recomputing the gradient, taking another step, and so
on. Across those steps, only the importance weight changes, together with the parameters that
backpropagation runs through; the advantage estimates stay based on the old policy (≈9:20). A small
arrow on slide 8 runs from step 5 back to step 4, and the slide asks "What can go wrong?"

## The surrogate objective

To see what can go wrong, it helps to look at the **surrogate objective** (slide 9,
≈10:09–12:26). In practice, the gradient is not computed by hand. One writes down an objective
whose gradient equals the desired gradient and lets automatic differentiation in PyTorch produce
it. The tool is the identity behind the original policy gradient, which the slide calls "a
convenient identity":

$$p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) = \nabla_\theta p_\theta(\tau)$$

In the gradient above, the importance weight's numerator times $\nabla_{\theta'} \log \pi_{\theta'}$
is exactly such a product, so it equals $\nabla_{\theta'} \pi_{\theta'}$. Replacing that gradient
by the probability itself gives the surrogate objective $\tilde{J}$, whose gradient with respect to
$\theta'$ is the off-policy gradient:

$$\tilde{J}(\theta') \approx \sum_{t,i} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$$

Finn uses $\tilde{J}$ because it is "technically not the true objective"; the true $J(\theta)$ is the
expected sum of rewards (≈27:51).

Read as something the policy maximizes, the surrogate says what the policy will do (≈12:26–13:12).
For actions with positive advantage, it will raise their probability, and especially for actions
that had very low probability under the old policy, since raising them makes the ratio very large.
For negative advantages, it will lower the probability. The slide's annotations put it as "Policy
will increase probability on actions with high advantages" and "Advantages based on old policy."

## What goes wrong with too many gradient steps

The advantages are based on the old policy and were estimated from a single batch of data. Take
too many gradient steps on the surrogate objective and the policy can overfit them: they go out of
date and no longer apply to the new policy, and learning becomes unstable (≈13:12). The slide puts
it as "Policy is incentivized to differ significantly from old policy, can easily overfit" (slide 9).

Finn sketches learning curves on the board (≈13:58–15:29). The horizontal axis is the number of
state-action examples collected and the vertical axis the reward the policy sees. With one gradient
step per batch, learning is slow, since each step needs a new batch. With something like five
gradient steps, reward rises faster per data point. With far more gradient steps, reward may rise
very quickly at first, but learning becomes unstable and may even collapse, as the policy overfits
the advantages, perhaps pushing the probability of some action to one in a way that makes the
policy perform poorly.

Asked what an overfit policy looks like in practice, Finn says it depends on the problem
(≈15:29–17:47). It might assign a very high probability to one action with a high advantage in the
data, and, because neural networks are somewhat smooth, raise the probability of nearby actions
too, giving a very deterministic policy that no longer explores. In a second board sketch, a 2D
agent moves on a grid toward a goal; a good policy goes right and then up. An overfit policy might
learn that going right is good without learning that it must stop and go up, and so go in a straight
line and end up in the corner, because it overfit one batch of data that told it to go that way.
(A word at the end of this explanation is unclear in the recording, at ≈17:00.) The five gradient
steps of the sketch are five loops of steps 4 and 5 before collecting a new batch and refitting the
value function, which so far is refitted only on the current batch.

Further questions fill in the picture (≈17:47–21:39):

- **Why look at the surrogate objective rather than the gradient?** It is the same formulation, but
  Finn finds the surrogate more intuitive for seeing that the importance weights make it very easy
  to increase probability. PPO's gradient also looks more complicated than its surrogate.
- **Why use importance weights at all?** They are what allows more than one gradient step: without
  them, the gradient cannot be estimated accurately on the old batch.
- **What happens when the advantages are too old?** They come from a limited batch, perhaps a few
  thousand time steps, and are true only for the old policy. After something like 50 or 100 gradient
  steps they are out of date. In the grid example, the old policy may have gone up, so the
  advantages say going right is good because "later you will go up"; the new policy will do
  something different, and those advantages are no longer valid. A new batch from the new policy is
  needed for accurate advantages.

There is a second effect (≈20:06–20:52). In raising probabilities, the policy is also pushed to
differ drastically from the old policy, and the more it changes, the less likely the advantages are
to fit it. So the key idea is to keep the new policy close to the original one, so that the
advantage estimates stay more likely to be valid (≈20:52). The policies are distributions over
actions, and there are two ways to encourage closeness (slide 10, ≈21:39).

![Slide 10 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-10.png)

*Slide 10 — the surrogate objective with its two annotations (actions with high advantages get pushed up; the advantages belong to the old policy), and the two fixes: a KL constraint, and bounding the importance weights, the idea behind PPO. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## Idea 1: a KL penalty

The first way adds a term to the surrogate objective that explicitly encourages the two policies to
be similar (≈21:39–23:15). On the board, Finn writes a new objective equal to the original surrogate
objective, to be maximized, minus a penalty for changing too much from the previous policy, the
**KL divergence** between the new policy's distribution and the old one's. Later answers name the
weight on that penalty $\beta$, a hyperparameter (≈25:33–26:19). From those spoken descriptions,
the board objective has the form

$$\tilde{J}(\theta') - \beta \thinspace D_{KL}(\pi_{\theta'} \thinspace\Vert\thinspace \pi_\theta)$$

The board itself is not visible in the recording, and the name Finn gives the new objective is
unclear (≈22:29). Slide 10 states the idea as a constraint instead, with $\delta$ the allowed
divergence:

$$\mathbb{E}_ {\mathbf{s} \sim \pi_\theta} \left[ D_{KL}(\pi_{\theta'}(\cdot \mid \mathbf{s}) \thinspace\Vert\thinspace \pi_\theta(\cdot \mid \mathbf{s})) \right] \leq \delta$$

Keeping the policies close while taking more gradient steps keeps the advantages more accurate, and
so allows more gradient steps stably. It still constrains the step size, so it is a balance; on the
board sketch, more gradient steps with a KL penalty give a curve that rises faster without collapsing
(≈23:15).

The questions on it fill in the details (≈23:15–28:37):

- **Why does the ratio need controlling?** As a student points out, the policy controls only the
  numerator; the denominator, the old policy, is fixed. So the policy can push the weight toward zero
  or make it very large, potentially infinite if the old probability was tiny. Constraining the
  difference keeps the ratio closer to one.
- **Memory.** The KL term needs the old parameters kept in memory, which is annoying and raises
  memory costs. With large language models, "you'll have to keep around the full set of weights for
  your old language model". One can instead cache the probabilities the old policy assigned for the
  batch of data.
- **The starting ratio.** The ratio starts at one and diverges only after the first gradient step.
- **Why change a little at all?** A policy taking images as input needs many gradient steps to
  change its image features, so one wants many steps without substantially changing the action
  distribution.
- **Choosing $\beta$.** It controls how much the policy may shift and is often a fairly important
  hyperparameter, but people usually tune it within an order of magnitude rather than very
  precisely.
- **Implementing it.** The surrogate objective is what goes into PyTorch, TensorFlow or JAX, which
  compute its gradient. To add the penalty, look up the KL divergence between two categorical
  distributions or two Gaussians, which for Gaussians is a function of the means and variances,
  and add it to the surrogate.

A KL constraint is "very common" (the slide adds "Will see in LLM preference optimization"), and
there is a version of PPO that uses one, though not the most common version (≈27:51). The lecture
leaves it there.

## Idea 2: clip the importance weights

The second way removes the incentive to change the ratio a lot (≈28:37–31:42). Clip the importance
weight so that it never goes above $1 + \epsilon$ or below $1 - \epsilon$, for some $\epsilon$. A
value used in practice is about $0.2$, which confines the weight to between 0.8 and 1.2: "it's not
tiny", unlike the tiny numbers $\epsilon$ often names (≈30:56). Written out on the board, with
$w$ the importance weight, clipping below by $1 - \epsilon$ takes $\max(w, 1 - \epsilon)$ and
clipping above by $1 + \epsilon$ then takes the minimum of that and $1 + \epsilon$ (≈29:24–30:11):

$$\mathrm{clip}(w, 1 - \epsilon, 1 + \epsilon) = \min(\max(w, 1 - \epsilon), 1 + \epsilon)$$

With clipped weights in the surrogate objective, once the policy has pushed the ratio above 1.2,
raising the probability further no longer helps: the objective stays at 1.2 times the advantage.
Clipping does not explicitly constrain the policy; it removes the incentive to move far from the old
one (≈30:11–31:42). Bounding the importance weights is one of the key ideas behind PPO, a popular RL
algorithm.

## PPO, trick 2: take the minimum

**Proximal policy optimization (PPO)** is "Off-policy actor-critic with a few tricks" (slide 11).
Trick 1 is the clipped surrogate objective, after which the "Policy no longer incentivized to
deviate significantly" (≈31:42):

$$\tilde{J}(\theta') \approx \sum_{t,i} \mathrm{clip} \left( \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$$

Trick 2 deals with a funny property of clipping: in some rare cases, clipping can make the objective
better than it would have been (≈32:28). PPO's final objective takes, for each data point, the
minimum of the clipped term and the original unclipped term, "(in rare event where clipping makes
objective better)" (slide 11):

$$\tilde{J}(\theta') \approx \sum_{t,i} \min \left( \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}), \mathrm{clip} \left( \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) \right)$$

The minimum ensures clipping never makes the objective better, so the result is a **lower bound** on
the original surrogate objective. If you are going to maximize something other than your original
objective, it is good for it to be a lower bound, because then maximizing it cannot overshoot the
original (≈32:28–33:13). The slide calls this "the final PPO surrogate objective!"

Asked for the motivation behind trick 2, Finn is not sure how often it matters in
practice (≈33:13–34:48). Suppose the policy raised the probability of a lot of the good actions and
lowered it for the bad ones, but in raising a good action it also raised a bad action in a similar
state. Clipping that bad action's weight at 1.2 then makes the objective better than it should be.
A worked example makes it concrete (≈34:48–36:20). Take a state-action pair whose advantage is
$-1000$. The new policy gives the action probability 0.9 and the old policy gave it 0.1, so the
ratio is 9. Clipped, it becomes 1.2, and $1.2 \times (-1000)$ is a much better objective than
$9 \times (-1000)$. With the minimum, the clipped value is not used for this data point, which
discourages such accidents.

Why not clip only above? A student asks, and the answer is that the mirror case exists too
(≈36:20–37:06). If the advantage is $+1000$ and the new policy gives the action 0.1 where the old
gave 0.9, the ratio is 1/9. Clipping it up to 0.8 would again make the objective better. Clipping on
both sides and taking the lower bound keeps clipping from encouraging either scenario.

## PPO, trick 3: generalized advantage estimation

The last trick concerns how the advantage is estimated: **generalized advantage estimation (GAE)**
(slide 12, ≈37:51–40:10). Finn is not sure this trick is necessary and has not done
many comparisons. The value function is still fitted the normal way, with Monte Carlo or
bootstrapping. But instead of the one-step advantage, the reward plus the next value minus the
current value, it uses varying horizons, like [n-step returns](policy-evaluation.md): sum several
rewards into the future and add the value further ahead. The $n$-step advantage and the weighted
mixture are

$$\hat{A}^\pi_ n(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{t+n} \gamma^{t'-t} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) - \hat{V}^\pi_\phi(\mathbf{s}_ t) + \gamma^n \hat{V}^\pi_\phi(\mathbf{s}_ {t+n})$$

$$\hat{A}^\pi_ {\mathrm{GAE}}(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{n=1}^{\infty} w_n \hat{A}^\pi_ n(\mathbf{s}_ t, \mathbf{a}_ t)$$

with weights $w_n \propto \lambda^{n-1}$ for $n = 1, \ldots, N$, where $N \ll T$; the slide's note
on the weights is "cut earlier for less variance". (As printed, the reward sum runs to $t+n$,
whereas slide 5's n-step target stops at $t+n-1$.) There is an argument in terms of bias and
variance: weighting advantages of different horizons can give estimates with less bias than the
one-step estimate, "But it's also not 100% clear if it's necessary", in Finn's opinion (≈39:24).

Questions on it (≈40:10–41:42):

- **How are the weights chosen?** They are a hyperparameter. Mostly the earlier horizons are used,
  for less variance, with weights that decay quickly as the horizon grows, so it is still mostly
  $n = 1$.
- **Is $\gamma$ set to one?** No, the two are used together; written out, a term multiplies
  $\gamma$ and $\lambda$.
- **Is this a weighted average of n-step returns?** The value function is still fitted the normal
  way; GAE changes only how the advantage is estimated, which is like averaging n-step estimates
  over different $n$.
- **What is the benefit?** Finn finds it hard to motivate. With a one-step estimate, especially when
  time steps are close together, an inaccurate $\hat{V}$ can throw the estimate off, and a longer
  horizon may help; a mixture may hedge when you are not sure which $n$ is right. But Finn has no
  strong motivation for it and is not sure how much it changes performance.

Slide 12's "Why?" is answered by a meme credited to "one of the GAE authors": a comic-book figure
saying "I made it up" to a smaller figure asking "Source?".

## The PPO algorithm and example hyperparameters

The full PPO algorithm has four steps (slide 13, ≈41:42–42:28): sample a batch of data from
$\pi_\theta$; fit $\hat{V}^{\pi_\theta}_ \phi$ to it; evaluate the GAE advantage; and update the
policy with $M$ gradient steps on the clipped-and-minimum surrogate objective. Then repeat: sample
a new batch, fit a new value function, and so on.

The slide's example hyperparameters give a sense of scale, not a recommendation; hyperparameters
are tuned per domain (≈42:28–44:00). A batch has about 2,000 time steps, which for a long-horizon
problem of a thousand time steps is only a couple of trajectories. The policy update runs about 10
epochs over the batch, which with batch size 64 is $M$ of about 300 gradient steps; thanks to the
clipping, 300 gradient steps are possible without unstable improvement. The clipping range is
$\epsilon = 0.2$. About 500 iterations of the outer loop give about a million time steps of
experience in total.

![Slide 13 — Proximal Policy Optimization (PPO)](../raw/images/05-off-policy-actor-critic/slide-13.jpg)

*Slide 13 — the four-step PPO loop, its example hyperparameters, and Half-Cheetah learning curves where PPO (purple) climbs to about 1800 reward in a million time steps while vanilla policy gradient (tan) reaches about 600. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

On a simulated half cheetah, "like the right half of a cheetah or something", the slide's chart
plots reward against time steps collected (≈44:00–44:48). The tan curve is vanilla policy gradient,
"a slightly smarter version" of it but basically vanilla policy gradient, with no fitted advantage
function and no clipping. PPO, the purple curve, improves much faster with less experience, because
it fits the advantage function and because clipping lets it take many more gradient steps on each
batch. The chart's four other curves are unlabelled and not discussed.

Two questions on the algorithm (≈44:48–46:19):

- **The arrow from step 4 to step 1.** After step 4, the loop returns to step 1 to collect a new
  batch with the current policy: $\theta$ becomes the previous $\theta'$. The value function can be
  warm-started from the previous one or trained from scratch, depending on the domain.
- **Hyperparameters in the comparison.** Each method has its own. Best practice is to find good
  hyperparameters on one or two environments and apply them to the rest, rather than tuning per
  environment, which spends more samples than the curve reflects. Some domains need their own set,
  for example controlling something at one hertz versus 50 hertz, or optimizing a language model
  versus a half cheetah.

## Version 2: replay buffers

So far, one batch of policy data has bought one gradient step (fully on-policy) or several
(starting to be off-policy) (slide 14, ≈46:19–47:50). "Can we be even more off-policy?" That is,
can all past trial-and-error data be reused, rather than recollecting a batch each time? Two ideas
are needed. The first is to retain all past data in a **replay buffer**, which students met in
Homework 1: a dataset of past experience that keeps growing, from which very old data may
eventually be removed. It is called a replay buffer "because you're sort of replaying your past
experiences". The second is to adjust the equations to remove their on-policy assumptions.

With a buffer, the algorithm can run much more online (slide 15, ≈47:50–49:23). Instead of
collecting a batch and then updating, it gets one time step of data, updates the policy, gets
another time step, updates again, and so on, with each update drawing on all past experience. The
collected experience is added to the buffer, and the value and policy updates use a minibatch
sampled from all previous data. Walking through it: start with an empty buffer, collect experience
with the policy and add it, and sample a batch of transitions $(\mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i)$
of state, action, reward and next state $\mathbf{s}'$ from the buffer $\mathcal{R}$. There is a
small chance a sampled transition is the one just collected, but otherwise it is something randomly
sampled from earlier experience. Then update the value function on the minibatch, evaluate the
advantage, and take a gradient step on the policy.

![Slide 15 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-15.jpg)

*Slide 15 — the actor-critic loop with a replay buffer: steps 2 to 5 run on minibatches from all previous data, and the timeline alternates getting a transition with updating theta from transitions seen at earlier time steps. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## Why the replay-buffer algorithm is broken

"The algorithm is currently broken", in two ways (slide 16, ≈49:23–53:18). The first is the value
function. With bootstrapping, its target is the current reward plus the value estimate for the next
state, $y_i = r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$, fitted with the minibatch loss
$\mathcal{L}(\phi) = \frac{1}{N} \sum_i \Vert \hat{V}^\pi_\phi(\mathbf{s}_ i) - y_i \Vert^2$, where
$N$ is now the minibatch size.

The lecture's think-pair-share asks the slide's question: "When we fit $V^\pi$ on all data from
replay buffer, what policy $\pi$ is it learning a value function for?" (≈50:10). Finn warns that the
answer is "not as clean" as for other questions. The students' answer is that it is not the current
policy: the buffer holds at most perhaps one data point from it, and the earlier policies' actions
differ, so the next states and rewards differ too (≈50:57). Finn phrases it as "some sort of
mixture of all of your past policies", which is not trivial to write down. If the data were
collected in a particular way, it might look like an average of the $n$ previous policies, but it
varies by state: one policy may have gone one way and another policy another, so in some states it
is a mixture and in others mostly one policy. What is definite is that it is not a value function
for the current policy, and the advantage and gradient need the current policy's (≈51:44–52:30).

![Slide 16 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-16.jpg)

*Slide 16 — the broken algorithm: the circled value target is not the right one, the circled buffer action is not one the current policy would have taken, and the teal question asks which policy a value function fitted on the buffer belongs to. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

The second break is in the policy update: the action $\mathbf{a}_ i$ taken from the buffer, in the
gradient and in the advantage, is "not the action $\pi_\theta$ would have taken!" (≈52:30).

## Fitting a Q-function with data from past policies

How can a value function for the current policy be fitted from a buffer of data from past policies?
The idea is to fit $Q$ instead of $V$ (slide 17, ≈53:18). On the board, Finn starts from the
definition of the Q-function as a sum of future rewards, under the expectation of running
$\pi_\theta$ after starting at $\mathbf{s}_ t$ and taking $\mathbf{a}_ t$ (≈53:18–54:49). Writing out
the first term of the sum, the first state and action are just $\mathbf{s}_ t$ and $\mathbf{a}_ t$,
the inputs to the Q-function, so the first reward is known: $r(\mathbf{s}_ t, \mathbf{a}_ t)$. The
rest is a sum of rewards from $t+1$ on, in expectation over the next state drawn from the dynamics
and the later actions drawn from $\pi_\theta$, which looks a lot like a value function at
$\mathbf{s}_ {t+1}$ (≈54:49–55:37; a few words here are unclear in the recording). Taking it one step
further, with $\mathbf{s}_ {t+1}$ from the dynamics and $\mathbf{a}_ {t+1}$ from the policy, the
Q-value is the current reward plus the expected next Q-value, a recursive definition (≈56:25–57:10).
Slide 17 prints it with the discount factor:

$$Q^{\pi_\theta}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a}), \bar{\mathbf{a}}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ Q^{\pi_\theta}(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

Here $p(\cdot \mid \mathbf{s}, \mathbf{a})$ is the dynamics and $\bar{\mathbf{a}}'$ an action the
current policy takes at the next state. What makes it useful is that it holds "For *any*
$(\mathbf{s}, \mathbf{a})$": it assumes nothing about which policy chose the action (≈57:10).

Look at what the buffer offers (slide 17, ≈57:10–57:57). Each data point is a state, action, reward
and next state, $(\mathbf{s}, \mathbf{a}, r, \mathbf{s}')$, plus the future reward of the trajectory
it came from. The action is from a past policy, and the future reward "depends on past policy's
actions" in a complicated way, so the future reward is not used. A Q-function, though, takes the
action as input, so it is fine if that action is a bit different from what the current policy would
have done.

A student asks whether the Q-function's value still reflects the old policy, since the rest of that
trajectory followed it (≈58:42–59:29). The recursion avoids that: its second term uses the data's
next state $\mathbf{s}'$ but nothing after it, and in place of the next action in the data it uses an
action sampled from the current policy, with the bootstrapped estimate of Q at the next step.

The approach (slide 18, ≈59:29–1:01:46):

1. Sample $(\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i)$ from the buffer. The action is from an old
   policy, but "the dynamics are constant".
2. Sample $\bar{\mathbf{a}}'_ i$ from the *current* policy at $\mathbf{s}'_ i$.

Then the target is

$$Q^{\pi_\theta}(\mathbf{s}_ i, \mathbf{a}_ i) \approx r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma Q^{\pi_\theta}(\mathbf{s}'_ i, \bar{\mathbf{a}}'_ i) = y_i$$

Because the next action comes from the current policy rather than the data, the estimate is of the
current policy's Q-function. Fitting it is supervised learning: a network $\hat{Q}^\pi_\phi$ takes
the state and action as input and is trained with the same squared error as the value function,
$\mathcal{L}(\phi) = \frac{1}{N} \sum_i \Vert \hat{Q}^\pi_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - y_i \Vert^2$
(≈1:01:46).

![Slide 18 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-18.jpg)

*Slide 18 — fitting Q from the buffer: the target y is the reward plus the Q-value of the next state with an action sampled from the current policy, regressed by a network that takes s and a as input. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

The slide adds a **note**: "To be accurate, targets $y_i$ need sufficient action coverage". The
questions draw this out (≈1:00:15–1:04:04):

- **Do we need data from every state-action pair?** No, but for the target to be accurate, the data
  must cover actions similar to what the policy would take. Otherwise the Q-function will be
  inaccurate and data from the current policy is needed. So it is not a good Q-function for free from
  past policies, but it does give "somewhat of a means to use old data". The policy's action cannot be
  wildly out of distribution.
- **Must the action have been taken from that state?** Having no data from a state is a problem, and
  so is having data from it only for very different actions. Neural networks interpolate reasonably
  well, which helps, but coverage is still needed.
- **The expectation.** The recursion has an expectation over the next state and action; the target
  estimates it with a single sample, the next state from the data and one sampled action. Some
  methods use multiple samples for a better estimate. Within a batch, each data point gets its own
  sample, and one could also sample several actions for the same next state.
- **Would a KL constraint help coverage?** It is a good idea, and KL constraints come up frequently in
  RL, because methods often assume they are not deviating too far from the previous policy. One of
  the algorithms the course will see does not use one, though it could make it better in some
  circumstances.

In the algorithm, this is the revised step 3: update $\hat{Q}^\pi_\phi$ with targets
$y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$, where the state comes from the
buffer but the next action $\mathbf{a}'_ i \sim \pi_\theta(\mathbf{a}'_ i \mid \mathbf{s}'_ i)$ is
"**not** from replay buffer $\mathcal{R}$!" (slide 19, ≈1:04:50).

![Slide 19 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-19.jpg)

*Slide 19 — the fix for the value function: the circled V target is replaced by a Q-function target whose next action is sampled from the current policy, not taken from the replay buffer. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## Fixing the policy update

Two changes fix the circled policy-gradient term (slide 20, ≈1:04:50–1:06:23). First, it is
convenient to use $\hat{Q}^\pi$ instead of $\hat{A}^\pi$, which means no baseline. That is higher
variance, "but that's going to be okay because we're going to be using a lot more data", all the
data in the buffer rather than just the current policy's. Second, the current policy's actions are
likely better than past policies' actions, so the update samples an action from the current policy
instead of using the buffer's:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \thinspace \hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i), \qquad \mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$$

It is the same trick as for the Q-function's target, applied at the current state rather than the
next one. (Slide 20 still prints the old advantage in step 4, which slide 21 drops.)

![Slide 20 — Off-Policy Actor-Critic Methods](../raw/images/05-off-policy-actor-critic/slide-20.jpg)

*Slide 20 — the fix for the policy update: weight by Q instead of the advantage, and use an action freshly sampled from the current policy, not one from the replay buffer. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## The remaining problem, and soft actor-critic

The full algorithm now has five steps (slide 21): take an action with $\pi_\theta$ and store the
transition in $\mathcal{R}$; sample a batch from $\mathcal{R}$; update $\hat{Q}^\pi_\phi$ with the
targets above; take the gradient with freshly sampled actions; and update $\theta$. One problem
remains: the states in the buffer did not come from the current policy's state distribution
$p_\theta(\mathbf{s})$. There is nothing to be done about it, and it is not necessarily bad: the
policy is trained on a broader state distribution than the one it would visit (≈1:06:23). The slide
puts it as "we want optimal policy on $p_\theta(\mathbf{s})$" "but we get optimal policy on a
*broader* distribution".

This is essentially the **soft actor-critic** algorithm, or SAC (slide 22, ≈1:06:23–1:07:08). There
are other ways to estimate the gradient that the lecture has no time for; the slide names the
**reparameterization trick** for a Gaussian policy. There are also fancier ways to fit the
Q-function, which the next lecture covers ([Lecture 6](06-q-learning.md)). The slide cites Haarnoja, Zhou, Abbeel and Levine, *Soft
Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*
(2018).

## More off-policy versus less off-policy

Slide 23 compares SAC, the more off-policy method with a replay buffer (orange), and PPO, the less
off-policy one with no replay buffer (brown), on two humanoid tasks (≈1:07:08–1:07:55). SAC learns
far more data-efficiently. On the second chart, SAC passes 4,000 average return by about 5 million
steps and ends near 5,500 at 10 million, while PPO ends near 2,300. The downside is that SAC "can be
a lot harder to tune hyperparameters, and generally it can be a bit less stable" with its
Q-functions than PPO. The slide's verdict reads "+ Off-policy with replay buffer (e.g. soft actor-critic) can be
far more data efficient" and "- They can also generally be a lot harder to tune hyperparameters,
less stable (than e.g. PPO)". The [slide file](../raw/slides/05-off-policy-actor-critic.md) gives
each curve, including DDPG, SQL and TD3, which the lecture does not discuss.

![Slide 23 — More off-policy vs. less off-policy actor critic?](../raw/images/05-off-policy-actor-critic/slide-23.jpg)

*Slide 23 — learning curves on two humanoid tasks: SAC (orange, with a replay buffer) learns far faster than PPO (brown, no replay buffer), most clearly on the right-hand task, at the cost of being harder to tune. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

## Examples: real robots with SAC, simulation with PPO

Fully off-policy learning can be efficient enough for reinforcement learning on real robots (slides
24–25, ≈1:07:55–1:08:40). In the first video, a four-legged robot learns to walk forward from
scratch in about two hours "just with the algorithm that we talked about" (slide 24, Haarnoja et
al., *Soft Actor-Critic Algorithms and Applications*, 2018).

![Slide 24 — Fully Off-Policy = Can be efficient enough for real-world RL](../raw/images/05-off-policy-actor-critic/slide-24.jpg)

*Slide 24 — the first frame of the video in which a small four-legged robot learns to walk from scratch with SAC in under two hours. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

The second example learns precise manipulation, such as motherboard and timing-belt assembly, with
an algorithm based on SAC, seeded with demonstrations (slide 25; Luo, Xu, Wu, Levine, *HIL-SERL*,
2024). The RL method reaches 100% success on every task in the slide's table, where imitation
learning (behaviour cloning) gets much lower success rates, and the RL policy is also faster. For
example, RAM insertion goes from 29% to 100% success and from 8.3 to 4.8 seconds per cycle, after
1.5 hours of training; the timing belt from 2% to 100%, after 6 hours. The slide file has the whole
table.

![Slide 25 — Fully Off-Policy = Can be efficient enough for real-world RL](../raw/images/05-off-policy-actor-critic/slide-25.jpg)

*Slide 25 — precise assembly learned with an SAC-based method seeded with demonstrations: the table's eight tasks all reach 100% success (10 of 10 for the whole IKEA assembly) with shorter cycle times than behaviour cloning. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

PPO is the common choice when you want stable learning and can accept less data-efficient learning
(slides 26–27, ≈1:08:40). In simulation, data is not a problem, so a policy can be learned there
and, with a carefully tuned simulator, may transfer to the real world, as with a quadruped that
learns to walk in simulation (slide 26; Tan et al., *Sim-to-Real: Learning Agile Locomotion For
Quadruped Robots*, 2018).

![Slide 26 — PPO = Common algorithm of choice for stable, less efficient learning](../raw/images/05-off-policy-actor-critic/slide-26.jpg)

*Slide 26 — a simulated quadruped that learns to walk with PPO; the policy transfers to the real robot with careful simulation choices. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

For manipulation, sim-to-real transfer is "a lot harder" in general. The example is a robot hand
solving a Rubik's cube with a policy trained in simulation with PPO, which completes the cube after
about 5 minutes of the roll-out (slide 27; OpenAI, *Solving Rubik's Cube with a Robot Hand*, 2019).
The inset grid shows the same hand and cube rendered with varied colours and textures.

![Slide 27 — PPO = Common algorithm of choice for stable, less efficient learning](../raw/images/05-off-policy-actor-critic/slide-27.jpg)

*Slide 27 — a robot hand trained in simulation with PPO to solve a Rubik's cube; the inset shows the hand and cube in many differently coloured simulated renderings. [Deck](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf)*

PPO is also a common choice for language models, which the course starts to discuss next week
(slide 28, ≈1:08:40); the slide says this will come with reward modeling.

## Not in the recording: reminders and next time

The recording begins on the recap and ends on slide 28, so three slides are known only from the
[slide file](../raw/slides/05-off-policy-actor-critic.md). Slide 2's course reminders are that the
project survey is due that night, Homework 1 is due Friday and Homework 2 comes out Friday. Slide
29 repeats the plan. Slide 30 announces the next lecture: Q-learning, the "last big online RL
method", and the practical implementation of Q-learning, with the same reminders. See
[Lecture 6 — Q-Learning](06-q-learning.md).

## Related pages

- [Actor-critic methods](actor-critic.md): the method as a whole, from the advantage-weighted
  gradient to the off-policy versions this lecture builds.
- [Proximal policy optimization](ppo.md): the clipped surrogate objective, the minimum, GAE and the
  algorithm.
- [Replay buffers and soft actor-critic](replay-buffers.md): reusing all past data, why it breaks a
  value function, and the Q-function fix.
- [Importance sampling](importance-sampling.md): the importance weights behind version 1.
- [Value functions](value-functions.md): $V^\pi$, $Q^\pi$, $A^\pi$, and this lecture's recursive form
  of $Q^\pi$.
- [Policy evaluation](policy-evaluation.md): Monte Carlo, bootstrapping and n-step returns, and
  fitting a Q-function from a buffer.
- [Offline and online algorithms](offline-and-online.md): on-policy versus off-policy, and how far
  off-policy each method goes.
- [RL algorithm families](rl-algorithm-families.md): where PPO and SAC sit among the course's
  methods.
- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md): the on-policy algorithm this lecture
  starts from.
- [Lecture 6 — Q-Learning](06-q-learning.md): the next lecture, which drops the actor and arrives
  at Q-learning.
