# RL algorithm families

Every algorithm in CS224R is a way of maximizing the same [objective](rl-objective.md), the
expected sum of rewards
$\mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$.
Lecture 1 sorts them into five families (slide 41, ≈49:20–50:53).

| Family | Idea (Lecture 1, slide 41) | Lecture in the Spring 2025 schedule |
| --- | --- | --- |
| Imitation learning | mimic a policy that achieves high reward | 2; see [imitation learning](imitation-learning.md) |
| Policy gradients | directly differentiate the objective | 3; see [policy gradients](policy-gradients.md) |
| Actor-critic | estimate the value of the current policy and use it to make the policy better | 4, 5; see [actor-critic methods](actor-critic.md) |
| Value-based | estimate the value of the optimal policy | 6; see [Q-learning](q-learning.md) |
| Model-based | learn to model the dynamics, and use it for planning or policy improvement | 11, 12; see [model-based RL](model-based-rl.md) |

The last column comes from the [course schedule](../sources.md#lectures), not from Lecture 1;
the schedule also has lectures on offline RL, reward learning, RL for language models,
multi-task and meta-RL, exploration, hierarchy and RL for robots. In the lecture's words, policy
gradients learn directly from practice and experience; value-based methods use the estimated
value of the optimal policy to back out a good policy; and model-based methods learn to predict
the future and use that for planning or to improve the policy (≈50:07). The
[value function and Q-function](value-functions.md) are the "value" these families estimate.

**This knowledge base so far covers Lectures 1–12.** Imitation learning, policy gradients,
actor-critic methods and value-based methods have their own pages,
[imitation learning](imitation-learning.md), [policy gradients](policy-gradients.md),
[actor-critic methods](actor-critic.md) and [Q-learning](q-learning.md), and Lecture 7's offline
RL has [offline RL](offline-rl.md), [implicit policy constraint methods](implicit-policy-constraints.md)
and [conservative Q-learning](conservative-q-learning.md); Lecture 8's question of where the reward
comes from has [reward learning](reward-learning.md) and
[learning rewards from preferences](preference-learning.md); Lecture 9's RL for language models has
[instruction finetuning](instruction-finetuning.md), [RL from human feedback](rlhf.md) and
[direct preference optimization](direct-preference-optimization.md); Lecture 10's RL for LLM reasoning has
[RL for LLM reasoning](llm-reasoning.md) and [process rewards and per-step credit](process-rewards.md);
Lecture 11's model-based methods have [model-based RL](model-based-rl.md) and
[planning with a learned model](planning-with-models.md); and Lecture 12's multi-task RL has
[multi-task RL](multi-task-rl.md) and [hindsight relabeling](hindsight-relabeling.md). Lecture 2 adds
two points about where imitation learning sits. It is fully **offline** and needs no reward
function, but it may need a great deal of data and gives the policy no way to improve on its own
from practicing; its summary slide notes that "Many successful methods **combine** imitation
learning and reinforcement learning!" (Lecture 2, slide 25). And the offline/online distinction it
introduces is a general way to classify algorithms; see
[offline and online algorithms](offline-and-online.md).

Lecture 3 makes the link between the first two families explicit. The policy gradient's first
factor is the gradient of the imitation-learning objective, so a policy gradient is the imitation
gradient weighted by reward, applied to the policy's own data (slide 11). Unlike imitation
learning, it can improve beyond a demonstrator, from practice. It is the course's first **online**
algorithm, and an **on-policy** one: it needs fresh data from the current policy for every
gradient step, unless importance sampling is used to reuse a batch (slide 20). The lecture's
verdict is that it works best with large batches and dense rewards (slide 26).

Lecture 4 builds the third family on the second. An actor-critic method is a policy gradient whose
weight on each action is not the sampled reward to go but an estimated advantage, computed from a
learned value function, the critic (Lecture 4, slides 10 and 22). In the lecture's review, policy
gradients "do more of the good stuff, less of the bad stuff, based on what you observe", while
actor-critic methods "are learning to estimate what's good and bad, and then doing more of the good
stuff according to your estimate" (≈1:02:21). Lecture 5 makes it off-policy and arrives at the two practical
algorithms the course names. [PPO](ppo.md) takes many gradient steps per batch with clipped
importance weights, and soft actor-critic learns a Q-function from a
[replay buffer](replay-buffers.md) of all past data. The off-policy version "can be far more data
efficient" but is harder to tune and less stable than PPO (Lecture 4, slide 35; Lecture 5, slide
23). That is the stability-against-efficiency trade-off listed below, and Lecture 5's examples show
both sides: SAC is efficient enough to learn on real robots, while PPO is the common choice in
simulation, where data is cheap, and for language models (Lecture 5, slides 24–28).

Lecture 6 builds the fourth family out of the third by dropping the actor. With a Q-function
critic, the greedy policy that takes the highest-valued action is at least as good as the current
policy, so the policy network and its gradient can go; putting the max into the critic's target
gives [Q-learning](q-learning.md), which fits the Q-function of the optimal policy through the
Bellman optimality equation (Lecture 6, slides 8–12). With a target network it is DQN. Lecture 6
ends the course's online methods with Finn's advice on choosing among the three practical ones
(slide 27, "Chelsea's advice", ≈59:41–1:00:27):

- **PPO and variants** when you care about stability and ease of use and not about data
  efficiency;
- **DQN and variants** when you have discrete actions or low-dimensional continuous actions;
- **SAC and variants** when you care most about data efficiency and are fine with tuning
  hyperparameters and less stability.

Lecture 7 adds advice for learning from a fixed dataset (slide 32, ≈1:06:42–1:07:51). To train offline
only, start with filtered behavior cloning; implicit Q-learning can stitch and stays within the data's
support, and will probably work better, especially with a broader dataset; conservative Q-learning has
"just one hyperparameter" (slide only). For offline pre-training followed by online fine-tuning,
implicit Q-learning "seems most performant". See [offline RL](offline-rl.md#which-to-use).

Lecture 11 turns to the fifth family with a recap that puts the first four, and offline RL, on one map
(slide 4, ≈0:05–4:36): policy gradient methods learn only an actor (REINFORCE), actor-critic methods
both (PPO, SAC), and Q-learning methods only a critic (DQN). Finn notes that PPO is often called a
policy gradient method, and this class calls it actor-critic to avoid confusion (≈2:21). Model-based RL
"can actually be applied to all of the different classes" of method (≈4:36). It learns a dynamics model
and plans against it, by backpropagation or by sampling with the cross-entropy method, replanning after
every action in model predictive control, or uses the model to generate data for a model-free method
(Lecture 11, slides 12–28). Slide 29 answers when to use it: "Whether to use a model depends on how hard
it is to learn!", which is the last trade-off below. See [model-based RL](model-based-rl.md) and
[offline and online algorithms](offline-and-online.md#the-whole-map-so-far).

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
- **Is it easy to learn the dynamics model?** Lecture 11's answer to when a model is worth having
  turns on exactly this (slide 29); see [model-based RL](model-based-rl.md#when-to-use-a-model).

The lecture says the course will return to which algorithm suits which setting in its later
parts.

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#types-of-algorithms-and-why-there-are-so-many):
  the five families and the trade-offs between them.
- [Lecture 2 — Imitation Learning](02-imitation-learning.md): the first family in depth.
- [Lecture 3 — Policy Gradients](03-policy-gradients.md): the second family in depth.
- [Lecture 4 — Actor-Critic Methods](04-actor-critic.md): the third family in depth.
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md): its off-policy versions, PPO
  and SAC, and the trade-off between them.
- [Lecture 6 — Q-Learning](06-q-learning.md): the fourth family in depth, and when to use PPO, DQN
  or SAC.
- [Lecture 7 — Offline RL](07-offline-rl.md#which-offline-rl-algorithm-to-use): which offline RL
  algorithm to use.
- [Lecture 8 — Reward Learning](08-reward-learning.md): conservative Q-learning, and how to get the
  reward every family but imitation learning needs.
- [Lecture 9 — RL for LLMs](09-rl-for-llms.md): policy gradients applied to language models, and
  DPO, which reaches the same objective with a classification loss.
- [Lecture 10 — RL for LLM Reasoning](10-rl-for-reasoning.md): imitation, offline RL and online
  RL for math reasoning, and how thinking models are trained.
- [Lecture 11 — Model-Based RL](11-model-based-rl.md): the map of the families so far, and the fifth
  family in depth.
- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md): model-based policy optimization, which feeds a
  model's synthetic data to any model-free method, and when to use a model.
