# Model-based RL

The fifth of the course's algorithm families: learn a model of the dynamics, a "simulator" that
predicts what happens next, and use it to choose actions or to train a policy. Lecture 1 names the
family, saying that model-based methods "learn to model the dynamics, and use it for planning or
policy improvement" (Lecture 1, slide 41). [Lecture 11](11-model-based-rl.md) teaches it. This page
collects the idea, how a model is learned, the two ways to use one, and when a model is worth having.
For the planning algorithms themselves, see [planning with a learned model](planning-with-models.md).

## The idea: learn a simulator

A dynamics model is a "predictive model of $\mathbf{s}_ {t+1}$ given $\mathbf{s}_ t$,
$\mathbf{a}_ t$", the next state given the current state $\mathbf{s}_ t$ and action $\mathbf{a}_ t$
(Lecture 11, slide 6). It lets the agent ask what would happen if it took some action, before taking
it (≈5:22). Model-based RL can be applied to every class of method in the course, offline or online,
on-policy or off-policy (≈4:36). What the model looks like depends on the domain (slide 6,
≈6:11–8:29). For a physical system whose parts can be observed directly, it models the physics; when
they cannot be observed, it looks more like video prediction conditioned on the agent's actions, the
kind of video that models like Veo 2 and Sora generate. In finance it predicts the stock market, and
whether the agent's actions move it. In games it is the rules, which are often known already, as in
chess, so that no model needs to be learned; other players can be treated as part of the dynamics but
are often modelled separately.

The simplest algorithm has three steps (slide 7, ≈8:29–9:15). Collect data
$\mathcal{D} = \lbrace \mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i \rbrace$ of states, actions and
next states with some base policy $\pi_0(\mathbf{a} \mid \mathbf{s})$. Learn a simulator
$f(\mathbf{s}, \mathbf{a}) = \mathbf{s}'$, for example by minimizing the squared error between
predicted and observed next states:

$$\min_ f \sum_ i \Vert f(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$$

Then run an RL method, or a planning method, inside the simulator.

## What goes wrong

Each step has its own problem (Lecture 11, slide 8, ≈10:02–13:05):

- **Data coverage matters a lot.** The model is accurate only where the data are. If the base policy
  $\pi_0$ visits different states from the policy finally learned, the model is wrong exactly where the
  learned policy goes, and an optimizer will **exploit** those errors, as policies exploit errors in
  reward models and in Q-functions; the errors can compound, and what was learned in the simulator
  fails to transfer (≈10:02–10:47). In slide 8's maze the data cover one corner, so nothing can be
  learned about reaching the rest.
- **Modeling is domain dependent.** "Some domains much easier to simulate than others!" A Sora video of
  a robot stacking cubes changes the blocks' colours and makes blocks out of thin air (≈11:33). The
  inputs, training and architecture of a model are domain-dependent choices (≈12:19).
- **A good simulator does not make the RL easy.** The RL still has to be run, and it has to "account
  for model inaccuracies" (slide 8).

![Lecture 11, slide 8 — Notional learned-simulator algorithm, annotated](../raw/images/11-model-based-rl/slide-8.jpg)

*Lecture 11, slide 8 — the three steps of the notional algorithm, each with its problem: data that cover one corner of a maze, a Sora video that gets the physics wrong, and model errors the RL step must account for. [Deck](https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf)*

## Learning a dynamics model

Lecture 11 treats this briefly (slide 10, ≈13:05–19:24). Sometimes the model is **known already**, as
in some games. Sometimes it is **approximately known**: a physical model may lack only a table's
friction coefficient, or a car model a few effects, and then the known structure can be kept and the
unknown parameters fitted from data, which Finn calls one of the cooler ways to put domain-specific
knowledge into RL (≈13:51–14:36). In almost every practical case it is **not known**, and is learned
from scratch with a neural network or another function approximator (≈14:36).

With image observations, the model can predict the entire future video, conditioned on actions, which
is expensive because every pixel has to be modelled; or it can predict in a learned lower-dimensional
representation $\mathbf{z}_ t$ produced by an encoder, which is cheaper but needs a good representation
in the first place (slide 10, ≈16:12–17:02). The representation loses information on purpose: details
such as the texture of the road usually do not matter, and not modelling them saves compute (≈17:49).

The model is trained for its ability to predict future states: how well it captures the distribution
of $\mathbf{s}_ {t+1}$, or, for more deterministic dynamics, the error between the predicted and actual
$\mathbf{s}_ {t+1}$ (≈19:24). The deck's models are trained by the squared error above, with a network
$f_ \phi(\mathbf{s}, \mathbf{a})$ with parameters $\phi$ (slide 12). A **reward model** is often needed
as well, because seeing into the future means predicting the rewards the future states bring, not only
the states (slide 10, ≈18:36); see [reward learning](reward-learning.md).

The case study in Lecture 11 shows one more ingredient: an **ensemble**. PDDM trains three networks,
each with two hidden layers of size 500, differing only in random initialization and minibatch order,
so that one model's optimism is counterbalanced by the others and the planner cannot exploit the errors
of any one model (slide 33, ≈1:08:12–1:09:01). Its ablations say it needs a sufficiently large model
and at least three ensemble members (slide 35).

## Two ways to use a model

Lecture 11's plan splits the use of a learned model into **planning** and **data generation** (slide
5).

**Planning** optimizes a sequence of actions against the model at every step, so no policy is learned:
the optimization is the "policy" (slide 12, ≈28:46–29:32). It can run by backpropagating rewards through
the model into the actions, or by sampling action sequences, with random shooting or the cross-entropy
method; it adds the executed data to the dataset and refits the model; and model predictive control
replans after every action (slides 12–22). It is simple and lets the reward be changed at test time,
but it is compute intensive at test time and practical only for short horizons, or very shaped rewards
(slide 23, ≈1:00:33–1:02:51). Distilling the planner's actions into a policy removes the test-time cost,
and a terminal value function lets planning reach past its horizon (slide 25, ≈1:02:51–1:05:09). All of
this is on [planning with a learned model](planning-with-models.md).

**Data generation** uses the model to make extra training data for a model-free method. Lecture 11
leaves it for the next lecture (≈1:06:40), so it is in the slides only. The key idea of model-based policy optimization is to
"augment data with model-simulated roll-outs" (slides 26–28). Full trajectories from the initial
states are a poor choice, because the "model may not be accurate for long horizons"; short trajectories
from the initial states "may not get good coverage of later states"; so the roll-outs are **partial
trajectories from all states in the data**, short branches off every real state (slide 27).

![Lecture 11, slide 27 — Model-based policy optimization, augmented roll-outs](../raw/images/11-model-based-rl/slide-27.png)

*Lecture 11, slide 27 — short model roll-outs, the pink dots, branch off states all along a real trajectory. [Deck](https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf)*

The full algorithm (slide 28, in the slide's notation, where $\pi_\phi$ is the policy and $p_\theta$ the
model) collects data with the current policy into $D_ {env}$; updates the model
$p_\theta(s' \mid s, a)$ on $D_ {env}$; collects synthetic roll-outs with $\pi_\phi$ in the model from
states in $D_ {env}$ into $D_ {model}$; updates the policy (and a critic $Q$) on $D_ {model}$; and
repeats. Any model-free RL method can do step 4, which can also use $D_ {env}$. Lecture 12's deck opens
with its own version of these slides.

## When to use a model

Slide 29 of Lecture 11 (not discussed aloud) weighs "Big upsides and big downsides". A model is
"Immensely useful, far more data efficient if model is easy to learn"; it "can be trained on data without
reward labels (fully self-supervised)"; and it "is somewhat task-agnostic (can sometimes be transferred
across rewards)". But "Models don't optimize for task performance", are "Sometimes harder to learn than
a policy", and are "Another thing to train, more hyperparameters, more compute intensive". The verdict:
"Whether to use a model depends on how hard it is to learn!" That is the last of the trade-offs Lecture
1 lists for choosing an algorithm, "Is it easy to learn the dynamics model?" (Lecture 1, slide 42); see
[RL algorithm families](rl-algorithm-families.md#why-so-many-algorithms).

The case study gives the upside in numbers. On a simulated handwriting task, Finn puts the model-based
method at maybe on the order of 100,000 time steps of data against around half a million for the
model-free methods (Lecture 11, slide 34, ≈1:11:21), and on a real 24-joint robot hand it learned to
rotate Baoding balls by 90° with around 100% success from about four hours of data (slide 36,
≈1:12:06). Data efficiency "is key for fragile hardware".

## Other kinds of models

Everything above models $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$. Slide 30 of Lecture 11
lists alternatives with "various purposes": an inverse model
$p(\mathbf{a}_ t \mid \mathbf{s}_ t, \mathbf{s}_ {t+1})$; a multi-step inverse model
$p(\mathbf{a}_ t \mid \mathbf{s}_ t, \mathbf{s}_ {t+n})$ or $p(\mathbf{a}_ {t:t+n} \mid \mathbf{s}_ t, \mathbf{s}_ {t+n})$;
future prediction without actions, $p(\mathbf{s}_ {t+1:t+n} \mid \mathbf{s}_ t)$; video interpolation,
$p(\mathbf{s}_ {t+1:t+n} \mid \mathbf{s}_ t, \mathbf{s}_ {t+n+1})$; and the transition distribution
$p(\mathbf{s}_ t, \mathbf{a}_ t, \mathbf{s}_ {t+1})$.

## Relation to the other families

Lecture 11's recap (slide 4) places the model-free methods on one map: offline RL (AWR, AWAC, IQL,
CQL), off-policy RL with a replay buffer (DQN, SAC) or multiple gradient steps (PPO, importance
sampling), and on-policy RL (REINFORCE); see [offline and online algorithms](offline-and-online.md).
Planning with a model is itself off-policy: the model can be fitted to data from any policy, and a broad
data-collecting policy makes it accurate in more places (≈59:47–1:00:33). Planning with a terminal
value function combines planning with value-based RL (≈1:05:09); see
[value functions](value-functions.md). Distilling a planner into a policy is training a policy to
match given actions, as in [imitation learning](imitation-learning.md). Model-based policy optimization
feeds a model-free method such as SAC, one of the case study's baselines; see
[replay buffers and soft actor-critic](replay-buffers.md).

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#types-of-algorithms-and-why-there-are-so-many): the
  family named, and whether the dynamics model is easy to learn as a reason to choose an algorithm.
- [Lecture 11 — Model-Based RL](11-model-based-rl.md): the whole family, its planning methods and the
  dexterous-manipulation case study.
- [Planning with a learned model](planning-with-models.md): gradient-based and sampling-based planning
  and model predictive control.
- [RL algorithm families](rl-algorithm-families.md): model-based methods among the course's algorithms.
