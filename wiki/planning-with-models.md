# Planning with a learned model

Planning chooses actions by optimizing against a model of the world, rather than by running a learned
policy. With a learned dynamics model $f_ \phi(\mathbf{s}, \mathbf{a})$, which predicts the next state
from a state $\mathbf{s}$ and action $\mathbf{a}$, and a reward $r(\mathbf{s}, \mathbf{a})$ that is
known or learned, the agent looks ahead over a horizon of $H$ steps and picks the action sequence with
the highest predicted sum of rewards. [Lecture 11](11-model-based-rl.md) builds it up from
backpropagation through the model to model predictive control. For the family it belongs to, see
[model-based RL](model-based-rl.md).

## The objective

The planner solves (Lecture 11, slide 12, ≈21:45–22:31)

$$\max_ {\mathbf{a}_ {t:t+H}} \sum_ t r(\mathbf{s}_ t, \mathbf{a}_ t)$$

over the actions from time $t$ to $t + H$, with each future state predicted by the model. "Planning"
means exactly this: looking into the future for a sequence of actions that maximizes reward. No policy
network is learned. The only things stored are the parameters of the dynamics model and of the reward
model, and at every state the agent runs the optimization again, thinking through what would happen if
it did this, this or this (≈28:46–29:32, ≈40:21–41:08). That is why Lecture 11 calls it a "policy" in
quotes: it maps a state to actions, but through an iterative optimization rather than a compact closed
form (≈39:35–40:21). It needs a lot of compute, in real time.

Every variant below shares the first two steps: run some policy, for example a random one, to collect
$\mathcal{D} = \lbrace (\mathbf{s}, \mathbf{a}, \mathbf{s}')_ i \rbrace$, and learn
$f_ \phi(\mathbf{s}, \mathbf{a})$ to minimize
$\sum_ i \Vert f_ \phi(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$ (slide 12). They differ in
how they optimize the actions and how they execute them.

## By backpropagation

If the reward model and the dynamics model are differentiable networks, the sum of rewards can be
backpropagated through them into the actions (Lecture 11, slide 12, ≈20:57–21:45), which methods that
did not know how actions affect states could not do. The actions are initialized, as network parameters
would be, and updated by gradient steps (≈24:55–26:28). By the chain rule the gradient is the
derivative of the rewards with respect to the states, from the reward model, times the derivative of the
states with respect to the actions, from the dynamics model (≈25:41). Since the rewards are maximized,
the step is gradient ascent,

$$\hat{\mathbf{a}}_ {t:t+H} \leftarrow \hat{\mathbf{a}}_ {t:t+H} + \alpha \thinspace \nabla_ {\mathbf{a}} \sum_ {t'=t}^{t+H} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$$

with step size $\alpha$; the lecture first writes it with a minus sign and corrects it (≈26:28, ≈32:40).
This assumes continuous actions and, so far, deterministic models; a stochastic model needs tricks that
depend on how it is modelled, such as differentiating through a Gaussian's samples (≈26:28–28:00).

## By sampling

Sampling-based, or gradient-free, optimization uses no gradients: it evaluates candidate action
sequences in the model and keeps the best (Lecture 11, slides 15–17, ≈45:01–48:05). Write
$\mathbf{A} := \mathbf{a}_ t, \ldots, \mathbf{a}_ {t+H}$ (slide 16).

- **Random shooting** ("Guess & check") samples many sequences $\mathbf{A}_ 1, \ldots, \mathbf{A}_ N$
  from some distribution, such as a uniform one, and executes the one with the highest predicted sum of
  rewards, $\sum_ {t'=t}^{t+H} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$. It shoots out possible futures and
  keeps the one shot with the highest reward (≈45:47–46:34), and it sometimes works better than one might
  expect (≈34:13).
- **The cross-entropy method (CEM)** improves the sampling distribution. Sample
  $\mathbf{A}_ 1, \ldots, \mathbf{A}_ N$ from $p(\mathbf{A})$; evaluate
  $J(\mathbf{A}_ i) = \sum_ {t'=t}^{t+H} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$; pick the $M \lt N$
  elites with the largest $J$; refit $p(\mathbf{A})$, for example a Gaussian, to the elites; and repeat,
  returning the best sample of the last round (slides 13 and 16, ≈38:03–38:50, ≈46:34–47:19). It is "Not
  to be confused with the cross-entropy loss!" (slide 13).

![Lecture 11, slide 13 — Gradient-based vs. sampling-based optimization](../raw/images/11-model-based-rl/slide-13.png)

*Lecture 11, slide 13 — first-order optimization follows the gradient, zeroth-order optimization only evaluates samples; below, the cross-entropy method's sample, rank and refit loop. [Deck](https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf)*

## Gradients or samples?

The trade-off is the one between first-order and zeroth-order optimization (Lecture 11, slide 14,
≈43:28–45:01). Gradient descent is scalable to high dimensions and works well in overparameterized
regimes, but needs a nice, smooth optimization landscape; for neural network training its pros heavily
outweigh its cons. Sampling is parallelizable, where gradient descent is inevitably sequential, which
can make it faster; it needs no gradients, so it works with discrete variables and rewards whose
gradient is unknown (≈36:31–37:18); but it scales poorly to high dimensions, because samples must cover
the space. For planning, the dimension is that of the action space times the horizon, "both $H$ and
$\vert \mathbf{a} \vert$", so CEM is fast and simple when that is small (slide 16, ≈47:19–48:05).

## Refitting the model on the planner's own data

A model fitted to the base policy's data is wrong wherever that policy never went, and the planner
exploits it. In Lecture 11's cliff, the reward is height, the data show that going right goes higher,
and the planned path runs over the edge (slide 18, ≈48:05–49:38): the **data distribution mismatch**
$p_ {\pi_ 0}(\mathbf{s}) \neq p_ {\pi_ f}(\mathbf{s})$ between the states of the base policy $\pi_0$ and
those of the final policy $\pi_ f$. The fix is to execute the planned actions, append the visited
$(\mathbf{s}, \mathbf{a}, \mathbf{s}')$ tuples to $\mathcal{D}$, refit the model and plan again (slide
19, ≈49:38–50:24). After falling once, the model knows the cliff, and the plan goes to the top and stops
(slide 20). The loop is off-policy: its data can come from any policy, and a broad one makes the model
accurate in more places (≈59:47–1:00:33).

## Model predictive control

Executing a whole planned sequence is **open loop**: the agent does not see the intermediate states.
Executing only the first action, observing the next state and replanning is **closed loop** (Lecture
11, slide 21, ≈51:13–52:52). Open loop is risky when the environment changes in ways the model did not
predict, and $H$ should never be so large that the agent runs too long without feedback (≈53:37–54:25).
Planning made closed loop is **model predictive control (MPC)** (slide 22, ≈54:25–55:11):

1. Run a base policy $\pi_0(\mathbf{a}_ t \mid \mathbf{s}_ t)$ to collect $\mathcal{D}$.
2. Learn $f_ \phi(\mathbf{s}, \mathbf{a})$.
3. Use the model to optimize an action sequence.
4. Execute the first planned action, observe the resulting state $\mathbf{s}'$, and go back to step 3.
5. Append $(\mathbf{s}, \mathbf{a}, \mathbf{s}')$ to $\mathcal{D}$, and periodically go back to step 2.

![Lecture 11, slide 22 — Plan and replan using the model](../raw/images/11-model-based-rl/slide-22.jpg)

*Lecture 11, slide 22 — model predictive control: replan after every executed action, refit the model now and then; replanning is what corrects the drifting car's model errors. [Deck](https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf)*

MPC is very widely used in robotics, for example in legged robots planning over the next couple of
seconds (≈55:11–55:56). Replanning lets the agent recover from a slip or a changed world, and it
corrects for **model errors**, which makes it much more robust when the model is wrong in parts of the
state space (≈55:56–56:42). Its cost is compute (slide 22). Three practices from Lecture 11's questions:

- **Warm-starting.** Initialize the plan for $\mathbf{a}_ {t+1}$ to $\mathbf{a}_ {t+1+H}$ with the
  previous plan minus its first action, plus a random last action. That gives a sensible starting point,
  a shorter optimization, and commitment to the previous plan, which damps oscillation between several
  good paths. A sampling-based planner can sample near the previous plan, with some noise (≈56:42–58:15).
- **Replanning less often.** A middle ground replans every few steps rather than every step (≈59:01),
  and once the model is good, the final policy can replan less than during learning (≈59:01–59:47).
- **Latency.** If the computation takes a couple of time steps, the agent can start from the plan's
  later actions, at the risk that the state differs from what the plan assumed; handling this well is
  somewhat of an open research problem (≈41:55–42:42).

During learning, the planner's tendency to exploit model errors also drives exploration toward exactly
the regions where the model is wrong (≈59:47).

## Limits, and two ways past them

Planning is simple and makes it easy to plug in different goals or rewards, even at test time: one good
model can make a robot run forwards, walk backwards or crouch (Lecture 11, slide 23, ≈1:00:33–1:01:20).
But it is compute intensive at test time and only practical for short horizons or very shaped rewards,
because long plans are too expensive and the model is not accurate far ahead (slide 23, ≈1:02:06–1:02:51).
A dense reward such as forward velocity lets a short plan approximate an endless task; a sparse
long-horizon problem does not suit it (≈1:02:06).

Two remedies are on slide 25 (≈1:02:51–1:05:09). **Distilling the planner** trains a policy to match the
planner's actions, which makes acting one forward pass but leaves the horizon limit in place. **A
terminal value function** adds the estimated value of the state reached after the $H$ planned actions to
their sum of rewards, so the plan accounts for what lies past its horizon; it combines planning with
value-based RL, at the cost of learning both a model and a value function (≈1:04:23–1:05:55). See
[value functions](value-functions.md).

## In practice: the case study

Lecture 11's case study, PDDM on a 24-joint robot hand, plans with a modified CEM, using a softer,
reward-weighted mean and temporal smoothing of the actions, against an ensemble of three models (slide
33, ≈1:09:01). Its ablations find that this modified CEM is crucial (it reaches about 0.95 normalized
reward, against about 0.55 for plain CEM and a very noisy curve for random shooting) and that the
planning horizon is a trade-off, since planning too far ahead is too high-dimensional (slide 35,
≈1:11:21–1:12:06). The baseline that only does random shooting, with no ensemble, does very poorly,
because it never hones in on good actions (slide 34, ≈1:10:35–1:11:21).

## Where this appears

- [Lecture 11 — Model-Based RL](11-model-based-rl.md#approach-1a-planning-by-backpropagation): the
  planning algorithms in order, with the questions on each, and the case study.
- [Model-based RL](model-based-rl.md): learning the model planning uses, and the other way to use it.
- [Value functions](value-functions.md): the terminal value function at the end of the horizon.
- [Imitation learning](imitation-learning.md): training a policy to match a planner's actions.
