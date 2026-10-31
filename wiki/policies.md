# Policies

A **policy** represents behavior: it selects actions based on states or observations
(Lecture 1, slide 44). CS224R writes it $\pi$, or $\pi_\theta$ when it has learnable parameters
$\theta$, and treats it as a conditional distribution over actions,
$\pi_\theta(\mathbf{a} \mid \mathbf{s})$. Symbols are as defined on
[MDPs and RL notation](mdp-and-notation.md).

## A policy is usually a neural network

The policy can be any function from states, observations or a sequence of observations to
actions: a convolutional network, a transformer, a fully connected network, or even a linear
function. $\theta$ denotes the parameters being learned, which for a neural network are its
weights (Lecture 1, ≈33:12–33:58). Slide 36 draws the example: a camera image of the towel and
hook (the state $\mathbf{s}$) goes through three convolutional layers ($7 \times 7$ with stride 2 and 64
filters, then two $5 \times 5$ layers with 32 filters each, all with ReLU) and three fully connected
layers (two ReLU, one linear) to an action $\mathbf{a}$, drawn as a red arrow on the same scene,
with $\theta$ pointing at every layer.

![Slide 36 — How to represent behavior with a neural network?](../raw/images/01-class-intro/slide-36.jpg)

*Slide 36 — the policy as a convolutional network from camera image to action, with θ marking that every layer's weights are learned; below it, the roll-out loop. [Deck](https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf)*

## Using a policy: roll-outs and episodes

Running a policy is a loop (slide 36, ≈33:58–34:44):

1. observe the state $\mathbf{s}_ t$;
2. take an action $\mathbf{a}_ t$, for example by sampling from $\pi_\theta(\cdot \mid \mathbf{s}_ t)$,
   which means a forward pass through the network;
3. observe the next state $\mathbf{s}_ {t+1}$, sampled from the unknown world dynamics
   $p(\cdot \mid \mathbf{s}_ t, \mathbf{a}_ t)$, and repeat.

The result is a trajectory $\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T$, also
called a policy **roll-out** or an **episode**. An episode is one attempt at the task. "Roll-out"
describes how the trajectory is produced, by unrolling the alternation of policy and dynamics
(≈34:44).

## Policies for observations need memory

A policy given the true state needs only the current state as input. A policy given only
observations needs **memory**, a window of past observations:

$$\pi_\theta(\mathbf{a}_ t \mid \mathbf{o}_ {t-m}, \ldots, \mathbf{o}_ t)$$

usually implemented with a sequence model, as language models do with the previous turns of a
conversation (slide 36, ≈35:29). Memory is avoided when possible because long histories use a lot
of memory on the computer or GPU. Adding the current observation to a policy that already has the
true state adds nothing, since the state has strictly more information (≈36:15–37:01). An
alternative for observation-only problems is a two-stage policy that first infers the state and
then acts on it, worthwhile when there is a good idea of what the underlying state should look
like (≈37:46).

## Why stochastic policies?

CS224R's policies are distributions, not deterministic functions, for two reasons (Lecture 1,
slide 39, ≈47:01–47:47):

- **Exploration.** To learn from its own experience, an agent must try different things, the way
  someone learning tennis should not repeat the same technique over and over.
- **Modeling stochastic behavior.** Data collected from people or other systems shows varying
  strategies, and a model of how they approach a problem has to be stochastic too.

This is where tools from **generative modeling** come in: a policy can be a generative model over
actions given states or observations.

Lecture 2 shows why the choice of distribution matters. When demonstrators do the same thing in
different ways, a policy that outputs a single action, or a single Gaussian, learns the average of
their actions, which can be an action none of them took (≈8:32–10:05). The network should output
the parameters of an expressive distribution instead, such as a Gaussian mixture, a discretized
autoregressive model or a diffusion model, trained to maximize the log-likelihood of the
demonstrated actions. And making the network bigger does not help on its own, because "neural
network expressivity is often distinct from distribution expressivity" (Lecture 2, ≈46:16). See
[expressive policy distributions](expressive-policy-distributions.md).

A stochastic policy is also one of the two sources of randomness in the RL objective; see
[The RL objective](rl-objective.md#where-the-randomness-comes-from).

Lecture 3 returns to both points when it asks why a policy gradient samples several trajectories
from the same policy: the policy is not deterministic, so different samples explore different
strategies, and the environment may be stochastic, so even a deterministic policy sees different
outcomes worth learning about (≈29:48–30:34).

## Deterministic greedy policies, and exploring around them

Lecture 6's value-based method uses a policy that is deterministic and is not a network at all:
given a Q-function, take the action with the highest Q-value with probability one (slide 8,
≈12:32–14:05),

$$\pi'(\mathbf{a}_ t \mid \mathbf{s}_ t) = \begin{cases} 1 & \text{if } \mathbf{a}_ t = \arg\max_{\mathbf{a}} Q^\pi(\mathbf{s}_ t, \mathbf{a}) \cr 0 & \text{otherwise} \end{cases}$$

which is at least as good as $\pi$ whatever $\pi$ was, given an accurate $Q^\pi$ (see
[Q-learning](q-learning.md)). Being deterministic, it is one reason not to collect data with it
(≈18:48–19:34). So [Q-learning](q-learning.md) collects data with a
stochastic exploration policy built around it (slide 14, ≈30:19–32:37). **Epsilon-greedy** takes a
uniformly random action with a small probability $\epsilon$ and the greedy action otherwise, often
starting with a larger $\epsilon$ and lowering it during training. **Boltzmann exploration** takes
actions with probability proportional to $\exp(Q_\phi(\mathbf{s}_ t, \mathbf{a}_ t))$, more often
the higher their Q-value. The policy deployed at the end is the greedy one (slide 15).

## A planner as a "policy"

Lecture 11's planning methods learn no policy at all. A dynamics model and a reward model are learned,
and at every state the agent optimizes a sequence of actions against them and executes the result
(slide 12, ≈28:46–29:32). Finn puts "policy" in quotes for this: it maps a state to an action, or to
several future actions, but through an iterative optimization process rather than a compact network or
closed form, and nothing but the two models' parameters is stored (≈39:35–41:08). It costs compute at
every step, in real time. Distilling the planner's actions into a network turns it into a policy in the
usual sense (slide 25, ≈1:02:51). See [planning with a learned model](planning-with-models.md).

## Task-conditioned policies

A multi-task policy takes the task as an input as well as the state,
$\pi_\theta(\mathbf{a} \mid \bar{\mathbf{s}}, \mathbf{z}_ i)$, where $\bar{\mathbf{s}}$ is the original
state and $\mathbf{z}_ i$ a task identifier: an index, a language description, a video, or a goal state
(Lecture 12, slides 15–17 and 23). The input is then inherently multimodal, an image beside a string or a
one-hot vector, so the architecture has to combine them: BC-Z encodes a language instruction or a human
video into an embedding that conditions the layers of its vision network, and modern policies such as
OpenVLA pass the instruction as a prompt to a fine-tuned language model that also reads the image (slides
19–20, ≈47:44–51:37). See [multi-task RL](multi-task-rl.md).

## Policies that learn in context

The black-box meta-RL policy of Lecture 13 is a sequence model whose memory has a different job: it reads
all of its experience in the current task, each state together with the previous reward and possibly the
previous action, and keeps its hidden state **across episodes** of the same task, cutting it only when a
new task begins (Lecture 13, slides 16–17, ≈20:45–24:34, ≈27:43). Written as a policy that conditions
on a small training set of experience $D_{\text{train}}$,
$\mathbf{a} \sim \pi^{\text{task}}(\cdot \mid \mathbf{s}, D_{\text{train}})$ (slide 14), it adapts to a
new task without any gradient step. Its inputs differ from a standard recurrent policy's in that the
reward is one of them, so a reward is needed even at test time (≈22:16), and past actions are worth
passing in when the policy is stochastic, since its sampled actions are not otherwise in its memory
(≈23:02). Such a policy can be read as a multi-task policy whose task identifier is its own experience
(slide 23, ≈42:16). See [meta-RL](meta-rl.md).

## Adapting from a history of states and actions

Lecture 17's RMA splits the job of memory differently. A base policy acts from the current state $x(t)$, the
previous action $a(t-1)$ and a short vector $z_t$ describing the environment, the **extrinsics**; an
**adaptation module** estimates $z_t$ from the last 50 states and actions, $x(t-50), a(t-50)$ through
$x(t-1), a(t-1)$ (Lecture 17, slides 18–20, ≈11:32–13:03). The discrepancy between commanded and actual
movement in that history tells it, for example, that a foot slipped, so the policy adapts within a fraction of
a second, "closer to in context learning where you take the history of everything and then the learning
happens in the feed forward pass itself" (≈20:05). The vision policy uses RNNs to integrate information over
time, enough context for walking though they do not preserve very long-horizon information (≈33:59–34:45).

Kumar's answers to students add two cautions (≈24:44–27:45). The previous action is worth keeping as an
input, because "the state is lossy" and the last command lets the policy be smooth and correct itself. But
most successful sim-to-real policies use only the current state, which makes them "state indexed" rather than
indexed by state and time; a long history in simulation gives the policy room to exploit the simulator, the
environment or the reward, as when a robot learns to brace only when it expects a periodic push. See
[sim-to-real transfer](sim-to-real.md).

## Hierarchical policies

For long-horizon tasks, Lecture 15 splits a policy in two: a high-level policy
$\pi_{\text{HL}}(\cdot \mid \mathbf{o}_ t, \text{prompt})$ that predicts an intermediate goal $\mathbf{g}_ t$,
and a low-level policy $\pi_{\text{LL}}(\cdot \mid \mathbf{o}_ t, \mathbf{g}_ t)$, a goal-conditioned policy
of the kind above, that pursues the goal by predicting actions at a higher frequency (Lecture 15, slides 7
and 8, ≈7:49–12:27). The goal can be language, an image or a goal state; trained end to end with a latent
goal, the pair is just "a flat policy" with a particular architecture (slide 16). A single policy can
instead predict the goal and then the action, as in chain of thought (slides 10–11, ≈16:20–17:53).
See [hierarchical RL and imitation](hierarchical-rl.md).

## Initializing a policy for online RL

An online RL algorithm starts by initializing the policy, as when training any neural network:
randomly, with imitation learning, or with heuristics, depending on the problem (Lecture 3,
slide 6, ≈2:27–3:14). Training then needs the gradient of
$\log \pi_\theta(\mathbf{a} \mid \mathbf{s})$ for actions the policy took: a forward pass gives
the distribution over actions, the action's log-probability is read off it, and a backward pass
through the network gives the gradient, which automatic differentiation computes (≈19:44–21:16).
See [policy gradients](policy-gradients.md).

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md#representing-behavior-the-policy): the policy as a
  neural network, roll-outs, memory, and why policies are stochastic.
- [Lecture 2 — Imitation Learning](02-imitation-learning.md#learning-distributions-with-neural-networks):
  the policy as a network that outputs a distribution's parameters, and the generative models that
  make it expressive.
- [Lecture 3 — Policy Gradients](03-policy-gradients.md#estimating-the-gradient-reinforce): how a
  policy is initialized for online RL, and how its log-probabilities are differentiated.
- [Lecture 6 — Q-Learning](06-q-learning.md#what-data-to-collect-epsilon-greedy-and-boltzmann-exploration):
  the deterministic greedy policy defined by a Q-function, and the epsilon-greedy and Boltzmann
  exploration policies built around it.
- [Lecture 11 — Model-Based RL](11-model-based-rl.md#approach-1a-planning-by-backpropagation): a
  planner as a "policy" in quotes, and distilling it into a network.
- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md#conditioning-on-the-task-in-practice): policies
  conditioned on a task identifier, and the architectures that take one.
- [Lecture 13 — Meta RL](13-meta-rl.md#black-box-meta-rl): sequence-model policies conditioned on their
  own experience, with memory across episodes.
- [Lecture 15 — Hierarchical RL and IL](15-hierarchical-rl.md#the-main-idea-high-level-and-low-level-policies):
  high-level and low-level policies, and how a hierarchical policy is rolled out.
- [Lecture 17 — Advancing Robot Intelligence](17-sim-to-real.md#deploying-it-extrinsics-from-the-observation-history):
  a policy that adapts from its recent history of states and actions, and when a longer history hurts.
