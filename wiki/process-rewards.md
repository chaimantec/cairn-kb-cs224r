# Process rewards and per-step credit

How to tell which steps of a sampled solution helped and which hurt, when the only reward is whether
the final answer is right. [Lecture 10](10-rl-for-reasoning.md) answers with value functions: the
Q-value of a step is the chance of finishing correctly after taking it, estimated by rolling out a
policy from the prefix, and the advantage of a step is how much it changed that chance. This page
collects that machinery. For the problem it solves and the methods that use it, see [RL for LLM
reasoning](llm-reasoning.md).

## The problem: one reward for a whole solution

In the reasoning MDP, a solution is a sequence of steps and the reward, 1 or 0, arrives only with the
final answer (Lecture 10, slide 7, ≈10:03). A correct solution can contain **spurious steps**,
incorrect or irrelevant steps that happened not to derail it, like slide 12's computation of
100 times 2 that the next line reverses (≈29:13). A failed solution can contain good steps. Imitating
whole correct solutions, as rejection finetuning does, teaches the spurious steps along with the good
ones, and its test error eventually rises (slides 11–12, ≈24:37–28:26). The idea of slide 14 is to
assign credit to individual steps: "If we can identify spurious steps somewhat *precisely*, then we
can avoid this issue".

## Rollouts from each prefix

Take one sampled solution. For each prefix, the problem plus the first few steps, ask a **rollout
policy** to complete the solution many times (Lecture 10, slide 15, ≈33:06–34:37). If the
completions mostly succeed, the prefix probably did no harm. If they always fail, something in it is
wrong. If they sometimes succeed, it is "sort of OK". The fraction of successes is an estimate of the
rollout policy's expected future 0/1 reward from that prefix, which is a value function: "This is
equivalent to the value function of the *rollout* policy…." (slide 15, ≈35:22).

![Lecture 10, slide 15 — Negative On-Policy Data Gives Advantages](../raw/images/10-rl-for-reasoning/slide-15.png)

*Lecture 10, slide 15 — from each boxed step of a failed solution, orange rollouts are run to completion; green circles are successes and red dots failures, so a step whose rollouts mostly succeed is likely not spurious. [Deck](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf)*

## The Q-value of a step

With $\boldsymbol{x}$ the problem, $\hat{\boldsymbol{y}}_ i$ the $i$-th step of the sampled solution,
$\hat{\boldsymbol{y}}_ {1:i-1}$ the steps before it, $\tilde{\pi}$ the rollout policy,
$\boldsymbol{y}^{\text{new}}_ {i+1:L}$ the steps it samples to finish the solution, and $r$ the 0/1
reward against the reference solution $\boldsymbol{y}$ (Lecture 10, slide 16, ≈36:10–36:55):

$$Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\thinspace \hat{\boldsymbol{y}}_ i) = \mathbb{E}_ {\boldsymbol{y}^{\text{new}}_ {i+1:L} \sim \tilde{\pi}(\cdot \mid \boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i})} \left[ r\left( [\hat{\boldsymbol{y}}_ {1:i}, \boldsymbol{y}^{\text{new}}_ {i+1:L}], \boldsymbol{y} \right) \right]$$

The state is the problem and the prefix; the action is step $i$. This is the course's
[Q-function](value-functions.md#the-q-function-qpi) with discount factor $\gamma = 1$ and a reward
that is zero until the end, so the expected sum of rewards collapses to the expected final reward
(≈37:42–38:27).

## The advantage of a step

The advantage is usually $A^\pi(\mathbf{s}, \mathbf{a}) = Q^\pi(\mathbf{s}, \mathbf{a}) - V^\pi(\mathbf{s})$.
In the reasoning MDP each state is the previous state with a step appended, so the value of the state
before step $i$ is the Q-value of step $i-1$, and the advantage is a difference of consecutive
Q-values (Lecture 10, slide 17, ≈40:47):

$$A_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\thinspace \hat{\boldsymbol{y}}_ i) = Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\thinspace \hat{\boldsymbol{y}}_ i) - Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-2};\thinspace \hat{\boldsymbol{y}}_ {i-1})$$

It is the "relative change in the value function having committed to a step" (slide 17). In the
slide's example, a failed solution's steps have Q-values 0.66, 0.0, 1.0, 0.25 and 0.0, so their
advantages are −0.66, 1.0, −0.75 and −0.25. A negative or low advantage marks a bad or spurious step
not worth learning from; a positive one marks a good step, even inside a failed solution
(≈41:32–42:19). Breaking a long trace's success into its steps' successes is what lets a model learn
the step that went wrong rather than guess, and that is where the gain in data efficiency comes from
(≈43:05–43:52).

![Lecture 10, slide 17 — Negative On-Policy Data Gives Advantages](../raw/images/10-rl-for-reasoning/slide-17.jpg)

*Lecture 10, slide 17 — the advantage of a step as its Q-value minus the previous step's, worked through on a failed solution: the arrows show which two Q-values each advantage subtracts. [Deck](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf)*

## Choosing the rollout policy

The definition works for any rollout policy (Lecture 10, ≈38:27). Kumar's view is that it should
differ from the policy being trained, which could otherwise fail in the same way. Part of the NLP
community uses the same policy and finds it works reasonably, and no apples-to-apples comparison has
settled the question (≈39:13). What makes a rollout policy good is a balance of quality and
diversity (≈53:50–55:22). If the model mostly fails, the rollout policy must succeed sometimes, or
there is no success to learn from. If the model is decent, diverse completions matter more, because
they cover the ways it fails. Kumar says one of the papers writes down an exact expression for this
trade-off, which the lecture does not cover. Slide 26, not read out, states it informally: the
optimal rollout policy "produces advantages that **most effectively distinguish** good and bad steps
from *all*" policies. The slides-only result on slide 28 points the same way. On problems that SFT
never solved in 256 tries, training with the rollout policy's advantages $A_{\tilde{\pi}}$ solves
about 15% at 256 samples, against about 3% with the current policy's own advantages $A_\pi$.

## Rollouts, or a learned model

Estimating the advantages by rollouts is one kind of work. The other fits a network to predict them
(Lecture 10, ≈42:19–43:05). In the NLP community, a model that scores every step is called a
**process reward model** (PRM), because it scores the process rather than only the final answer, as
the reward models of [RLHF](rlhf.md) do. Kumar does not think PRMs and Q-values are necessarily the
same thing semantically. Kumar's group fits the network to the advantages and calls it a **process
advantage verifier** (PAV). The name breaks down as process because it rewards every step, advantage
because it fits an advantage function, and verifier because reward models are often called verifiers
in reasoning (slides 25–26, ≈1:03:52–1:06:10).

Splitting a solution into steps is itself a choice. Lecture 10's work splits at every sentence;
models can be prompted to write in a style that heuristics can segment, and some work marks each step
with a "step hash one colon" convention (≈56:09–57:41).

## Three uses

- **Filter what to imitate.** Keep only positive-advantage steps, each trained given its own prefix
  (Lecture 10, slide 18, ≈43:52–46:58). This is filtered behavior cloning on steps, a cousin of
  [advantage-weighted regression](implicit-policy-constraints.md#awr-monte-carlo-advantages), and it
  stops the rising error of rejection finetuning (slide 19, ≈50:48).
- **Build preference pairs for offline RL.** A successful rollout and a failing one from a shared
  prefix make a pair for [DPO](direct-preference-optimization.md#per-step-preference-pairs-for-reasoning),
  which needs about 8x less data than SFT (slides 20–23, ≈51:34–59:59).
- **Reward each step in online RL.** Add the PAV's advantages to the final reward as a dense bonus:
  5–6x the sample efficiency of the outcome reward alone, and 6–7% higher accuracy (slides 25–27,
  ≈1:04:39–1:06:56).

![Lecture 10, slide 27 — Results: Dense-Reward RL Performance](../raw/images/10-rl-for-reasoning/slide-27.jpg)

*Lecture 10, slide 27 — accuracy over training iterations for PAV-RL (solid) and outcome-reward ORM-RL (dashed) on 2B and 9B models; the horizontal arrows mark the 6x and 5x sample-efficiency gains and the vertical ones the 7% and 6% accuracy gaps. [Deck](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf)*

## Relation to earlier lectures

The construction reuses the course's [value functions](value-functions.md): a Q-value under some
policy, and an advantage that compares an action with what the policy would get anyway. Filtering by
advantage is [Lecture 7](07-offline-rl.md)'s idea of weighting imitation by the advantage, and Kumar
confirms that it is "exactly" related to trajectory stitching and advantage weights (Lecture 10,
≈39:13–40:00). A process reward model is RLHF's reward model, applied to each step instead of the
whole response (≈1:03:52).

## Where this appears

- [Lecture 10 — RL for LLM Reasoning](10-rl-for-reasoning.md#assigning-credit-to-steps-with-rollouts):
  the whole construction, its uses and the student questions on it.
- [RL for LLM reasoning](llm-reasoning.md): the methods that use it.
- [Value functions](value-functions.md): the Q-function and advantage it applies to solution prefixes.
- [Implicit policy constraints](implicit-policy-constraints.md): advantage-weighted imitation.
- [Reward learning](reward-learning.md#process-rewards-for-reasoning): learned rewards, here per step.
