# Lecture 14 — Exploration

Stanford CS224R, Spring 2025 (Friday, May 16) ·
[video](https://www.youtube.com/watch?v=4tlSKdi8teU) (73 min) ·
[slides (PDF)](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf) ·
[slide-by-slide text](../raw/slides/14-exploration.md) ·
[transcript](../raw/transcripts/14-exploration.md)

**What this lecture establishes.** Exploration is the problem of finding rewarding behaviour that no
single action reveals, and of deciding, at every step, whether to try something new or do the best
thing known so far. It is easy to state and hard to do well: a game like Montezuma's Revenge, with
sparse rewards whose meaning the agent cannot know, defeats RL methods that master Breakout. The
lecture first studies exploration where it can be analysed, in **multi-armed bandits**, measuring an
algorithm by its **regret** and introducing two families of strategies, **optimism under uncertainty**
(upper confidence bound) and **probability matching** (posterior sampling), which the class then
plays against in a drug-trial game. In large MDPs such as robot control and language modelling, Finn
argues, exploration from scratch is intractable, so in practice exploration comes from demonstrations,
pretrained base models and shaped rewards. The second half returns to Lecture 13's problem of
**learning to explore** in meta-RL and presents **DREAM**, which decouples the exploration policy from
the execution policy: the execution policy is trained with the task identifier and a variational
information bottleneck that keeps only task-relevant information, and the exploration policy is
rewarded for the information it gathers about that bottlenecked representation. DREAM is part of
Homework 4. The lecture ends with an application: exploring student programs to find bugs and help
grade them.

Slide numbers below are the numbers printed on the slides, which equal the PDF page numbers.
Timestamps (≈M:SS) point into the video and the [transcript](../raw/transcripts/14-exploration.md).
**The recording covers slides 3–32.** It opens with an overview that follows slide 3's plan (≈0:05)
without discussing slide 2's reminders, and ends at about 1:12:40 on slide 32's grading results. The
closing plan and reminders (slides 33–34) are not shown; the
[section on them](#not-in-the-recording-the-plan-and-reminders) gives them from the
[slide file](../raw/slides/14-exploration.md).

Slide 12 prints only "Exploration game": the game itself runs on a website the PDF does not contain
(≈24:53–31:54), so this page describes it from the transcript. Finn also draws on the slides or a board
the recording does not show: regret curves on slide 10 (≈12:32–18:46), the noise and bottleneck
construction (≈46:34–49:41), four example tasks (≈51:13), the execution objective (≈52:46–55:52) and the
exploration reward (≈1:00:32–1:03:38). Where the transcript says "this" or "right here", this page
reconstructs what was meant from the surrounding words and the slides, and says so. Videos play on
slides 27, 30 and 31; the PDF holds only a frame of each.

## Contents

1. [What is the exploration problem?](#what-is-the-exploration-problem)
2. [Exploration and exploitation](#exploration-and-exploitation)
3. [Can we explore optimally? Bandits](#can-we-explore-optimally-bandits)
4. [Measuring exploration: regret](#measuring-exploration-regret)
5. [Two strategies for bandits](#two-strategies-for-bandits)
6. [The exploration game](#the-exploration-game)
7. [Exploration in robotics and language models](#exploration-in-robotics-and-language-models)
8. [Exploring at test time: meta-RL revisited](#exploring-at-test-time-meta-rl-revisited)
9. [Solution 3.0: explore to identify the task](#solution-30-explore-to-identify-the-task)
10. [Decoupling with a task representation](#decoupling-with-a-task-representation)
11. [Aside: bottlenecking a representation](#aside-bottlenecking-a-representation)
12. [Training the execution policy](#training-the-execution-policy)
13. [Rewarding the exploration policy: DREAM](#rewarding-the-exploration-policy-dream)
14. [Theory and experiments](#theory-and-experiments)
15. [Three ways to learn to explore](#three-ways-to-learn-to-explore)
16. [Application: exploring student programs](#application-exploring-student-programs)
17. [Not in the recording: the plan and reminders](#not-in-the-recording-the-plan-and-reminders)
18. [Summary](#summary)

## What is the exploration problem?

The lecture starts with exploration in standard RL settings, mostly bandits, then how that translates
to robotics and large language models, and then revisits Lecture 13's meta-RL to ask how an agent can
learn to explore (slide 3, ≈0:05). Slide 3 marks the meta-RL part, "Efficiently Meta-Learning Optimal
Exploration", as "Part of HW 4".

Applied to some problems, RL does very well. Trained to play Breakout with the game's score as the
reward, an agent becomes a performant player that gets a high score; trained on Montezuma's Revenge,
RL systems find it really, really hard to play (slide 4, "this is easy (mostly)" against "this is
impossible", ≈0:05–0:50).

![Slide 4 — What's the problem?](../raw/images/14-exploration/slide-4.jpg)

*Slide 4 — Breakout (left, "this is easy (mostly)") against Montezuma's Revenge (right, "this is impossible"): the same kind of agent masters one and fails at the other. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Why the difference? In Montezuma's Revenge you get a reward for getting a key and for opening a door;
there is a skull rolling around that people know to avoid, and falling off a ledge is something you
would avoid, but the agent does not know any of that itself (slide 5, ≈0:50–1:37). Getting killed by
the skull earns "nothing (is it good? bad?)". The rewarding events are sparse and discrete, and
"Finishing the game only weakly correlates with rewarding events"; each rewarding event is also few
and far between. At the start the agent gets no reward at all, and it needs a long sequence of
actions even to reach the key. People know what to do because "we **understand** what these sprites
mean!", and the algorithm does not (≈1:37–2:23).

To see the problem from the algorithm's side, Finn uses a card game called Mao, "half a game, half a
joke in some ways" (slide 6, ≈2:23–3:10). You are not told the rules in advance; breaking a rule
incurs a penalty without telling you what the rule was; you can only discover the rules through trial
and error, and they don't always make sense to you. Temporally extended tasks like Montezuma's Revenge
get harder the longer the task and the less you know about the rules. An RL agent is in the position
of someone whose "goal in life was to win 50 games of Mao…" "(and you didn't know this in advance)":
it learns that goal only from the rewards for winning and penalties for losing, which is really, really
hard, especially with sparse rewards (≈3:10–3:57).

![Slide 6 — Put yourself in the algorithm's shoes](../raw/images/14-exploration/slide-6.jpg)

*Slide 6 — the card game Mao, where the only rule you may be told is that there are rules you will be penalised for breaking: the position an RL agent is in. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

## Exploration and exploitation

Slide 7 gives "Two potential definitions of exploration problem" (≈3:57–4:44). One asks how an agent
can discover high-reward strategies "that require a temporally extended sequence of complex behaviors
that, individually, are not rewarding": seeking out rewards by finding sequences of actions that pay
off only together. The other asks how an agent can decide "whether to attempt new behaviors (to
discover ones with higher reward) or continue to do the best thing it knows so far". These are
"Actually the same problem" (≈4:44–5:31). **Exploitation** is "doing what you know will yield highest
reward": doing the things you have seen before that had high reward, acting according to your Q-function,
for example. **Exploration** is "doing things you haven't done before, in the hopes of getting even
higher reward". An agent maximizing a sum of rewards over time has to balance the two.

Slide 8's examples, from David Silver's lecture notes (≈5:31–6:19): in a new city, exploitation is going
to your favourite restaurant and exploration is trying a new one. In online ad placement, one very
common use of RL, exploitation shows the most successful advertisement and exploration shows a
different random one. In oil drilling, exploitation drills at the best known location and exploration
at a new location.

## Can we explore optimally? Bandits

To design an exploration algorithm, one might ask for the best strategy, the best trade-off between
exploration and exploitation, but first one has to ask what optimality in exploration even means
(slide 9, ≈6:19–7:05). Slide 9 lays RL problems along a line from "theoretically tractable" to
"theoretically intractable": multi-armed bandits ("1-step stateless RL problems"), contextual bandits
("1-step RL problems", which add features of the state you are in), small tabular MDPs, large MDPs with
known low-rank features, and "large MDPs without known special structure". At the simple end one can
analyse whether a strategy is optimal; at the complex end it is intractable both to analyse optimality
and to develop algorithms that explore well (≈7:05). Most problems in this course are large MDPs without
known structure, the hardest both for good exploration strategies and for thinking about exploration
theoretically (≈7:54).

![Slide 9 — Exploration is hard](../raw/images/14-exploration/slide-9.png)

*Slide 9 — RL problems from theoretically tractable (multi-armed bandits) to theoretically intractable (large MDPs without known structure); the lecture starts at the left end. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

So the lecture spends some time on bandits: they give a clean way to think about exploration, and they
are used in practice for a variety of problems (≈7:54). A bandit, "not the kind of someone that runs
around and maybe tries to steal stuff", is a formulation of RL in which the horizon is one time step per
episode and there is no state, and in which, unlike the MDPs so far in the course, the reward is
stochastic (≈7:54–8:40). Choosing a restaurant is a bandit problem: each episode you select one
restaurant; your experience in previous episodes informs whether to pick a new restaurant, about whose
reward you know nothing, or one that gave you high reward before; and the reward is stochastic because
your experience at a restaurant is not the same every time (≈8:40–9:26).

## Measuring exploration: regret

How good is an exploration algorithm? Compare the reward it gets with what acting optimally would get
(slide 10, ≈9:26–10:13). Where the course has so far looked at sums of rewards, often averaged over
episodes, bandit analysis looks at the whole RL process, exploration followed by exploitation (≈10:13).
After $T$ episodes, the best one could have done is the expected reward of the optimal action times
$T$. That is an oracle, an upper bound: at the start you know nothing about the problem and have to take
a random action, so it is the best you could have done in retrospect (≈11:01). Subtract the sum of the
rewards actually observed from the actions taken, one per episode, and the gap is a kind of optimality
gap called **regret**, how much we regret the actions we took compared with the optimal one
(≈11:01–11:47). Slide 10 writes it as

$$\mathrm{Reg}(T) = T\thinspace E[r(a^\star)] - \sum_{t=1}^{T} r(a_ t)$$

where $\mathrm{Reg}(T)$ is the "regret after doing RL for $T$ episodes", $T\thinspace E[r(a^\star)]$ uses the
"expected reward of best action (the best we can hope for in expectation)", $a^\star$ being the best
action, and $r(a_ t)$ is the "reward of action actually taken" in episode $t$. Here time indexes
episodes, since each episode is a single time step (≈12:32).

![Slide 10 — How do we measure optimality in bandits?](../raw/images/14-exploration/slide-10.png)

*Slide 10 — regret after T episodes: T times the expected reward of the best action, minus the rewards of the actions actually taken. Finn's regret curves were drawn over this slide and are not in the PDF. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Finn then draws regret against the number of episodes; the drawing is not in the PDF ("The blue markers
never work that well", ≈14:52), so what follows is from the spoken description (≈12:32–15:39):

- **Acting optimally** gives regret of zero, staying at zero as the episodes accumulate, in expectation
  (there might be some randomness). This is not achievable in practice, because the first action has
  to be random.
- **Acting at random**, or always picking an action $a_1$ that is not the best, incurs the same expected
  shortfall in every episode, so regret grows **linearly**. Linear regret means the algorithm is not
  improving: it is not using exploration to get better rewards.
- **A smarter algorithm** has **sublinear** regret: the curve starts out looking roughly linear while
  the agent explores and gathers information, then flattens out once it reaches the optimal behaviour,
  in the exploitation phase.

Students' questions about these curves (≈15:39–19:31) fill in what the formula means. Each curve is a
single RL process, drawn smooth; in practice it will be noisier, since an unlucky step can push it up
and a lucky one push it down even when the optimal action was taken. The first term has an expectation
and the second does not because the first is the optimal expected reward after $T$ steps and the second
is the reward actually observed; taking an expectation over runs of the RL algorithm would average the
differently-shaped curves you get from different luck (≈16:27–17:13). Random actions give a noisy curve
that is linear in expectation, and with deterministic rewards and an agent that always picks $a_1$ it
would be an exact line (≈17:13–17:58). Asked how the formula could be measured at all, Finn explains
that regret is "more of an object that's used for analysis and not our optimization objective": we aim
to minimize regret, but cannot measure it without knowledge of the bandit or the MDP (≈17:58–18:46).
With an expectation the curve is monotonic; without one, because the reward is stochastic, it can go
down, for instance when the best action gives a good reward 90% of the time and the agent happens to
get a good reward, relative to the oracle that averages over outcomes (≈18:46–19:31).

There is a variety of relatively simple exploration strategies, including some that can be shown to be
optimal, and algorithms can often be analysed to give theoretical guarantees on regret; slide 10 adds
"Variety of optimal algorithms (up to a constant factor)" and "But empirical performance may vary…"
(≈19:31–20:17).

## Two strategies for bandits

Slide 11 sets out two approaches for bandits (not for MDPs), in two columns (≈20:17–24:53).

![Slide 11 — Exploration in Bandits](../raw/images/14-exploration/slide-11.png)

*Slide 11 — optimism when uncertain (UCB: add a bonus for uncertainty to the average reward) beside probability matching (posterior sampling: form beliefs, sample an MDP, act optimally in it, repeat). [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

**Optimism when uncertain**, for example the upper confidence bound (UCB). If you have not tried a
restaurant, be optimistic about it, which incentivizes trying new restaurants instead of always being
greedy about the ones you have tried (≈20:17–21:02). Track the average reward $\bar{r}_ a$ received for
each action $a$. To exploit, take the action with the maximum average reward,
$a_{\mathrm{exploit}} = \arg\max_ a \bar{r}_ a$. To explore, still favour actions with high average
reward, so that a really bad restaurant is not revisited, but add a term from "some kind of variance
estimate" $\sigma_ a$ of the reward estimate, scaled by a constant $C$:

$$a_{\mathrm{explore}} = \arg\max_ a \bar{r}_ a + C\sigma_ a$$

A high variance means the agent is uncertain about that action's reward, and the term acts as a reward
bonus that encourages exploring uncertain things (≈21:02–21:49). The intuition, "assume unknown = good",
is that unknown things are taken to be good until observed; if they turn out to be bad, the variance and
the reward estimate both go down and the agent acts accordingly, which balances exploration and
exploitation. Asked how an action that has never been seen gets chosen, Finn says its variance estimate
would be something like infinite, so the rule explores every untried action at least once before the
uncertainty about each comes down over time (≈21:49–22:35). One version of this popular approach, which
has derivable guarantees, is called upper confidence bound (≈22:35).

**Probability matching**, for example posterior sampling, is harder to show guarantees for but can work
pretty well in practice; something like it appeared in the previous lecture (≈22:35–23:21). Assume each
arm's rewards come from a distribution, $r(a_i) \sim p_{\psi_i}(r)$ for each arm $a_i$, with unknown
parameters $\psi_i$. That "Defines a POMDP with $\mathbf{s} := [\psi_1, \ldots, \psi_n]$", a partially observed
MDP whose hidden state is the reward parameters: you do not know whether you will get a good or a bad
reward (≈23:21–24:07). Then loop:

1. "Form beliefs $\hat{p}(\psi_1, \ldots, \psi_n)$ from data": model the parameters and your uncertainty about
   them from the data seen so far.
2. "Sample an MDP $\psi_1, \ldots, \psi_n \sim \hat{p}(\cdot)$" from the model of possible worlds.
3. "Take optimal action $a$ assuming MDP is correct", then update the beliefs and sample again.

The intuition: "iteratively improve model of bandit, act according to one model" (≈24:07–24:53).

These methods are widely used and studied in industry for recommenders and ad placement (slide 11's
footer, ≈24:53). Everything so far is for small bandit problems, but they are relevant to a number of
practical problems even without longer episodes or state.

## The exploration game

To build intuition, the class plays a simulation from a website (slide 12, "Exploration game";
≈24:53–31:54). The game is not in the PDF. You have been hired as head of drug development at a
prestigious research hospital; there is a plague; and you want to save lives by giving patients the
drug that is most effective. It is a bandit problem: there are maybe around 5,000 patients and around 10
drugs whose efficacy is unknown, and the goal is to save as many cumulative lives as possible, balancing
saving lives now against learning about the drugs to save lives later (≈25:39–26:26).

Students call out drugs and the site reports whether each patient lived or died (≈26:26–28:48). Drugs
1, 4 and 7 look better than 10 early on; after more tries Finn suggests that "Maybe we want to start
exploiting" (≈27:14), the class keeps returning to 7, tries others, and settles on 9, which keeps
working. The site then shows the true efficacy of each drug: 9 was the best, so the class found it, and
7 was also pretty good (≈28:48). Two plots compare the class, in purple, with algorithms, in a single
run: on lives saved (higher is better) the class did pretty well relative to the algorithms, and on
regret over time it did better than all of them, though "maybe there's some luck involved" (≈29:34).

In the comparison, greedy (green) just picks the best thing seen so far and gets a lot worse over time,
especially with more trials, whether patients or ad placements. Posterior sampling (yellow), which here
models the probability of each drug's efficacy and picks a better drug based on that estimate, is "by
far the best algorithm". Epsilon-greedy (red) is a bit more greedy than UCB (blue), so it does better
with fewer patients, but in the long run UCB does better because it makes more sure of exploring more
actions and can then take more optimal actions (≈28:48–30:21).

*[Student question]* Is there a mathematical reason why posterior sampling does better than UCB and
epsilon-greedy? Finn does not go into the math: for this problem one could probably analyse posterior
sampling, but in general it can be hard to derive theoretical guarantees for it (≈30:21–31:07). For
epsilon-greedy, much comes down to the value of $\epsilon$ (the probability of taking a random action)
and tuning it; $\epsilon$ is often made smaller over time, and that hyperparameter can drastically
affect performance, since with a constant $\epsilon$ the agent keeps occasionally prescribing a random
drug long after it knows a lot about the drugs, which is a very poor choice (≈31:07–31:54).
Epsilon-greedy first appeared in [Lecture 6](06-q-learning.md#what-data-to-collect-epsilon-greedy-and-boltzmann-exploration)
as Q-learning's exploration policy.

## Exploration in robotics and language models

What does this look like for robotics, language models and larger MDPs? "Different people have
different opinions on this", and Finn's is "somewhat pessimistic" (slide 13, ≈31:54). There is a lot of
research translating these ideas to large MDPs, such as estimating whether an action has been taken
before and giving reward bonuses for novelty, for visiting new states or taking new actions, or, like
posterior sampling, modelling different dynamics and acting accordingly (≈31:54–32:40). But "at the end
of the day, **exploration from scratch in large MDPs is intractable**." Randomly initialize a
transformer language model and try to learn to make a user happy, or to maximize a reward like upvotes,
and it will not work: the space of possible token sequences is far too large, at least relative to the
patience most users will have (≈32:40–33:26). A robot learning from scratch even to pour water from one
cup to another would just move its arms around randomly rather than do anything useful (≈33:26).

Instead, exploration in these domains relies heavily on demonstrations, or on base models pretrained on
data and demonstrations (slide 13: "rely on demonstrations, or on base models pre-trained on
demonstrations"). RL has been quite successful in language models particularly when a strong base model
already writes coherent sentences and knows roughly how to approach certain prompts or problems, and in
robotics, demonstrations of high-reward behaviour make a very powerful starting point for RL
(≈33:26–34:12). These models provide an initialization that is excellent for exploration; slide 13
adds, in green, "These models are *excellent* at exploration!" Wherever feasible, shaped rewards help
too, because they guide any exploration process: in simulation you can hand-code dense rewards, and
single-step rewards learned from preferences also give a fairly dense signal (slide 13: "e.g.
hand-coded in simulation for legged robots, single step rewards from preferences for LLMs"; ≈34:12–34:58).
Some of the novelty ideas can be combined with demonstrations and base models, but one gets a lot
further by starting from that kind of initialization (≈34:58).

*[Student answer, inaudible]* Agreeing with a student, Finn adds that in these MDPs the state space and
the set of possible actions are enormous, and the initialization biases the model towards the part of
the state space that will get high reward: "these are the kinds of things I want you to do" (≈34:58–35:44).
See [imitation learning](imitation-learning.md), [RLHF](rlhf.md) and
[RL for LLM reasoning](llm-reasoning.md) for those starting points.

## Exploring at test time: meta-RL revisited

Even so, an exploration problem remains (slide 14 begins part 2, "Learning to Explore via
Meta-Learning"). A policy trained on a variety of things may still need to explore when it is deployed,
because even with a lot of prior experience it may not have the information it needs to get high reward
(slide 15, ≈35:44). Slide 15 gives two examples from Lecture 13: under **partial observability**, finding
ingredients in a new kitchen, where the agent cannot just start cooking but has to open drawers and
cabinets to seek out information; and in **LLM reasoning**, identifying strategies for solving a complex
problem rather than simply writing tokens that might lead to a solution (≈36:29). The slide's reasoning
example is a model's thinking about an inequality ("Alternatively, perhaps there's a way to pair terms
or use symmetry?").

![Slide 15 — What if we need to explore at test time?](../raw/images/14-exploration/slide-15.jpg)

*Slide 15 — two settings where an agent must explore at test time: finding ingredients in an unfamiliar kitchen, and trying strategies on a hard problem. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

[Lecture 13](13-meta-rl.md#exploration-in-meta-rl-end-to-end-optimization) optimized a policy with
history, or memory, which should let an agent explore and exploit: try actions, remember what it found,
and use where the ingredients are to solve the task (≈36:29–37:15). But optimizing exploration and
exploitation together with an end-to-end objective is hard because of a chicken-and-egg problem (slide
16). If the agent has not learned to explore, it cannot get high reward, because it has not found the
ingredients; and if it does not yet know how to solve the task, any exploration earns low reward. That is
the **coupling problem**: "learning exploration and execution depend on each other", which "can lead to
poor local optima, poor sample efficiency" (≈37:15).

![Slide 16 — Recall: Why is End-to-End Training Hard for Meta-Learning with Exploration?](../raw/images/14-exploration/slide-16.png)

*Slide 16 — the coupling problem as a chicken-and-egg cycle: without finding the ingredients the agent cannot learn to cook, and without being able to cook no exploration earns reward. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Lecture 13 also introduced alternative exploration strategies that are easier to optimize but can be
suboptimal (slide 17, ≈37:15–38:01): posterior sampling (also called Thompson sampling, as in PEARL),
intrinsic rewards (MAME), and task dynamics and reward prediction (MetaCURE). See
[alternative exploration strategies](13-meta-rl.md#alternative-exploration-strategies).

Slide 18 asks: "Can we avoid the chicken-and-egg problem *without* sacrificing optimality?", the best of
both worlds, and answers "Yes!". The method gets the optimal exploration-exploitation trade-off and is
also efficient to train (≈38:01).

It builds on the last alternative, task dynamics and reward prediction, which Finn writes down
(≈38:01): for different environments or tasks, fit a dynamics model and a reward model, "Train model
$f(\mathbf{s}', r \mid \mathbf{s}, \mathbf{a}, \mathscr{D}_ {\mathrm{tr}})$ & collect $\mathscr{D}_ {\mathrm{tr}}$ so that model is accurate"
(slide 19's top line), where $\mathbf{s}'$ is the next state, $r$ the reward and $\mathscr{D}_ {\mathrm{tr}}$ the
experience collected in the current task. That incentivizes the agent to seek out information in a way
that is decoupled from executing the task (≈38:48). In slide 17's picture, where different MDPs mean
navigating to different circles, rewarding trajectories that improve the reward and dynamics models
produces something like the purple trajectory, which explicitly visits states that reveal where the
reward is.

![Slide 17 — Alternative: Leverage Alternative Exploration Strategies](../raw/images/14-exploration/slide-17.jpg)

*Slide 17 — the alternative strategies recapped from Lecture 13; the purple trajectory in the plot sweeps the arc of possible goals, the exploration a dynamics-and-reward-prediction objective produces. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

This learns efficient exploration strategies but can be suboptimal (≈38:48–39:36). With distractors, such
as pictures on the wall, it will look at every picture so as to model the environment well; slide 17
gives "Lots of distractors, or complex, high-dim state dynamics" as when it might be bad. Slide 19 asks
"Do we have to learn a *full dynamics & reward model*?"

## Solution 3.0: explore to identify the task

Instead of learning a dynamics and reward model, learn to predict **what the task is** (slide 19,
≈39:36–41:09). "Idea 3.0: Label each training task with a unique ID $\mu$": the identifier could be a
one-hot index, a language description, or a more continuous description such as a vector. Then explore in
a way that lets the agent predict the task at hand. It helps to think of two policies, an exploration
policy and an exploitation (execution) policy, although in practice they can be combined (≈40:24).

![Slide 19 — Solution #3](../raw/images/14-exploration/slide-19.png)

*Slide 19 — Idea 3.0: meta-training learns an exploration policy whose experience lets a model identify the task, and a separate execution policy given the task ID; meta-testing explores, infers the task, then performs it. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

During meta-training (≈40:24–42:40):

- **Exploration policy**: train a policy $\pi^{\exp}(\mathbf{a} \mid \mathbf{s})$ and a task identification model
  $q(\mu \mid \mathscr{D}_ {\mathrm{tr}})$ so that the experience $\mathscr{D}_ {\mathrm{tr}} \sim \pi^{\exp}$ "allows accurate task
  prediction". The exploration policy produces experience in a task, and a separately trained model
  predicts the task identifier from it. The goal of exploring is to discern different tasks, which is
  completely separate from executing the task. (The slide's line ends "from $f$", although the
  identification model on the slide is $q$; the text layer confirms the printed $f$.)
- **Execution policy**: train an ID-conditioned policy $\pi^{\mathrm{exec}}(\mathbf{a} \mid \mathbf{s}, \mu_i)$, given
  the identifier $\mu_i$ of task $i$. That is just [multi-task RL](multi-task-rl.md), trained by
  optimizing reward, and it does not rely on good exploration (≈41:09–41:55).

So the execution objective does not depend on exploration, and the exploration objective, identifying
the task, does not depend on how well the task is done (≈41:55). *[Student question]* Which part is the
exploration, the model that predicts the identifier? The exploration is the exploration policy; the
identification model supplies its reward, which is basically how well the model identifies the task,
the negative error of the task identifier model (≈42:40).

At meta-test time (≈42:40–43:26), run the exploration policy to collect experience,
$\mathscr{D}_ {\mathrm{tr}} \sim \pi^{\exp}(\mathbf{a} \mid \mathbf{s})$; feed it to the identification model to infer the task,
$\hat{\mu} \sim q(\mu \mid \mathscr{D}_ {\mathrm{tr}})$; and give that ID to the execution policy to perform the
task, $\pi^{\mathrm{exec}}(\mathbf{a} \mid \mathbf{s}, \hat{\mu})$: explore, identify the task, then exploit.

The upside is "+ no longer need to model dynamics, rewards": the only model predicts the task
identifier (≈43:26). The downside: "may not generalize well for one-hot $\mu$". A one-hot identifier
probably will not generalize to new tasks, which also answers a student's question about new, unseen
task categories (≈42:40–43:26). *[Student question, inaudible]* How does this differ from black-box
meta-RL? Different MDPs are still sampled; there are two main differences. One is the objective for the
exploration part; the other is that the execution part, which maximizes rewards as black-box meta-RL
does, is given the task identifier, so it does not rely on good exploration when learning to do the task
(≈44:13).

## Decoupling with a task representation

To fix generalization, learn a more compressed representation of the task, which will generalize better
than a one-hot vector (slide 20, ≈44:13–45:46). Slide 20 splits meta-training into two steps, marked with
an egg and a hatching chick.

![Slide 20 — Solution #3: Decouple by acquiring representation of task relevant information](../raw/images/14-exploration/slide-20.png)

*Slide 20 — step 1 bottlenecks the MDP identifier (wall color, ingredients, decorations) down to the task-relevant part z for the execution policy; step 2 rewards the exploration policy for episodes that recover z. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

**1) Learn execution & identify key information.** While learning the execution policy, also identify
the information needed to solve the task. Say the identifier $\mu$ contains some relevant and some
irrelevant things: the wall colour and the decorations, and where the ingredients are. A one-hot vector
likewise carries information that may be irrelevant, since two kitchens might have their ingredients in
the same places but differ in spurious ways (≈45:00). An encoder **bottlenecks** the identifier into a
representation $z$ that compresses away what is not needed for the task; on the slide, $\mu$'s three
entries (wall color 2, ingredients 0, decorations 1) become a $z$ holding only the ingredients' 0, which
feeds the execution policy $\pi^{\mathrm{exec}}$. Such a representation should generalize better (≈45:46).

**2) Learn to explore by recovering that information.** The exploration policy $\pi^{\exp}$ learns to
explore so that it can recover the compressed representation $z$, rather than the full identifier: its
exploration episode $\tau$ earns an "Information recovery reward $\mathrm{MI}(z; \tau)$", the mutual
information between $z$ and the episode (≈45:46).

## Aside: bottlenecking a representation

How do you compress a representation in a neural network (slide 21, ≈45:46–52:46)? Take the task
identifier $\mu_i$, a network that predicts $z_i$ from it, and a policy that takes $z_i$ and the state
and outputs the action. Trained end to end, nothing incentivizes the network to compress, to remove
information about decorations, for example (≈46:34). Finn builds the answer up on the slide, and the
drawing is not in the PDF.

**V0: add noise to the representation.** Instead of passing $z_i$ on directly, add noise sampled from a
Gaussian: slide 21's $\epsilon \sim \mathcal{N}(0, I)$ and $\bar{\mathbf{z}} = \mathbf{z} + \epsilon$, with $\bar{\mathbf{z}}$ the noisy
representation (≈47:21). Noise corrupts the signal, so it "will discard information", but done only at
test time it may discard good information too (the slide prints "my discard good info"). Training with
the noise should encourage the model to discard information from $z$, but the model can compensate by
increasing the magnitude of $z_i$ until the noise is small in comparison, still passing a lot of
information through (≈47:21–48:08): "if done during training, model can increase magnitude of
$\mathbf{z}$".

**V1: variational information bottleneck.** The key ideas are "1. Add Gaussian noise during training"
and "2. Prevent the model from increasing magnitude" (≈48:08–48:55). Add the noise before passing the
representation to the next layer, and modify the training loss $L_{\mathrm{tr}}$ with a term that
minimizes the magnitude of $z$:

$$L_{\mathrm{tr}} + \Vert \mathbf{z} \Vert^2$$

If the model blows up the magnitude of $z$, the loss is high, so it is forced to keep the magnitude
down, and whatever magnitude it does introduce is incentivized to carry only information needed to
predict the action (≈48:55). This is a **variational information bottleneck**, "a fancy term for saying
that you're adding noise and then also modifying your loss term to have this L2 penalty on your
representation" (≈48:55; the slide cites Alemi, Fischer, Dillon and Murphy, *Deep Variational
Information Bottleneck*, ICLR 2017). It is called variational because the objective is equivalent to a
KL divergence (≈49:41): the noisy representation is now a Gaussian distribution with mean $z_i$ and the
variance of $\epsilon$, and the penalty is equivalent to minimizing the KL divergence between that
distribution and the unit Gaussian. Slide 21 writes it as $D_{KL}\big(F(z \mid \mu_i) \Vert \mathcal{N}(0, 1)\big)$,
where $F$ is the encoder's distribution over $z$ given $\mu_i$; since $z$ is a vector, Finn says the
unit Gaussian's $1$ "should be an $I$", the identity, so the target is $\mathcal{N}(0, I)$ (≈55:52; see
the next section).

Students' questions fill in why (≈49:41–52:46):

- **Why make a representation of a one-hot vector?** After exploring a new task, the agent has to
  identify it; a new task may not fit cleanly into any previous one-hot vector, so a more continuous,
  compact space, a latent task representation, generalizes to new tasks (≈49:41–50:27).
- **Don't we need information about how tasks relate?** A one-hot vector does not seem to carry much
  information, but it is actually less compressed than a continuous representation. In Finn's drawn
  example, four tasks are four places to go, and tasks 1 and 4 are very similar; one-hot vectors
  represent them completely separately, while a learned compressed representation puts them much closer
  together (≈50:27–51:59).
- **Does $\epsilon$ go into the loss?** No. The noise is added when training the network and resampled
  every time; its distribution is what comes into the loss, which comes out to the penalty above
  (≈51:59–52:46).

## Training the execution policy

Putting step 1 together (slide 22, ≈52:46–55:52): the task identifier $\mu_i$ goes through a network to
$z_i$, noise is added to get $z_i + \epsilon$, and that and the state go into the execution policy, which
produces an action. There are two parts with parameters: the execution policy, which takes the noisy
$z_i$ and the state, and an encoder, written as a distribution because the noise can be folded into it,
a Gaussian centred on the encoder's output with some variance (≈53:32–54:19; the spoken labels for the
noisy and mean vectors are hard to follow in the transcript, so this page uses the slide's notation).
Slide 22: "Train $\pi^{\mathrm{exec}}(\mathbf{a} \mid \mathbf{s}, z_i)$ and encoder $F(z_i \mid \mu_i)$ to"

$$\max \sum_{i} \mathbb{E}_ {\pi^{\mathrm{exec}}}[r_i] - D_{\mathrm{KL}}\big(F(z_i \mid \mu_i) \Vert \mathcal{N}(0, I)\big)$$

Both networks are trained to maximize the sum over tasks $i$ of the expected reward $r_i$ under the
execution policy, so that the representation is one from which the task can be solved, while
maximizing the negative KL divergence to the unit Gaussian, the regularization on $z$ (≈54:19–55:52).
**Slides 22 and 23 print the unit Gaussian as $\mathcal{N}(0, 1)$, as slide 21's KL line does;** Finn
says that since $z$ is a vector, "this should be an I" (≈55:52), the identity covariance written above.
The objective trains the task and learns a latent task representation from the task ID at the same
time.

![Slide 22 — Solution #3: Decouple by acquiring representation of task relevant information](../raw/images/14-exploration/slide-22.jpg)

*Slide 22 — the two training objectives under the diagram: the execution policy and encoder maximize reward minus a KL bottleneck, and the exploration policy must collect data predictive of z. The slide prints the unit Gaussian as N(0, 1); Finn corrects it to the identity. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Questions on this step (≈55:52–59:42):

- **Why the bottleneck?** Without the KL term, $z$ may still hold a lot of irrelevant information: if two
  tasks have the ingredients in exactly the same place but the wall colour or the positions of other
  objects differ, $z$ will include those irrelevant things, and the exploration policy will then explore
  things irrelevant to the task (≈55:52–56:38).
- **The task ID is given in training; what about test time?** It is not needed at test time. Faced with
  a new task, the exploration policy explores, the agent predicts $z_i$, the latent task representation,
  instead of $\mu_i$, and the execution policy uses that to exploit (≈56:38–57:24).
- **Could the execution policy learn to attend to the relevant parts?** With a task representation that
  already has no irrelevant parts, the bottleneck can be skipped. Language strings such as "turn left",
  "turn right" and "move forward", where two strings for the same task are basically the same, already
  tell you how tasks relate; so would the coordinates of where you are trying to go. Those
  representations are already compressed. With a one-hot or uninformative task ID, the irrelevant
  information has to be compressed out, and the bottleneck is "one of the better mathematical frameworks"
  for encouraging that; in principle, changing how the policy makes decisions based on $\mu_i$ might work
  too (≈57:24–59:42).
- **Why output a distribution?** The stochastic network is what allows compression: adding noise
  removes information. Passing a deterministic continuous vector, there are heuristic ways to compress,
  such as reducing the dimensionality, but even a one-dimensional continuous variable can carry infinite
  information; this approach reduces information even in theory (≈58:56–59:42).

In practice, slide 22 notes, "(1) and (2) can be trained simultaneously"; Finn says the same of the two
policies (≈1:00:32).

## Rewarding the exploration policy: DREAM

Step 2 trains the exploration policy so that the data it collects is predictive of $z_i$ (slides 22–23,
≈59:42–1:04:26). Slide 23 asks: "How to formulate the *reward function* for $\pi^{\exp}$?" First,
"(a) Train model $q(z_i \mid \mathscr{D}_ {\mathrm{tr}})$" that predicts $z_i$ from the data collected so far (Finn picks
the notation on the spot, "I'll just pick some notation", ≈1:00:32).

One option is to collect a trajectory and, at its end, reward the policy with the negative prediction
error for $z$, and zero everywhere else. That gives a very sparse reward, only at the end of the
trajectory (≈1:01:17). Better is to measure, at each time step, how much information was gained about
$z$, which gives a dense reward. The negative prediction error at time step $t$ alone would be fairly
reasonable, but a single step contributes only one step's worth of information; to quantify that,
compare the prediction error at the previous step with the error at the current one (≈1:02:05). Slide 23:
"(b) $r_t$ = per-step information gain",

$$r_t = \text{prediction error from } \tau_{1:t-1} - \text{prediction error from } \tau_{1:t}$$

where $\tau_{1:t}$ is the trajectory up to time $t$ (the slide prints the minus as an em dash). The
reward is how much better the model predicts $z_i$ at the current step than at the previous one
(≈1:02:51). Finn writes the prediction error as something like $\hat{z}$ minus the true $z_i$, where
$\hat{z}$ at time $t$ is the estimate of $z$ from everything in the trajectory up to time $t$
(≈1:02:51–1:03:38). (The transcript's form of the reward's arguments at ≈1:02:05 is unclear.)

![Slide 24 — Solution #3: Decouple by acquiring representation of task relevant information](../raw/images/14-exploration/slide-24.png)

*Slide 24 — the informal theory: on a bandit-like problem where arm 0 reveals the rewarding arm, end-to-end RL² needs on the order of A² log A samples and DREAM A log A; DREAM's curve stays flat near zero. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Like many research algorithms, this has "a somewhat mangled forced acronym" (≈1:04:26): slide 23's
"**D**ecoupled **R**eward-free **E**xplor**A**tion and Execution in **M**eta-Reinforcement Learning
(DREAM)". **Students implement DREAM in Homework 4** (≈1:04:26). The paper is Liu, Raghunathan, Liang
and Finn, *Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices*,
ICML 2021, the source the site lists for this lecture.

## Theory and experiments

Recall that end-to-end meta-RL, optimizing for exploration and execution together, gets optimal
exploration but is very, very inefficient because of the coupling problem (≈1:04:26). Slide 24's
"(Informal) Theoretical Analysis" makes two points:

1. "DREAM objective is *consistent* with end-to-end optimization" "[under mild assumptions]", so it "can
   in principle recover the optimal exploration strategy" (≈1:04:26–1:05:12).
2. On efficiency, where end-to-end optimization struggled, "Consider a bandit-like setting with
   $|\mathcal{A}|$ arms": "In MDP $i$, arm $i$ yields reward. In all MDPs, arm 0 reveals the rewarding arm."
   Arm 0, instead of being a typical action, tells you the optimal action, so taking it and then the
   optimal action is optimal exploration followed by exploitation (≈1:05:12). End-to-end optimization
   ($\text{RL}^2$, the black-box method) needs $\Omega(|\mathcal{A}|^2 \log |\mathcal{A}|)$ samples for
   meta-optimization, where $|\mathcal{A}|$ is the number of arms or actions, while DREAM needs
   $\mathcal{O}(|\mathcal{A}| \log |\mathcal{A}|)$, "[assuming Q-learning with uniform outer-loop
   exploration]" (≈1:05:58).

Slide 24's chart, "Sample Complexity", plots samples until optimality (in thousands, 0 to 300) against
the number of actions (up to about 128). $\text{RL}^2$, the solid orange curve, rises steeply to about 285
thousand samples at the right end, close above its dashed $\Omega(|\mathcal{A}|^2 \log |\mathcal{A}|)$ curve;
DREAM, solid blue, stays flat near zero, below its dashed $\mathcal{O}(|\mathcal{A}| \log |\mathcal{A}|)$
curve, which reaches about 60. As the problem gets more complicated, DREAM does far, far better than
end-to-end optimization (≈1:05:58–1:06:44).

![Slide 25 — Empirical Comparison: Sparse Reward 3D Visual Navigation Problem](../raw/images/14-exploration/slide-25.jpg)

*Slide 25 — the 3D navigation task: the sign on the far wall names the colour, and the agent must walk around the barrier to read it before going to the key, block or ball of that colour. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Empirically (slide 25, ≈1:06:44): in a meta-RL problem, different tasks correspond to going to objects of
different colours, and the colour is written on a sign on the wall. "Task: go to the (key / block /
ball), color specified by the sign"; the agent starts on the other side of a barrier and must walk
around it to read the sign, then go to the correct object, seeking information and then exploiting, from
"Pixels observations (80 x 60 RGB)" with a "Sparse binary reward".

![Slide 26 — Quantitative Comparison](../raw/images/14-exploration/slide-26.jpg)

*Slide 26 — average returns on 3D navigation: DREAM (blue) climbs towards the optimal 1.0 while the end-to-end methods RL², IMPORT and VariBAD stay at zero, and the PEARL upper bound (dashed pink) sits near 0.17. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Slide 26's learning curves plot average returns against timesteps (in thousands, to 3,500). The
end-to-end algorithms, $\text{RL}^2$, IMPORT and VariBAD, stay at about zero throughout: they really, really
struggle with the end-to-end problem of learning to explore, "perform poorly due to **coupling**"
(≈1:06:44–1:07:29). PEARL-UB, a dashed line at about 0.17, is an "Upper-bound on PEARL: optimal policy
and Thompson-Sampling exploration, does not learn the optimal exploration strategy"; one of Lecture 13's
alternative strategies can do better in theory, but its exploration is still suboptimal. DREAM rises
from about zero to about 0.75–0.8 by 2,500–3,500 thousand steps, against an optimal return of 1.0:
"DREAM achieves near-optimal reward", learning to go find the sign and then go to the correct object
(≈1:07:29).

![Slide 27 — Qualitative Results for DREAM](../raw/images/14-exploration/slide-27.jpg)

*Slide 27 — frames of DREAM's exploration episode (top), in which it walks around to read the sign, and its execution episode (bottom), in which it goes to the key the sign named. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Slide 27 holds frames of a video (≈1:07:29–1:08:16): in the exploration episode the agent walks around to
read the sign on the wall in this simple environment, and in the execution episode, with the goal "Go to
key", it goes to the blue key.

*[Student question, inaudible]* Here exploration and exploitation separate cleanly: first explore, then
exploit. What about tasks that interleave them, finding some information, using it to accomplish a
subtask, then exploring again? "We have done some work that tries to combine them", Finn says: for
example, a single policy can use the reward for identifying the task combined with the task reward, and
Finn offers to point interested students to that work (≈1:08:16).

## Three ways to learn to explore

Slide 28, "How Do We Learn to Explore in Meta-RL?", summarizes (≈1:08:16–1:09:49):

- **End-to-end**: "+ leads to optimal strategy in principle", but "-- challenging optimization when
  exploration is hard"; in the navigation experiment the end-to-end methods flatlined, learning no good
  exploration.
- **Alternative strategies**: "+ easy to optimize" and "+ many based on principled strategies", but "--
  suboptimal by arbitrarily large amount in some environments", and very inefficient at exploration.
- **Decoupled exploration and execution**: "+ leads to optimal strategy in principle" and "+ easy to
  optimize in practice", the best of both worlds; but "-- requires task identifier". You need at least to
  be able to enumerate your tasks and know which one you are in during training, and that is the key
  thing it uses to get its benefits.

## Application: exploring student programs

With the remaining time, Finn applies DREAM to a different domain: finding bugs in computer programs,
specifically students' programs in intro computer science courses (slides 29–30, ≈1:09:49). Writing a
program that makes a game is very rewarding for students learning to code. On Code.org, students code up
a game called Bounce, which looks a bit like Pong, or one side of Pong, where the goal is to hit a ball
into the goal; in Stanford's CS106A, students code up Breakout (≈1:09:49–1:10:35). Slide 30's Breakout
panel is a frame of a video of a buggy student game, in which the ball gets "skewered" by the paddle,
bouncing off it rather than losing a point or a life (≈1:10:35).

![Slide 30 — Example application: finding bugs & providing feedback in student programs](../raw/images/14-exploration/slide-30.png)

*Slide 30 — two student assignments treated as tasks to explore: Code.org's Bounce (its readout at timestep 0) and CS106A's Breakout, shown in the lecture as a video of a buggy game. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

Each student program is a different task: the goal is to explore and find the bugs that might exist,
and ultimately to give the student feedback on them (≈1:10:35). It is also "Time-consuming for
instructors/TAs to give feedback, grades", since they often end up playing the game to find the bugs.
So: "*Use meta-RL to learn exploration!*" Exactly the DREAM algorithm can be trained to identify the
distinguishing factors of different students' programs (≈1:11:20).

![Slide 31 — Experiments: Learned Exploration Behavior on Bounce](../raw/images/14-exploration/slide-31.jpg)

*Slide 31 — the first frames of three videos of the learned exploration policy on Bounce, probing what happens when the ball hits the goal, the floor and the wall. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

The learned exploration strategies probe "What happens when…" (slide 31, from Liu, Stephan, Nie, Piech,
Brunskill and Finn, NeurIPS 2022, ≈1:11:20). Its videos show the policy trying to play the game
successfully, to see what happens when the ball hits the goal; purposely not hitting the ball, to see
what happens when it hits the floor; and hitting the wall instead of the goal, on a buggy program that
creates lots of balls as soon as the ball hits the wall.

![Slide 32 — Experiments: AI-Assisted Grading in CS106A (Spring 2023)](../raw/images/14-exploration/slide-32.jpg)

*Slide 32 — the grading interface, in which the autograder pre-fills the rubric and shows videos, beside the timing and accuracy results and the TAs' survey scores. [Deck](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf)*

In CS106A in the spring 2023 quarter, TAs used the meta-RL algorithm and its policy to pre-populate the
grading rubric, then verified the rubric answers by watching videos of the agent playing each student's
game (slide 32, ≈1:11:20–1:12:08). The slide's headline is "Leads to 44% faster & 6% more accurate
grading." Its table:

| Grading scheme | Human grading time | Grading accuracy |
| --- | --- | --- |
| Manual | 8 min 35s ± 6 min 47s | 86.4% ± 8.9% |
| Autograder with human | 4 min 49s ± 2 min 5s | 92.3% ± 7.6% |
| Autograder only | — | 90.1% ± 11.0% |

"Stanford TAs like using it": on a five-point Likert scale they rated the autograder as easier (4.5) and
faster (4.5) than grading manually, more accurate (3.9), useful (4.4) and enjoyable (4.6), and gave 9.0
out of 10 for recommending it over manual grading. They also used it in at least one subsequent quarter,
because they liked it so much (≈1:12:08). The slide cites Liu, Yuan, Ahmed, Cornwall, Woodrow, Burns,
Nie, Brunskill, Piech and Finn, *A Fast and Accurate Machine Learning Autograder for the Breakout
Assignment*, SIGCSE 2024. The recording ends here.

## Not in the recording: the plan and reminders

Slide 33 repeats the plan: "1. Exploration" (why it is hard, algorithms for exploration in bandits,
exploration in robotics and LLMs) and "2. Learning to Explore via Meta-Learning" (efficiently
meta-learning optimal exploration, "Part of HW 4", and meta-exploration for CS education; the slide
prints "Meta-Exploration in for CS Education"). Slide 34's reminders, the same as slide 2's: "Homework 3
due **tonight**", "(and HW4 out today)" and "Project milestone due **next Friday**". It previews next
week: "Can we make reinforcement learning more autonomous?" and "Can we do extra long-horizon tasks
leveraging hierarchy?" See [course logistics](course-logistics.md).

## Summary

Exploration means finding high-reward behaviour that individually unrewarding actions lead to, and
choosing between trying something new and doing the best thing known; the two are the same problem, the
balance between exploration and exploitation. In multi-armed bandits, one-step stateless problems with
stochastic rewards, exploration can be analysed: regret, the shortfall of the rewards obtained against T
times the best action's expected reward, grows linearly for an algorithm that does not learn and
sublinearly for one that explores and then exploits. Optimism under uncertainty (UCB) adds a bonus for
uncertain actions; probability matching (posterior sampling) samples a model of the bandit from its
beliefs and acts optimally for it; in the class's drug-trial game, posterior sampling did best, UCB beat
epsilon-greedy in the long run and greedy did worst. In large MDPs, exploration from scratch is
intractable, so demonstrations, pretrained base models and shaped rewards supply it. For exploration at
test time in meta-RL, end-to-end training suffers from the coupling of exploration and execution, and
alternative strategies can be badly suboptimal. DREAM decouples the two: the execution policy and an
encoder are trained with the task identifier to maximize reward under a variational information
bottleneck that keeps only task-relevant information in a representation z, and the exploration policy
is rewarded for per-step information gain about z. It is consistent with end-to-end optimization, needs
far fewer samples on a bandit-like problem, reaches near-optimal reward where end-to-end methods fail,
and requires a task identifier during training. Applied to student programs, it learned to probe games
for bugs and made grading in CS106A 44% faster and 6% more accurate.

## Related pages

- [Exploration](exploration.md): the exploration problem, regret, bandit strategies and learning to
  explore, as a topic.
- [Meta-RL](meta-rl.md): the meta-RL problem and the exploration strategies this lecture compares,
  including DREAM.
- [Lecture 13 — Meta RL](13-meta-rl.md): the previous lecture, which introduces the coupling problem and
  the alternative exploration strategies.
- [Multi-task RL](multi-task-rl.md): the ID-conditioned execution policy is multi-task RL.
- [Q-learning](q-learning.md): epsilon-greedy and Boltzmann exploration in Q-learning.
- [Imitation learning](imitation-learning.md) and [RLHF](rlhf.md): demonstrations and preference rewards,
  two of the starting points that make exploration feasible in large MDPs.
- [RL for LLM reasoning](llm-reasoning.md): RL from a strong base model, and exploring strategies when
  reasoning.
- [Course logistics](course-logistics.md): Homework 3's due date, Homework 4 and the project milestone.
