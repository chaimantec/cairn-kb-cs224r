---
title: "Lecture 11 — Model-Based RL (slide deck)"
lecture: 11
slides: 38
printed_numbers: 1–38
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf
numbering: 1:1 with PDF pages, printed bottom right; page 3 prints no number, label inferred from position, unambiguous
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (4, 12, 27, 34, 35, 36) were checked by Claude Opus 5.5 against 1400px renders, with slide 32's author line checked on a crop. Slides 4, 27 and 36 (every value of both series) agree, and slide 34's two charts agree series by series. Corrections were small: slide 12's backward (magenta) arrows are on every link but the first, from the first state to the next; slide 35's CEM curve rises from about 0.4 to about 0.55 rather than staying flat, and its random-shooting curve swings between about 0.3 and 0.9; slide 34's PETS curve runs behind the legend, so where it ends cannot be seen.
images: 17 slides are rendered and shown under their headings (4, 6, 7, 8, 12, 13, 17, 18, 19, 20, 21, 22, 27, 33, 34, 35, 36). Not rendered: slide 5, an agenda whose video frame shows the scene of slide 36's; build steps whose content a later or earlier rendered slide shows (14 repeats slide 13's plots under text transcribed here, 15 is slide 17 without its last line, 26 is slide 27 without the augmented roll-outs); text, equation and algorithm slides transcribed here in full (10, 16, 23, 25, 28, 29, 30); slide 32, a paper title block transcribed here; and the title, divider, agenda and reminder slides (1, 2, 3, 9, 11, 24, 31, 37, 38).
---

# Lecture 11 — Model-Based RL: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 11 deck, "Model-Based Reinforcement Learning", transcribed from the PDF. The title slide (slide 1) names the course, "CS 224R", and no lecturer. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

The recording (about 73 minutes) opens on the recap (slide 4) without discussing slide 2's reminders, and follows the deck through slide 23 and the first half of slide 25 (distilling the planner into a policy, ≈1:02:51, and planning with a terminal value function, ≈1:03:36–1:06:40). At ≈1:06:40 Finn says "I think I might cover this second approach on Friday" and skips to the case study: slides 26–30 (model-based policy optimization's augmented roll-outs and full algorithm, when to use model-based RL, other kinds of models) are not discussed, so for them this file is the only source. Lecture 12's deck opens with its own version of the model-based policy optimization slides. The case study (slides 32–36) runs from ≈1:06:40 to the end at about 1:13:20, and slides 37–38 are not discussed beyond the closing "next time" (≈1:12:52). Finn works three things on a board the recording does not show: a latent-state model with an encoder (≈15:23–17:49), the gradient-based planning algorithm with its chain rule and update (≈23:17–31:07), and sketches of gradient-based and sampling-based optimization over actions (≈31:54–35:44); the open-loop and closed-loop execution of a plan is also described at the board (≈51:13–52:52). The transcript is the only source for those.

## Slide numbers vs PDF pages

The deck is 1:1 with its printed numbers: PDF page N is printed slide N, for all 38 pages, and the number prints in the bottom-right corner (bottom centre on some pages). Page 3 prints no number; its label follows the 1:1 sequence and is unambiguous.

| PDF page | Printed slide |
| -------- | ------------- |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 (inferred from position) |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 10 | 10 |
| 11 | 11 |
| 12 | 12 |
| 13 | 13 |
| 14 | 14 |
| 15 | 15 |
| 16 | 16 |
| 17 | 17 |
| 18 | 18 |
| 19 | 19 |
| 20 | 20 |
| 21 | 21 |
| 22 | 22 |
| 23 | 23 |
| 24 | 24 |
| 25 | 25 |
| 26 | 26 |
| 27 | 27 |
| 28 | 28 |
| 29 | 29 |
| 30 | 30 |
| 31 | 31 |
| 32 | 32 |
| 33 | 33 |
| 34 | 34 |
| 35 | 35 |
| 36 | 36 |
| 37 | 37 |
| 38 | 38 |

## Sections

| Section | Slides |
| ------- | ------ |
| Title, course reminders and the high-level algorithm recap (offline versus online) | 1–4 |
| Plan; model-based RL key idea: can we learn a "simulator", and the notional learned-simulator algorithm | 5–8 |
| How to learn a dynamics model (in brief) | 9–10 |
| How to use a learned model, part i: planning (gradient-based and sampling-based optimization, CEM, data-distribution mismatch, open-loop versus closed-loop, MPC) | 11–23 |
| How to use a learned model, part ii: data generation (model-based policy optimization, when to use model-based RL, other kinds of models) | 24–30 |
| Case study in dexterous robotic manipulation (Deep Dynamics Models / PDDM, simulated and real results) | 31–36 |
| Closing plan, course reminders and next time | 37–38 |

## Slide 1 — Model-Based Reinforcement Learning

Title slide. "Model-Based Reinforcement Learning" with "CS 224R" below it. No lecturer or affiliation is printed.

## Slide 2 — Course reminders

- Homework 3 due next Friday, May 16
- Project milestone due Friday, May 23

## Slide 3 — High-Level Algorithm Recap

Section divider: the title alone on a white page. (No number prints on this page.)

## Slide 4 — Offline vs. online algorithms (recap diagram)

![Slide 4 — Offline vs. online algorithms (recap diagram)](../images/11-model-based-rl/slide-4.png)

Untitled recap diagram, a taxonomy of the algorithms covered so far, split by a vertical dotted line into Offline (left) and Online (right).

- Left column header "Offline" (magenta), sub-label "No policy collection". Right header "Online" (teal), sub-label "Involves policy data collection".
- A grey band across the top: "Offline imitation learning" with "Behavior cloning" under it (left of the dotted line); "Online imitation learning" with "DAgger" (right of it); at far right in grey text "Requires expert data. Doesn't need reward." The band covers both columns.
- Below the band, three columns. Offline RL (left, grey): "Constrain to actions in data", "AWR, AWAC, IQL"; then "Implicitly constrain via conservatism", "CQL".
- Off-policy RL (middle): "Can reuse data from other policies"; two sub-columns "Replay buffer" with "DQN, SAC" (DQN in green, SAC in teal) and "Mult. grad steps" with "PPO, Imp. Sampling" (PPO in teal, Imp. Sampling in blue).
- On-policy RL (right, separated from off-policy by a second vertical dotted line): "Only use data from curr. policy", with "REINFORCE / vanilla PG" in blue.
- Along the bottom, three coloured labels under the online columns: "Q-learning (critic only)" in green, "actor-critic (both)" in teal, "policy gradient (actor only)" in blue.
- A horizontal arrow under them pointing right, labelled "Requires more online data".

## Slide 5 — The plan for today

Agenda, numbered list:

1. Model-based reinforcement learning
   - a. Key idea
   - b. How to learn a dynamics model? (in brief)
   - c. How to use a learned dynamics model?
     - i. Planning
     - ii. Data generation
2. Case study in dexterous robotic manipulation

"Key learning goals:" then two dashes: "How to learn and use dynamics models"; "The key challenges and trade-offs arising in model-based RL" (the words "key challenges" in red).

At upper right, a grey caption bar "How to get a robot to do this!" above a video frame (this is a video frame, a single still): a dark scene in which a multi-fingered robot hand with teal fingertips holds a yellow ball and a pink ball in its palm, with a cardboard piece at the left and a marker tag at the top.

In the recording (≈5:22) Finn plays the video as a teaser for the case study and says the hand has more than 20 degrees of freedom, more than 20 different joints that it is controlling.

## Slide 6 — Can we learn a "simulator"?

![Slide 6 — Can we learn a "simulator"?](../images/11-model-based-rl/slide-6.jpg)

Subtitle: "i.e. predictive model of $\mathbf{s}_ {t+1}$ given $\mathbf{s}_ t$, $\mathbf{a}_ t$".

Left side, three domains:

- "Robotics & physical systems": indented "modeling physics of (directly observed) physical system" and "video prediction, conditioned on actions".
- "Finance: stock market predictor".
- "Games: rules of the game", with a grey arrow pointing up at it from the grey text "May not need to learn a model! (e.g. chess)". Beside it: "May need to model other players — Sometimes considered part of dynamics, sometimes modeled separately (multi-agent RL)".

Right side, two generated-video stills: top, captioned "Veo 2", the rear of a grey-and-red city bus (route sign "42") at a stop with pedestrians; bottom, captioned "Sora", an aerial view of a large columned building and park in a city.

In the recording (≈6:56) these are videos, which Finn introduces as videos generated by large models like Veo 2 and Sora.

## Slide 7 — Notional "learned simulator" algorithm

![Slide 7 — Notional "learned simulator" algorithm](../images/11-model-based-rl/slide-7.jpg)

Left, a numbered algorithm:

1. Get some data $\mathcal{D} = \lbrace \mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i \rbrace$, "collected using some base policy $\pi_0(\mathbf{a} \mid \mathbf{s})$".
2. Learn "simulator" $f(\mathbf{s}, \mathbf{a}) = \mathbf{s}'$, with the objective $\min_ f \sum_ i \Vert f(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$.
3. "Run your favorite RL method inside "simulator" $f$" with a bullet "Using one of the RL methods we covered, or a planning method".

At the bottom in teal: "Question: What might go wrong?"

Right, two figures. Top: a black hand-drawn tangle of looping trajectory curves labelled $\mathcal{D}$ at upper left, with arrowheads at several loose ends (upper left, top, right, and one inside the tangle), a green dot at the lower left of the tangle and an orange dot at the right, outside the tangle. Bottom: a fully connected neural network diagram with four layers; input labelled $\mathbf{s}$ and $\mathbf{a}$ at the left, output labelled $\mathbf{s}'$ at the right.

## Slide 8 — Notional "learned simulator" algorithm

![Slide 8 — Notional "learned simulator" algorithm](../images/11-model-based-rl/slide-8.jpg)

Same title and the same three-step algorithm as slide 7, now in coloured boxes: step 1 in a blue box, step 2 in a green box, step 3 in a pink box. New relative to slide 7: annotations.

- Next to step 1, blue text: "Data coverage matters a lot!", with a maze image at upper right: top-down view of an orange-walled maze on a blue checkerboard floor, with the label $\mathcal{D}$ in yellow at its upper left and a yellow hand-drawn loop with an arrowhead covering only the upper-left corner of the maze (the data cover a small part of it).
- Next to step 2, green text: "Modeling is **domain dependent**. Some domains much easier to simulate than others!", with a generated-image still at lower right: a metallic humanoid robot arm beside a wooden table with a small green cube on it. Caption under the image: "A humanoid robot standing near the table with red, green and blue cubes on it performs a cube stacking task, with red in the bottom and blue on the top."

The image is one frame of a video. In the recording (≈11:33) Finn plays it as a video generated by Sora from that prompt, and says the generation "does some creative things with physics": it changes the colours of the blocks and synthesizes blocks out of thin air, which is not what robots can do in reality.
- Under step 3, pink text in two columns: "Even if we have a good simulator, this part isn't necessarily easy!" and "Need to account for model inaccuracies."

## Slide 9 — The plan for today

Same agenda as slide 5 (without the robot image or learning goals). New: item b, "How to learn a dynamics model? (in brief)", is in bold.

## Slide 10 — How to learn a good dynamics model? (in brief)

Three coloured dash items:

- "You know it already" (green) "(e.g. some games)"
- "You approximately know most of it" (teal) "(e.g. certain physical models)"; sub-bullet "Can fit the unknown parameters using the data"
- "You don't know it" (blue) "(almost all scenarios in practice)"; sub-bullets "Learn it end-to-end" and "Learn a (low-dim) state representation, then learn model over representations"

At the bottom: "Note: Often also need to learn a reward model!"

## Slide 11 — The plan for today

Same agenda as slide 9. New: item c, "How to use a learned dynamics model?", and its sub-item i, "Planning", are in bold.

## Slide 12 — Approach 1: Optimize over actions using model

![Slide 12 — Approach 1: Optimize over actions using model](../images/11-model-based-rl/slide-12.png)

Title line: "Approach 1: Optimize over actions using model", followed by the displayed objective $\max_ {\mathbf{a}_ {t:t+H}} \sum_ t r(\mathbf{s}_ t, \mathbf{a}_ t)$, with the blue label "planning" below it.

Diagram (left): an unrolled chain $\mathbf{s}_ t \rightarrow \mathbf{s}_ {t+1} \rightarrow \mathbf{s}_ {t+2} \rightarrow \cdots \rightarrow \mathbf{s}_ H$. Actions $\mathbf{a}_ t$, $\mathbf{a}_ {t+1}$, ..., $\mathbf{a}_ {H-1}$ sit below and each has a black arrow up-right into the next state. Rewards $r_ {t+1}$, $r_ {t+2}$, ..., $r_ H$ sit above the states $\mathbf{s}_ {t+1}$, $\mathbf{s}_ {t+2}$, ..., $\mathbf{s}_ H$ with a black arrow from the state up to the reward. Parallel magenta arrows run in the opposite direction on every link except the first ($\mathbf{s}_ t$ to $\mathbf{s}_ {t+1}$, which has only its black arrow): reward down to state, state back to previous state, and state back down to action, showing the backward gradient flow.

At right, in blue: "Approach 1a: via *backpropagation* (i.e. *gradient-based* optimization)".

"Algorithm:" with three steps:

1. Run some policy (e.g. random policy) to collect data $\mathcal{D} = \lbrace (\mathbf{s}, \mathbf{a}, \mathbf{s}')_ i \rbrace$
2. Learn model $f_ \phi(\mathbf{s}, \mathbf{a})$ to minimize $\sum_ i \Vert f_ \phi(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$
3. Backpropagate through $f_ \phi(\mathbf{s}, \mathbf{a})$ to choose actions

## Slide 13 — Aside: Gradient-based vs. sampling-based optimization

![Slide 13 — Aside: Gradient-based vs. sampling-based optimization](../images/11-model-based-rl/slide-13.png)

Two side-by-side plots, each with axes "loss $L(\theta)$" (vertical) and "parameters $\theta$" (horizontal) and a blue U-shaped curve (decreasing then rising on the right).

- Left, headed "Gradient-based (1st order)": a short red dashed tangent segment drawn on the right-hand rising part of the curve, just right of the minimum.
- Right, headed "Sampling-based (0th order)": seven red dots on the same curve, two at the far left, two just left of the minimum (one near it), one at the minimum and three on the right rising part.

Below, "Cross-entropy method (CEM)" with the note "(Not to be confused with the cross-entropy loss!)". Three numbered steps with a curved arrow at the left labelled "repeat" from step 3 back up to step 1:

1. Sample from distribution $p_ i(\theta)$
2. Rank samples according to loss $\theta_ 1, \ldots, \theta_ K$
3. Fit Gaussian distribution $p_ {i+1}$ to "elite" samples $\theta_ 1, \ldots, \theta_ k$

At the right of step 3: "Eventually return $\theta_ 1$".

## Slide 14 — Aside: Gradient-based vs. sampling-based optimization

Same two plots as slide 13 (gradient-based first-order at left with the red tangent segment; sampling-based zeroth-order at right with seven red dots). New relative to slide 13: the CEM steps are gone and pros and cons are added under each plot.

Left: "+ scalable to high dimensions"; "+ works well *especially* in overparametrized regimes" (the asterisks print literally around "especially"); "- requires nice optimization landscape". Pros are green, the con red.

Right: "+ parallelizable"; "+ requires no gradient information"; "- scales poorly to high dimensions".

## Slide 15 — Approach 1: Optimize over actions using model

Same title line and objective as slide 12, $\max_ {\mathbf{a}_ {t:t+H}} \sum_ t r(\mathbf{s}_ t, \mathbf{a}_ t)$, with the blue label "planning". The chain diagram is the same as on slide 12 but has only the black forward arrows (no magenta backward arrows).

At right, in magenta: "Approach 1b: via *sampling* (i.e. *gradient-free* optimization)".

Algorithm: steps 1 and 2 are as on slide 12 (collect data $\mathcal{D} = \lbrace (\mathbf{s}, \mathbf{a}, \mathbf{s}')_ i \rbrace$ with some policy, e.g. a random policy; learn $f_ \phi(\mathbf{s}, \mathbf{a})$ to minimize $\sum_ i \Vert f_ \phi(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$). Step 3 is now "Iteratively sample action sequences, run through model $f_ \phi(\mathbf{s}, \mathbf{a})$ to choose actions".

## Slide 16 — Sampling-Based Optimization

"Denote $\mathbf{A} := \mathbf{a}_ t, \ldots, \mathbf{a}_ {t+H}$".

Version 1 (blue): "Guess & check", with the blue label "random shooting" at right.

a. Sample many $\mathbf{A}_ 1, \ldots, \mathbf{A}_ N$ from some distribution (e.g. uniform). The words "some distribution" are underlined in red.

b. Choose $\mathbf{A}_ i$ based on $\arg\max_ i \sum_ {t'=t}^{t+H} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$.

At right of step b, in red-orange: "Can we improve this distribution?"

Version 2 (red-orange): "Cross-entropy method".

a. Sample many $\mathbf{A}_ 1, \ldots, \mathbf{A}_ N$ from $p(\mathbf{A})$

b. Evaluate $J(\mathbf{A}_ i) = \sum_ {t'=t}^{t+H} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$

c. Pick the elites $\mathbf{A}_ {i_ 1}, \ldots, \mathbf{A}_ {i_ M}$ with the largest $J(\mathbf{A}_ i)$, where $M \lt N$

d. Refit $p(\mathbf{A})$ to the elites $\mathbf{A}_ {i_ 1}, \ldots, \mathbf{A}_ {i_ M}$

At right of version 2: "+ fast, if parallelized" and "+ simple" in green; "- doesn't scale to high dimensions" in red, then "(including both $H$ and $\vert \mathbf{a} \vert$)" in red.

## Slide 17 — Approach 1: Optimize over actions using model

![Slide 17 — Approach 1: Optimize over actions using model](../images/11-model-based-rl/slide-17.png)

Same as slide 15 (Approach 1b via sampling, i.e. gradient-free optimization; same chain diagram with forward arrows only; same three-step algorithm). New: under step 3, in magenta, "(e.g. with random shooting or cross-entropy method)".

## Slide 18 — How can this approach fail?

![Slide 18 — How can this approach fail?](../images/11-model-based-rl/slide-18.jpg)

Left: a photograph of a grassy cliff edge above a dark sea under a grey sky, with three hand-drawn overlays. A red-orange scribble of tight loops and a short arrow on the grass at the left (the region the base-policy data cover). A yellow arrow starting near that scribble, running up and right along the grass and then curving over the cliff edge and down into the sea. A green horizontal arrow across the cliff face pointing right, with the green text "We should go right" over it. Below the image, in red-orange: "Going right means that we can go higher!"

Right: the three-step algorithm from slide 17 in small type (steps 1 to 3 without the pink note). Below it, in red: "Data distribution mismatch" and $p_ {\pi_ 0}(\mathbf{s}) \neq p_ {\pi_ f}(\mathbf{s})$.

At the bottom: "**Thought Exercise:** How might you alleviate this issue?" (teal).

## Slide 19 — Approach 1: Optimize over actions using model

![Slide 19 — Approach 1: Optimize over actions using model](../images/11-model-based-rl/slide-19.png)

Same top part as slide 17 (objective, forward-arrow chain diagram, "Approach 1b: via *sampling* (i.e. *gradient-free* optimization)" now in blue rather than magenta, the "planning" label is absent). Algorithm, now four steps:

1. Run some policy (e.g. random policy) to collect data $\mathcal{D} = \lbrace (\mathbf{s}, \mathbf{a}, \mathbf{s}')_ i \rbrace$
2. Learn model $f_ \phi(\mathbf{s}, \mathbf{a})$ to minimize $\sum_ i \Vert f_ \phi(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$
3. Iteratively sample action sequences, run through model $f_ \phi(\mathbf{s}, \mathbf{a})$ to choose actions
4. Execute planned actions, appending visiting tuples $(\mathbf{s}, \mathbf{a}, \mathbf{s}')$ to $\mathcal{D}$

New: step 4, and a green curved arrow at the left from step 4 back up to step 2.

## Slide 20 — Revisiting the cliff

![Slide 20 — Revisiting the cliff](../images/11-model-based-rl/slide-20.jpg)

Left: the cliff photograph from slide 18 with the red-orange scribble and the red-orange arrow going up the grass and over the edge into the sea, and a new yellow arrow along the grass from the lower left toward the cliff top (pointing up and right, stopping before the edge). Below it, in red-orange: "Going right means that we can go higher!" and in orange: "**Final policy**: go to the top and stop."

Right: the four-step algorithm of slide 19 in small type, with no arrow.

## Slide 21 — Can we do even better?

![Slide 21 — Can we do even better?](../images/11-model-based-rl/slide-21.png)

Two diagrams side by side. Left, "Open-loop": a black silhouette of a person with hands on hips, an arrow curving to a green globe (Africa and Europe visible) labelled $\mathbf{a}_ {1:H}$. Right, "Closed-loop": the same person and globe, with an arrow from the person to the globe labelled $\mathbf{a}_ t$ and a return arrow from the globe to the person labelled $\mathbf{s}_ {t+1}$.

Below: "When might it be bad to be open-loop?" and "Ideas on how to make planning closed-loop?"

## Slide 22 — Approach 2: Plan & replan using model

![Slide 22 — Approach 2: Plan & replan using model](../images/11-model-based-rl/slide-22.jpg)

Subtitle in italics: "model-predictive control (MPC)". Five steps:

1. run base policy $\pi_0(\mathbf{a}_ t \mid \mathbf{s}_ t)$ (e.g., random policy) to collect $\mathcal{D} = \lbrace (\mathbf{s}, \mathbf{a}, \mathbf{s}')_ i \rbrace$
2. learn model $f_ \phi(\mathbf{s}, \mathbf{a})$ to minimize $\sum_ i \Vert f_ \phi(\mathbf{s}_ i, \mathbf{a}_ i) - \mathbf{s}'_ i \Vert^2$
3. use model $f_ \phi(\mathbf{s}, \mathbf{a})$ to optimize action sequence
4. execute the first planned action, observe resulting state $\mathbf{s}'$
5. append $(\mathbf{s}, \mathbf{a}, \mathbf{s}')$ to dataset $\mathcal{D}$

Two green curved arrows at the left: a long one from step 5 back to step 2 and a shorter one from step 4 back to step 3.

Below, a top-down picture of a road between two black lines with an orange dashed centre line: a white car drawn in two overlapping poses, the front one angled slightly across the line, and a small green arrow near its nose pointing down toward the line. Red slanted text: "REPLANNING HELPS WITH MODEL ERRORS". At the bottom: "+ replan to correct for model errors" (green) and "- compute intensive" (red).

## Slide 23 — So far: Planning with learned models

1. Can *plan* $\mathbf{a}_ 1, \ldots, \mathbf{a}_ H$ with gradient-based (blue) or sampling-based (magenta) optimization
2. *Update the model* using data collected with planning
3. *Replan* periodically to help account for mistakes.

Two columns of pros and cons. Pros (green): "+ Simple"; "+ Easy to plug in different goals / rewards (possibly even at test time!)". Cons (red): "- Compute intensive at test time"; "- Only practical for short-horizon problems (or very shaped reward functions)".

"Why only short horizons?" with two answers: "(a) too compute expensive to make long plans" and "(b) model is not accurate for long horizons".

At the bottom: "Can we *train a policy* using a learned model?"

## Slide 24 — The plan for today

Same agenda as slide 9. New: item ii, "Data generation", is in bold.

## Slide 25 — Model-based policy optimization

"**Option 1**: Distill planner's actions into a policy (i.e. train policy to match actions taken by planner)".

"+ no longer compute intensive at test time" (green); "- still limited to short-horizon problems" (red).

"How might we solve **longer-horizon** problems using a model?"

1. Plan with terminal value function
2. Augment model-free RL methods with data from model

At the lower right: "Let's focus on #2".

## Slide 26 — Model-based policy optimization

"**Key idea**: augment data with model-simulated roll-outs."

Left, headed "Example real trajectory": six dots joined by arrows running from lower left to upper right, labelled $\mathbf{s}_ 1$ (blue dot, the start, lower left), $\mathbf{s}_ 2$, $\mathbf{s}_ 3$, $\mathbf{s}_ 4$, $\mathbf{s}_ 5$, $\mathbf{s}_ 6$ (teal dots), with an arrow from each to the next.

Right, headed "How to augment?":

- generate full trajectories from initial states? (the words "initial states" in blue)
  - model may not be accurate for long horizons
- generate *partial trajectories* (magenta) from initial states? (blue)
  - may not get good coverage of later states

## Slide 27 — Model-based policy optimization

![Slide 27 — Model-based policy optimization](../images/11-model-based-rl/slide-27.png)

Same title and key idea as slide 26: "**Key idea**: augment data with model-simulated roll-outs."

Left, "Example real trajectory": the real trajectory $\mathbf{s}_ 1$ (blue) to $\mathbf{s}_ 6$ (teal) as on slide 26, now with magenta arrows and pink dots (labelled "Augmented data" in magenta at the bottom) added. New: from $\mathbf{s}_ 1$ a magenta arrow to a pink dot, then a second short magenta arrow to another pink dot (a two-step model rollout off the start state); from $\mathbf{s}_ 3$ a magenta arrow to a pink dot and on to a second pink dot at the top; from $\mathbf{s}_ 4$ a magenta arrow to a pink dot and on to a further pink dot at upper right. There is no augmented branch from $\mathbf{s}_ 2$, $\mathbf{s}_ 5$ or $\mathbf{s}_ 6$.

Right, "How to augment?" repeats the two bullets of slide 26 and adds a third: "generate *partial trajectories* from *all states* in the data" (the words "all states" in teal), followed by a light-bulb emoji.

## Slide 28 — Model-based policy optimization

Key idea as on slide 27. "**Full algorithm**":

1. Collect data using current policy $\pi_ \phi$, add to $D_ {env}$
2. Update model $p_ \theta(s' \mid s, a)$ using $D_ {env}$
3. Collect synthetic roll-outs using $\pi_ \phi$ in model $p_ \theta$ from states in $D_ {env}$; add to $D_ {model}$
4. Update policy $\pi$ (and critic $Q$) using $D_ {model}$

A grey curved arrow at the left runs from step 4 back up to step 1.

"Notes:" with two dashes: "compatible with variety of model-free RL methods (step 4)"; "could additionally use $D_ {env}$ in policy update".

(The slide prints these symbols in non-bold italic, $s$, $a$, $\pi$, rather than the bold $\mathbf{s}$, $\mathbf{a}$ used elsewhere in the deck; the slide's own notation is kept above.)

## Slide 29 — When to use model-based RL?

Subtitle: "Big upsides and big downsides".

Green pros:

- "+ Immensely useful, far more data efficient if model is easy to learn"
- "+ Model can be trained on data without reward labels (fully self-supervised)"
- "+ Model is somewhat task-agnostic (can sometimes be transferred across rewards)"

Red cons:

- "- Models don't optimize for task performance"
- "- Sometimes harder to learn than a policy"
- "- Another thing to train, more hyperparameters, more compute intensive"

Closing line: "Whether to use a model depends on how hard it is to learn!"

## Slide 30 — Other kinds of models

"So far: modeling $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$".

"Many alternatives", a list:

- inverse model $p(\mathbf{a}_ t \mid \mathbf{s}_ t, \mathbf{s}_ {t+1})$
- multi-step inverse model $p(\mathbf{a}_ t \mid \mathbf{s}_ t, \mathbf{s}_ {t+n})$ or $p(\mathbf{a}_ {t:t+n} \mid \mathbf{s}_ t, \mathbf{s}_ {t+n})$
- future prediction without actions $p(\mathbf{s}_ {t+1:t+n} \mid \mathbf{s}_ t)$
- video interpolation $p(\mathbf{s}_ {t+1:t+n} \mid \mathbf{s}_ t, s_ {t+n+1})$ (the slide prints the last state in italic, non-bold $s$)
- transition distribution $p(\mathbf{s}_ t, \mathbf{a}_ t, \mathbf{s}_ {t+1})$

Closing line: "These kinds of models can be useful too! Have various purposes."

## Slide 31 — The plan for today

Same agenda as slide 9. New: item 2, "Case study in dexterous robotic manipulation", is in bold.

## Slide 32 — Case study: Model-based RL for dexterous manipulation

The slide shows the title block of a paper, set in a serif face as a pasted image:

"Deep Dynamics Models for Learning Dexterous Manipulation", "Anusha Nagabandi, Kurt Konoglie, Sergey Levine, Vikash Kumar" (the name "Konoglie" as printed), "Google Brain". Below it, in the slide's own type: "September 2019".

"**Why this paper?**" then "Still one of the most impressive results with five-fingered hands!" and "Nice comparisons on what is important for good performance".

## Slide 33 — Case study: Model-based RL for dexterous manipulation

![Slide 33 — Case study: Model-based RL for dexterous manipulation](../images/11-model-based-rl/slide-33.jpg)

Left, text:

- "**State space**: hand & object positions"
- "**Action space**: controlling 5-fingered hand (24 DoF)"
- "**Reward**: track target object trajectory + penalty for dropping"
- "**Model**: Ensemble of 3 neural networks, each with 2 hidden layers of size 500"
- "**Planner**: modified version of CEM optimizer; softer reward-weighted mean & temporal smoothing on actions"
- "Alternate between collecting ~30 trajectories with planner & updating model."

Right, four stacked pictures. Top three are simulator renderings of a gold-coloured five-fingered robot hand on a blue checkerboard floor: the first holds a small die-like cube with colour-axis markers and a ghost target cube beside it; the second holds a yellow pencil upright above a white disc; the third holds two silver balls in its palm. The bottom picture is a photograph of a real black robot hand with teal fingertips holding a pink ball and a yellow ball.

## Slide 34 — Case study: Model-based RL for dexterous manipulation

![Slide 34 — Case study: Model-based RL for dexterous manipulation](../images/11-model-based-rl/slide-34.jpg)

Subtitle: "Simulated experiments". A box printed over the bottom of the slide, with a blue outline, covers the lower part of both charts, the x-axis of the left chart and the slide number: "More efficient than model-free methods" and "More performant than other model-based methods".

Left, a list of methods, each name coloured:

- "Model-free methods:" "SAC: actor-critic method" (blue); "NPG: policy gradient method" (red).
- "Model-based methods:" "PDDM: proposed method" (green); "MBPO: RL with model-generated data" (purple); "PETS: CEM-based planner" (teal); "Nagabandi et al.: random shooting, no ensembles" (orange).

Middle and right, two simulator stills above two charts. Left still: a gold robot hand holding two silver balls (Baoding balls). Right still: two views of the gold hand holding a yellow pencil over a sheet of paper (handwriting). In the recording (≈1:09:47–1:10:35) these are videos, of the hand rotating the Baoding balls and writing digits on the paper.

Left chart, "Baoding Balls" (the lower part is hidden by the box). Y axis "Task Reward", with tick labels 0.0 and -0.1 visible; x axis not visible, since it is covered. Legend entries visible: MBPO, PETS, Nagabandi et. al, SAC (further entries, if any, are covered). Lines with shaded bands, read approximately:

- PDDM (green; its legend entry is covered, identified by colour and the other chart's legend): starts near -0.06, climbs to about -0.015 and then stays flat near -0.015 to -0.02.
- MBPO (purple): noisy, oscillating between about -0.05 and -0.09 across the whole run.
- SAC (blue): noisy, oscillating around -0.07 across the whole run.
- A red line, noisy, around -0.08 across the run (its legend entry is covered; by colour it is the NPG series of the other chart's legend).
- PETS (light blue): starts about -0.11 and rises slowly to about -0.09; at the right it runs behind the legend, so where it ends cannot be seen.
- Nagabandi et. al (orange): noisy, about -0.11 to -0.12 across the run, starting lowest at the left.
- A dashed grey horizontal line near -0.015.

Right chart, "Handwriting: Fixed Trajectory". Y axis "Mean Tracking Error of Pencil Tip (mm)", ticks -5, -10, -15, -20, -25, -30 (the lower part of the axis is hidden). X axis "Number of datapoints (M)", tick labels 0.2, 0.4, 0.6, 0.8, 1.0 visible; the left end of the axis is partly hidden. Legend: PETS (light blue), Nagabandi et. al (orange), SAC (blue), NPG (red), PDDM (Ours) (green). Approximate readings:

- PDDM (Ours) (green): rises almost vertically at the far left, from about -26 to about -7.5 within the first few thousand datapoints, then reaches the top dashed line.
- SAC (blue): about -19 to -21 at the start, a rise to about -11 near 0.2, a dip to about -13, then rising to about -7.5 near 0.4, with a wide shaded band.
- NPG (red): from about -27 at the start to about -14 near 0.2, then a slow steady rise to about -7.5 at 1.0.
- PETS (light blue): from below the visible range up to about -15 near 0.2, then a dashed light-blue horizontal line at about -14.5 out to 1.0.
- Nagabandi et. al (orange): a small bump to about -31.5 near 0.15, then a dashed orange horizontal line at about -33 out to 1.0.
- A dashed grey horizontal line at about -7.5.

The numbers above are approximate reads of the picture, since the boxed text covers part of the left chart's axes.

## Slide 35 — Case study: Model-based RL for dexterous manipulation

![Slide 35 — Case study: Model-based RL for dexterous manipulation](../images/11-model-based-rl/slide-35.jpg)

Subtitle: "Simulated ablations". Four line charts in a 2 by 2 grid, each with y axis "Normalized Task Reward" (ticks 0.0, 0.5, 1.0) and x axis "Number of datapoints". Each series is a noisy line with a shaded band. Readings are approximate.

- Top left, "Model Architecture" (x ticks 0, 100000, 200000; runs to about 270000). Three series: 2x64 (orange) rises to about 0.5 by roughly 20000 datapoints and then stays flat, oscillating around 0.4 to 0.5, ending about 0.4; 2x250 (green) rises more slowly and reaches about 0.8 to 0.9 by 200000 and ends about 0.9; 2x500 (blue) rises fastest to about 0.7 by 50000 and reaches about 0.9 to 1.0 by 200000.
- Top right, "Ensemble Size" (x ticks 0, 100000, 200000; runs to about 270000). Three series: 1 (orange) is lowest, about 0.1 to 0.3 early, about 0.4 to 0.6 in the middle, rising to about 1.0 at the end with large variance; 3 (green) is about 0.5 early, about 0.3 to 0.8 in the middle, about 1.0 at the end; 5 (blue) starts highest, about 0.7 to 0.8 early, and ends about 1.0. All three overlap and are noisy.
- Bottom left, "Horizon" (x ticks 0, 50000, 100000, 150000; runs to about 190000). Four series: 2 (red) rises to about 0.6 early, then wanders about 0.3 to 0.6 and ends about 0.5; 5 (orange) rises to about 0.7 and ends about 0.75; 7 (green) rises to about 0.8 and ends about 0.75; 15 (blue) is lowest, about 0.3 to 0.4 for most of the run, ending about 0.5.
- Bottom right, "Controller" (x ticks 0, 50000, 100000; runs to about 140000). Three series: Filtering + Reward Weighting (green) rises from about 0.5 to about 0.95 and plateaus there, with a narrow band; CEM (blue) rises slowly from about 0.4 to about 0.55, with a narrow band; Random-shooting (orange) is very noisy, swinging between about 0.3 and 0.9, centred near 0.6.

At right, four dashes: "Need sufficiently large model"; "Need at least 3 ensemble members"; "Planning horizon trade-offs"; "Modified CEM is crucial".

## Slide 36 — Case study: Model-based RL for dexterous manipulation

![Slide 36 — Case study: Model-based RL for dexterous manipulation](../images/11-model-based-rl/slide-36.jpg)

Subtitle: "Real-world dexterous control with ShadowHand".

Left, a video frame (a single still) of the real black robot hand with teal fingertips holding a yellow and a pink ball, a cardboard piece at the left and a marker tag above, on a dark surface (the same scene as the slide 5 frame). In the recording (≈1:12:06–1:12:52) Finn plays this as a video of the method on a real hand, and adds that when the hand dropped a ball, the system was reset by another robot.

Right, three dashes: "Efficiency is key for fragile hardware"; "Learns Baoding ball rotation in ~4 hours"; "Ball is reset with another robot arm".

Below, a line chart. Y axis "Success Rate (%)", 0 to 100 in steps of 20; x axis "Amount of data (hours)", ticks 0, 1, 2, 3, 4. Two series, read approximately:

- 90° turns (blue): 0 at 0 hours, about 70 at 0.9, about 88 at 1.4, a dip to about 65 at 1.8, about 80 at 2.0, then a nearly straight rise to 100 at about 4.1.
- 180° turns (orange): 0 at 0 and near 0 until about 0.3, about 5 at 0.9, about 35 at 1.4, about 28 at 1.8, about 35 at 2.0, then a straight rise to about 52 at about 4.1.

Next to the chart, a photograph of the real setup: a black robot hand at the lower left holding a yellow ball in a cardboard ramp, and a white and black robot arm (at the right) holding a red ball above it, with a metal frame behind.

## Slide 37 — The plan for today

Same agenda as slide 5 (no bold item, no robot image or learning goals), as a closing recap.

## Slide 38 — Course reminders

- Homework 3 due next Friday, May 16
- Project milestone due Friday, May 23

Then "Next time": "RL for multiple tasks and goals (e.g. learning generalist policies)".
