---
title: "Lecture 13 — Meta RL (slide deck)"
lecture: 13
slides: 39
printed_numbers: 1–39
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/13_cs224r_metarl_2025.pdf
numbering: 1:1 with PDF pages. Pages 1, 2, 5, 8, 13–18, 20–22, 26, 27, 38 and 39 print their number at bottom centre in the text layer; the other 24 pages (3, 4, 6, 7, 9–12, 19, 23–25, 28–37) print no number in the text layer, and their labels are inferred but unambiguous because they sit between consecutive numbered pages with no gap
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure and table slides (12, 19, 21, 22, 26, 35) were checked by Claude Opus 5.5 against 1400px renders, with crops of slide 21's table and the charts of slides 22 and 26. Slides 12, 21 (every cell), 22 (series, colours, arrows and values) and 35 agree, and slide 26's series, colours, legend and values agree. Corrections: slide 26's dashed lines had been called the other methods' final performance, which the slide does not say; slide 19's attention network has seven rows of nodes in a different order from the one given. Slide 20's images were found to be frames of two videos that play in the lecture (≈33:08–34:40), which the transcription had not said. The text layer confirms slide 22's printed "ICML 2019", slide 9's "using on", and plain n on slide 13.
images: 19 slides are rendered and shown under their headings (3, 6, 8, 9, 11, 12, 16, 17, 19, 20, 22, 23, 25, 26, 30, 32, 33, 35, 36). Not rendered: build steps whose figure a rendered slide shows (10, whose two mazes slide 11 repeats; 24, slide 16's chain with a question arrow, answered by slide 25); slide 21's table, transcribed here cell by cell; the episodic and online diagram and the slides that repeat the chains of slides 16 and 17 beside text (14, 27, 31, 34); equation and text slides (4, 7, 13, 18, 29, 37); and the title, reminder, agenda and divider slides (1, 2, 5, 15, 28, 38, 39).
---

# Lecture 13 — Meta RL: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025) Lecture 13 deck, transcribed from the PDF. The title slide (slide 1) prints "Meta Reinforcement Learning" and "CS 224R", and names no lecturer. Cite these as "slide N". Diagrams, photographs and plots are described in prose since the KB is read as text.

The recording (about 69 minutes) opens with the recaps of multi-task and goal-conditioned learning (slides 3–4) and the plan (slide 5), without discussing slide 2's reminders. It follows the deck through slide 37 and ends at about 1:09:10, on slide 37's summary of alternative exploration strategies, so the closing plan, next time and reminders (slides 38–39) are not shown or discussed: for them this file is the only source. Finn writes the black-box meta-RL algorithm on a board the recording does not show (≈24:34–26:55) before slide 17 prints it, with its meta-test steps (≈30:03–30:49) and a discount factor (≈36:10); the transcript is the only source for those. Slide 20's images are frames of two videos that play in the lecture (≈33:08–34:40); what they showed is under that slide.

## Slide numbers vs PDF pages

The deck is 1:1 with its page index: PDF page N is slide N, for all 39 pages. Pages 1, 2, 5, 8, 13, 14, 15, 16, 17, 18, 20, 21, 22, 26, 27, 38 and 39 print their number at bottom centre in the text layer, each equal to its page index. The other 24 pages print no number in the text layer; their labels are inferred but unambiguous because they sit between consecutive numbered pages with no gap.

| PDF page | Slide |
| -------- | ----- |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
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
| 39 | 39 |

## Sections

| Section | Slides |
| ------- | ------ |
| Title, reminders, recaps of multi-task and goal-conditioned learning, plan | 1–5 |
| Motivation: why meta-learning, transfer framings, few-shot learning outside RL | 6–9 |
| The meta-RL problem: maze example, task distributions, formal settings, episodic and online variants | 10–14 |
| Plan (black-box methods highlighted) | 15 |
| Black-box meta-RL: overview, algorithm, architectures and optimizers, the first two examples (maze navigation, test-time compute) | 16–22 |
| Digression: connection to multi-task policies, and the POMDP view | 23–25 |
| Meta-RL example 3 (continuous control, PEARL) | 26 |
| Summary of black-box methods | 27 |
| Exploration in meta-RL: end-to-end optimization (hallway and kitchen examples, the coupling problem), alternative strategies (posterior sampling, intrinsic rewards, dynamics and reward prediction) | 28–37 |
| Plan (closing), next time and reminders | 38–39 |

## Slide 1 — Meta Reinforcement Learning

Title slide. Centred title "Meta Reinforcement Learning" with "CS 224R" below it. No lecturer or affiliation is printed.

## Slide 2 — Reminders

Two blocks, both centred. "Reminders": "Homework 3 due **Friday**" and "Project milestone due **next Friday**" (the bold words are Friday and next Friday). "Announcements": "**More evening office hours!**" and, in smaller text, "(Andy and Marcel moving to evening times)".

## Slide 3 — Recap: Multi-task learning

![Slide 3 — Recap: Multi-task learning](../images/13-meta-rl/slide-3.jpg)

Subtitle: "Multi-task RL as single-task RL in a joint MDP", with the equation $\mathbf{s} = (\bar{\mathbf{s}}, \mathbf{z}_ i)$ (state is the task-free state $\bar{\mathbf{s}}$ paired with the task identifier $\mathbf{z}_ i$; the bar is printed over the bold s).

Diagram. At far left the text "Sample task $i$" with three dotted arrows fanning out to three rows, with a vertical ellipsis between the second and third rows. Each row is an MDP, labelled at the far right "MDP 1", "MDP 2", "MDP $N$". Row 1: initial-state distribution $p_1(\bar{\mathbf{s}}_ 1)$, then a solid arrow from $\bar{\mathbf{s}}_ 1$ to $\bar{\mathbf{s}}_ 2$ labelled $\pi(\mathbf{a} \mid \bar{\mathbf{s}}_ 1, \mathbf{z}_ 1)$, then a second solid arrow to "etc..". Row 2 is the same with $p_2(\bar{\mathbf{s}}_ 1)$ and $\pi(\mathbf{a} \mid \bar{\mathbf{s}}_ 1, \mathbf{z}_ 2)$. Row $N$ has $p_N(\bar{\mathbf{s}}_ 1)$ and $\pi(\mathbf{a} \mid \bar{\mathbf{s}}_ 1, \mathbf{z}_ N)$. The figure shows that a sampled task picks which MDP's initial states and which task identifier the single policy sees.

Bottom line: "This is going to be a bit different for meta-RL!"

## Slide 4 — Recap: Goal-conditioned learning

"Special case of multi-task RL, where $\mathbf{z}_ i = \mathbf{s}_ g$". "Each task aims to reach some goal state $\mathbf{s}_ g$".

Printed as two equation lines, each followed by its case. For discrete states:

$$r(\mathbf{s}, \mathbf{a}, \mathbf{s}_ g) = \delta(\mathbf{s} = \mathbf{s}_ g)$$

For continuous states:

$$r(\mathbf{s}, \mathbf{a}, \mathbf{s}_ g) = \delta(\lVert \mathbf{s} - \mathbf{s}_ g \rVert \leq \epsilon)$$

Green: "+ No need to define reward (self-supervised!)" and "+ Many tasks can be formed as goal reaching tasks". Red: "- Can be pretty hard to train".

## Slide 5 — Plan for Today

Numbered list: 1. "Meta-RL problem statement"; 2. "Black-box meta-RL methods"; then "Next time: Exploration + Learning to explore." A tall bracket spanning the two list items and the next-time line, at right, is labelled "part of HW4" (the bracket reaches down to the next-time line in the render).

Bottom, "Lecture goals:" with two dashes: "Understand the meta-RL problem statement & set-up" and "Understand the basics & challenges of black-box meta RL algorithms", with "meta-RL problem statement" in red and "black-box meta RL algorithms" in blue.

## Slide 6 — What is the problem?

![Slide 6 — What is the problem?](../images/13-meta-rl/slide-6.jpg)

"Say you wanted to learn to make coffee with a new espresso machine." Left: a photograph of an espresso machine with two portafilter spouts pouring espresso into a small brown cup, one hand holding the cup from below and another hand at the top of the machine; a tattooed forearm is visible. Right: "Training a robot to do this with PPO would take **millions of attempts**." and "With some instruction or prior coffee experience, a person could learn in **minutes**."

Below: "People aren't starting from scratch!" then "Experience with similar tasks, either espresso-making on other machines or general motor control." and "Can also learn to solve a challenging math problem more quickly using past problem solving experience!"

## Slide 7 — Can RL algorithms leverage experience from previous tasks when learning a new task?

Title: "Can RL algorithms leverage experience from *previous* tasks when learning a *new* task?" (previous and new italic). Subtitle "Framing transfer learning problems". Three numbered, colour-coded items:

1. (magenta) "Forward transfer: learn policies that transfer effectively". a) "Train on source task, then fine-tune on target task".
2. (green) "Multi-task transfer: train on many tasks, transfer to a new task". a) "Task prompt/descriptor $\mathbf{z}_ i$ needs to capture task structure for zero-shot transfer" ("zero-shot" in orange).
3. (blue) "Meta-learning: learn to *learn* on many tasks". a) "Accounts for the fact that we'll be adapting to a new task during training!" b) "The "task descriptor" is a few data examples provided in-context".

Then "For (1): Source and target tasks must be similar" and "For (2) and (3): Target task must be similar to the *distribution* of training tasks", with the numbers coloured as above. Footer small grey: "Slide adapted from Sergey Levine".

## Slide 8 — What does few-shot learning look like outside of RL?

![Slide 8 — What does few-shot learning look like outside of RL?](../images/13-meta-rl/slide-8.jpg)

Left, headed "training data", two columns of three paintings each, headed "Braque" and "Cezanne". Braque column: three cubist paintings in browns, teals and greys (a house among trees, a bowl-like still life, boats in a harbour). Cezanne column: an orchard with a house, a bowl of red apples with lemons, a still life with a bottle, carafe and fruit on a cloth. Right, headed "test datapoint", a larger painting of a cubist landscape with an arched viaduct above orange-roofed houses and green trees. Below it in blue: "By Braque or Cezanne?"

## Slide 9 — What does few-shot learning look like outside of RL?

![Slide 9 — What does few-shot learning look like outside of RL?](../images/13-meta-rl/slide-9.jpg)

Same title as slide 8; builds the figure into a formal framing. Two rows.

Row 1, labelled "Few-shot image classification": a teal rounded box labelled $D_{\text{train}}$ (teal) holds the same six paintings in two rows labelled "Braque" and "Cezanne". To its right the test painting is labelled $x$, with the text $\hat{y}$ = Braque beside it.

Row 2, labelled "In-context learning in LLMs": a teal rounded box labelled $D_{\text{train}}$ holds monospaced text:

```
Poor English input: I eated the purple berries.
Good English output: I ate the purple berries.
Poor English input: Thank you for picking me as your designer. I'd appreciate it.
Good English output: Thank you for choosing me as your designer. I appreciate it.
Poor English input: The mentioned changes have done. or I did the alteration that you requested. or I changed things you wanted and did the modifications.
Good English output: The requested changes have been made. or I made the alteration that you requested. or I changed things you wanted and made the modifications.
```

Below the box, in grey, the query labelled $x$: "Poor English input: Please provide me with a short brief of the design you're looking for and that'd be nice if you could share some examples or project you did before." Then in bold black, labelled $\hat{y}$: "Good English output: Please provide me with a brief description of the design you're looking for and that would be nice if you could share some examples or projects you have done before."

Bottom: "Train a model $\hat{y} = f(x, D_{\text{train}})$ to be able to make predictions using on a few examples" (the printed text reads "using on a few examples").

## Slide 10 — Meta-RL Example: Maze Navigation

Left label "Goal:". Two maze diagrams side by side. Left maze, headed "Collect small amount of experience in new MDP": a line-drawn maze with a small blue square (the goal) near the middle; blue and red dotted trajectories wander around the outer corridors, one reaching near the goal from the left, and the other running along the top, right and bottom edges. Caption beneath: "Collect $\mathscr{D}_ {\text{tr}} \sim \pi^{\text{exp}}$". Right maze, headed "Learn policy that solves that MDP": the same maze with the blue goal square, and a short blue/pink trajectory that goes directly from the top-left corridor down to the goal. Caption: $\mathscr{D}_ {\text{tr}} \rightarrow \pi^{\text{task}}$.

Bottom: "Exploration-exploitation trade-off is a unique part of meta-RL!" Footer: "diagram adapted from Duan et al. '17".

## Slide 11 — Meta-RL Example: Maze Navigation

![Slide 11 — Meta-RL Example: Maze Navigation](../images/13-meta-rl/slide-11.jpg)

Builds on slide 10: slide 10's two test-time mazes now sit at the right, and a meta-training panel is new at the left.

Left, headed "Meta-Train Time:" with "Learn how to efficiently explore & solve many MDPs:". Two different mazes, each with a goal square (the first has a blue square at bottom left, the second a blue square at left-middle and a pink square at bottom) and dotted blue/red/magenta trajectories exploring toward the goal; then three dots and "meta-training tasks". Caption: "Meta-train $\pi^{\text{exp}}, \pi^{\text{task}}$".

Right, headed "Meta-Test Time:", the slide 10 pair: "Collect small amount of experience in new MDP" (maze with exploration trajectories, caption "Collect $\mathscr{D}_ {\text{tr}} \sim \pi^{\text{exp}}$") and "Learn policy that solves that MDP" (maze with direct path, caption $\mathscr{D}_ {\text{tr}} \rightarrow \pi^{\text{task}}$).

Bottom, in blue: "Key assumption: Meta-testing MDPs come from same task distribution as meta-training MDPs." then in black "(so that we can expect generalization)". Footer: "diagram adapted from Duan et al. '17".

## Slide 12 — Meta-reinforcement learning task distributions

![Slide 12 — Meta-reinforcement learning task distributions](../images/13-meta-rl/slide-12.jpg)

Subtitle "Examples of meta-RL tasks". Four panels:

- "Navigation through different mazes": the two meta-training mazes from slide 11 with trajectories, then three dots.
- "Locomotion on different terrains, slopes": three photographs, each showing a small dark robot on a different floor: green artificial grass, a plain grey floor, and a pinkish-brown surface; then three dots.
- "Object manipulation with different objects, goals": a grid image headed "Train tasks" of small simulated robot-arm scenes in five rows of nine, each captioned with a tiny task name (the first is "1. turn on faucet", then "2. sweep", "3. stack", "4. unstack", and so on to "45. open drawer"; the captions are too small to read reliably beyond these), and a column headed "Test tasks" of five scenes numbered 46 to 50 ("46. open box", "47. close box", "48. lock door", "49. unlock door", "50. pick bin" as far as legible).
- "Dialog with different users w/ different preferences": clip-art of two overlapping speech bubbles (light blue and lime green) above four coloured head-and-shoulders silhouettes (blue, pink, green, orange).

## Slide 13 — More Formal Problem Settings

Three coloured blocks.

"Multi-Task Learning" (green): "Solve multiple tasks $\mathcal{T}_ 1, \cdots, \mathcal{T}_ T$ at once." with

$$\min_ {\theta} \sum_ {i=1}^{T} \mathcal{L}_ i(\theta, \mathcal{D}_ i)$$

"Transfer Learning" (magenta): "Solve target task $\mathcal{T}_ b$ after solving source task $\mathcal{T}_ a$ by *transferring* knowledge learned from $\mathcal{T}_ a$".

"The Meta-Learning Problem" (blue): "Given data from $\mathcal{T}_ 1, \ldots, \mathcal{T}_ {n}$, quickly solve new task $\mathcal{T}_ {\text{test}}$" (a small prime-like mark follows the $n$ in the render; the text layer reads a plain n).

Bottom: "A reinforcement learning task:" ("task" in teal) followed by

$$\mathcal{T}_ i \triangleq \lbrace \mathcal{S}_ i, \mathcal{A}_ i, p_i(\mathbf{s}_ 1), p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}), r_i(\mathbf{s}, \mathbf{a}) \rbrace$$

with grey arrows labelling the terms: "state space" points to $\mathcal{S}_ i$, "action space" to $\mathcal{A}_ i$, "initial state distribution" to $p_i(\mathbf{s}_ 1)$, "dynamics" to $p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$, and "reward" to $r_i(\mathbf{s}, \mathbf{a})$. At right: "Meta-reinforcement learning = meta-learning with RL tasks" ("meta-learning" in blue).

## Slide 14 — The Meta Reinforcement Learning Problem

Two rows of the same input-output diagram, each in green with a green curved arrow.

Row 1, label "*Episodic Variant*": "Inputs: $\mathcal{D}_ {\text{train}}$ $\mathbf{s}_ t$" with a curly brace under $\mathcal{D}_ {\text{train}}$ labelled "k rollouts from $\pi^{\text{exp}}$"; "Outputs: $\mathbf{a}_ t$". The curved arrow runs from the inputs to the output and is labelled $\mathbf{a} \sim \pi^{\text{task}}(\cdot \mid \mathbf{s}, D_{\text{train}})$. A blue curved arrow from the blue text "can view as a task identifier $\mathbf{z}_ i$" points at that label (at $D_{\text{train}}$ inside it).

Row 2, label "*Online Variant*": the same inputs, outputs and policy label, with the brace labelled "1...k timesteps from $\pi^{\text{exp}}$".

Bottom: "**Note:** exploration policy $\pi^{\text{exp}}$ and adaptation policy $\pi^{\text{task}}$ could share parameters."

## Slide 15 — Plan for Today

The slide 5 agenda again without the lecture goals: "1. Meta-RL problem statement", "**2. Black-box meta-RL methods**" (item 2 now in bold, marking the current section), "Next time: Exploration + Learning to explore." The bracket at right is again labelled "part of HW4".

## Slide 16 — Black-Box Meta-RL: Overview

![Slide 16 — Black-Box Meta-RL: Overview](../images/13-meta-rl/slide-16.png)

Left, in blue: "Black-box neural net" with "(Transformer, NN with memory)" below, then $\mathbf{a} \sim \pi^{\text{task}}(\cdot \mid \mathbf{s}, D_{\text{train}})$.

Diagram at right: a left-to-right chain of four boxes (the last separated by "..."), each with an arrow going up to an action and an arrow coming up from its input, and a horizontal arrow from each box to the next (the hidden state). Inputs from left to right: $\mathbf{s}_ 1, 0$; $\mathbf{s}_ 2, r_ 1$; $\mathbf{s}_ 3, r_ 2$; and, for the last box, $\mathbf{s}_ t, r_ {t-1}$ labelled "current state". Outputs: $\mathbf{a}_ 1$, $\mathbf{a}_ 2$, $\mathbf{a}_ 3$, $\mathbf{a}_ t$. A grey brace under the first three inputs and the dots is labelled $D_{\text{train}}$ with grey text "context/training set gets larger over time".

Below, in teal: "**Question:** How is this different from simply doing RL with a recurrent policy?" Two answers: "Reward is passed as input (& trained across multiple MDPs)" and "Hidden state maintained **across episodes** within a task!"

## Slide 17 — Black-Box Meta-RL: Algorithm

![Slide 17 — Black-Box Meta-RL: Algorithm](../images/13-meta-rl/slide-17.jpg)

Diagram: the recurrent chain of slide 16 unrolled over episodes. Boxes with inputs $\mathbf{s}_ 1, 0$; $\mathbf{s}_ 2, r_ 1$; ...; $\mathbf{s}_ T, r_ {T-1}$ and outputs $\mathbf{a}_ 1$, $\mathbf{a}_ 2$, ... form "Episode 1" (grey brace). The hidden-state arrow continues from the last box of Episode 1 into the first box of Episode 2, whose inputs again start $\mathbf{s}_ 1, 0$; $\mathbf{s}_ 2, r_ 1$; ... $\mathbf{s}_ T, r_ {T-1}$ ("Episode 2", grey brace). A vertical dashed line then separates a further box at the far right with input $\mathbf{s}_ 1, 0$ and output $\mathbf{a}_ 1$, cut off by the slide edge: the start of a new task, where, as Finn says, the memory is cut when collection starts for a newly sampled MDP (≈27:43). Finn notes that $n = 2$ in this diagram (≈27:43).

Numbered list (serif type), with a blue curved arrow from item 4 back up to item 1:

1. Sample task $\mathcal{T}_ i$
2. Roll-out policy $\pi(a \mid s, \mathscr{D}_ i^{\text{tr}})$ for N episodes
3. Store sequence in replay buffer for task $\mathcal{T}_ i$.
4. Update policy to maximize discounted return for all tasks.

At right of item 2: the parenthetical "(under dynamics" $p_i(s' \mid s, a)$ "and reward" $r_i(s, a)$ ")". The slide prints $a$, $s$ non-bold here.

## Slide 18 — Black-Box Meta-RL: Algorithm

Builds on slide 17 with the diagram removed and a meta-test section added. "Meta-Training": the same four steps with the blue arrow from step 4 to step 1 and the parenthetical "(under dynamics" $p_i(s' \mid s, a)$ "and reward" $r_i(s, a)$ ")". "Meta-Test Time": 1. "Sample *new* task $\mathcal{T}_ j$"; 2. "Roll-out policy $\pi(a \mid s, \mathscr{D}_ j^{\text{tr}})$ for up to N episodes".

## Slide 19 — Black-Box Meta-RL: Architectures & Optimizers

![Slide 19 — Black-Box Meta-RL: Architectures & Optimizers](../images/13-meta-rl/slide-19.jpg)

No number printed in the text layer. Three panels, each pairing an architecture (blue) with an optimizer (green).

Top left: "RNN architecture" (blue), "TRPO/A3C (similar to PPO)" (green). Diagram: a dashed box "Agent" holding a row of recurrent hidden-state squares $h_0$ to $h_6$ joined left to right; below, a dashed box "MDP 1" holding two episodes of states in circles: Episode 1 is $s_0, s_1, s_2, s_3$ and Episode 2 is $s_0, s_1, s_2$ (each episode under a brace). Arrows go from the states up to the hidden states, labelled $r_0,d_0$, $r_1,d_1$, $r_2,d_2$ (and $r_0,d_0$, $r_1,d_1$ in Episode 2), and from the hidden states back down to the next state, labelled actions $a_0$, $a_1$, $a_2$ (and $a_0$, $a_1$). Citations in grey: "Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel. *RL²: Fast Reinforcement Learning via Slow Reinforcement Learning*. 2017" and "Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick. *Learning to Reinforcement Learn*. CogSci 2017".

Top right: "Attention + 1D conv" (blue), "TRPO (similar to PPO)" (green). Citation: "Mishra, Rohaninejad, Chen, Abbeel. *A Simple Neural Attentive Meta-Learner*. ICLR 2018". Diagram titled "Reinforcement Learning": a dashed box containing four columns of coloured nodes in seven rows: from the bottom, a red row, two orange rows, a green row, two more orange rows and a green row at the top. Short diagonal lines join each orange node to the node one column to its left in the row below, and curved arrows run into each green node from the nodes of earlier columns in the row below it. Above the box, actions $a_{t-3}, a_{t-2}, a_{t-1}, a_t$ ("Actions"); below it, inputs, left to right, $o_{t-3}$ with two dashes; $o_{t-2}$, $a_{t-3}$, $r_{t-3}$; $o_{t-1}$, $a_{t-2}$, $r_{t-2}$; and $o_t$, $a_{t-1}$, $r_{t-1}$, labelled "(Observations, Actions, Rewards)".

Bottom: "Feedforward + average" (blue), "SAC (off-policy with replay buffer)" (green). Diagram: transition tuples $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)_ 1$ down to $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)_ N$ each feed a blue box $\phi$ (a vertical dashed double arrow links the two boxes, indicating shared weights), producing $\Psi_ \phi(\mathbf{z} \mid \mathbf{c}_ 1)$ through $\Psi_ \phi(\mathbf{z} \mid \mathbf{c}_ N)$; these are combined with a multiplication sign into a blue Gaussian bell curve labelled $q_\phi(\mathbf{z} \mid \mathbf{c})$; an arrow leads to two blue boxes $Q_\theta(\mathbf{s}, \mathbf{a}, \mathbf{z})$ and $\pi_\theta(\mathbf{a} \mid \mathbf{s}, \mathbf{z})$, which feed boxes $\mathcal{L}_ {critic}$ (orange) and $\mathcal{L}_ {actor}$ (green), with coloured arrows from the loss boxes back to the networks. Citation: "Rakelly, Zhou, Quillen, Finn, Levine. *Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables*. ICML 2019."

## Slide 20 — Meta-RL Example #1

![Slide 20 — Meta-RL Example #1](../images/13-meta-rl/slide-20.jpg)

"From: Mishra, Rohaninejad, Chen, Abbeel. *A Simple Neural Attentive Meta-Learner*. ICLR 2018". "**Experiment:** Learning to visually navigate a maze" with two dashes: "train on 1000 small mazes", "test on held-out small mazes and large mazes".

Four images in a row: (1) a close-up first-person view of a blue stone wall; (2) a top-down line drawing of a small maze with a blue square goal at bottom left and a small red dot near the centre (the agent); (3) a first-person view down a corridor with blue stone walls, grey floor tiles and a grey ceiling; (4) a top-down line drawing of a larger maze with a small blue square goal at bottom right. The pairs show what the agent sees (pixels) next to the maze layout. The four images are frames of two videos that play in the lecture, one for a small maze and one for a large maze, each showing the agent's first-person view at left and "the trace of the exploration" on the layout at right; the agent "only gets these images as input" (≈33:08). In the small maze the agent starts in the middle, explores, and eventually finds the goal; in the second episode, still with its memory of the first, it goes "straight to the goal because it decided that it already explored the extent of the maze" (≈33:08–34:40). Finn replays the first video ("So kind of watching that again", ≈33:53). In the large maze it starts at one point, explores the top part, does not see the goal and treats it as a dead end, then sees the goal, and after that first episode goes straight to it (≈34:40). The PDF holds only these frames.

## Slide 21 — Meta-RL Example #1

Same header, experiment text and citation as slide 20. A table, captioned "Table 5: Average time to find the goal on each episode":

| Method | Small Maze Episode 1 | Small Maze Episode 2 | Large Maze Episode 1 | Large Maze Episode 2 |
| --- | --- | --- | --- | --- |
| Random | 188.6 ± 3.5 | 187.7 ± 3.5 | 420.2 ± 1.2 | 420.8 ± 1.2 |
| LSTM | 52.4 ± 1.3 | 39.1 ± 0.9 | 180.1 ± 6.0 | 150.6 ± 5.9 |
| SNAIL (ours) | **50.3 ± 0.3** | **34.8 ± 0.2** | **140.5 ± 4.2** | **105.9 ± 2.4** |

The SNAIL row is bold in the original.

## Slide 22 — Meta-RL Example #2

![Slide 22 — Meta-RL Example #2](../images/13-meta-rl/slide-22.jpg)

Citation: "Qu, Yang, Setlur, Tunstall, Beeching, Salakhutdinov, Kumar. *Optimizing Test-Time Compute via Meta Reinforcement Finetuning*. ICML 2019." (printed year as given). Top left, a pink rounded box: "Tasks: solving different math problems". Next to it a blue box with a math problem: "Show that the inequality $\sum_{i=1}^{n}\sum_{j=1}^{n}\sqrt{\lvert x_i - x_j \rvert} \leq \sum_{i=1}^{n}\sum_{j=1}^{n}\sqrt{\lvert x_i + x_j \rvert}$ holds for all real numbers". Right of it: "Explore different strategies for solving the problem & then produce answer based on best strategy."

Lower left, headed "DeepSeek R1", two cards. "Hard Problem": the same inequality in a blue box, then a green box reading "<think> Okay, so I need to show that for any real numbers ... Alternatively, **perhaps there's a way to pair terms or use symmetry?** Alternatively, **could we relate this inequality to some function property?** ... Time is up </think> Step-by-Step Explanation: ...". "Easy Problem": a blue box "2+2=?" and a green box "<think> Okay, so I need to figure out what 2 plus 2 equals. Let me count them out. One, two, and then three, four... </think> Step-by-Step Explanation: - Start with the numbers: 2 and 2. - Combine their quantities: Add the two numbers together. - Result: 2 + 2 = 4. Answer: 4". Under the cards: "**Key idea**: Can we optimize test-time compute?"

Lower right, a line chart headed "Math 500". X axis "Number of Tokens", ticks 10000 to 40000; y axis "Success Rate", ticks 0.20 to 0.50. Three series, ten markers each, at roughly 4000 tokens and then every 4000 tokens to about 39000 (approximate values):

- Base (blue, downward triangles): 0.18, 0.245, 0.285, 0.31, 0.33, 0.35, 0.355, 0.365, 0.375, 0.375.
- GRPO (green, stars): 0.265, 0.35, 0.385, 0.415, 0.44, 0.45, 0.46, 0.47, 0.48, 0.48.
- MRT (orange, stars): 0.30, 0.38, 0.42, 0.45, 0.465, 0.48, 0.49, 0.50, 0.51, 0.51.

Two black dashed horizontal arrows point left: one at about 0.48 from GRPO's last point back to where MRT reaches 0.48, labelled "1.6x"; one at about 0.375 from Base's last point back to where MRT reaches 0.38 (about 8000 tokens), labelled "5.4x". Under the chart: "Higher performance with same # of tokens" and "Same accuracy with 1.6x fewer tokens".

## Slide 23 — Digression: Connection to Multi-Task Policies

![Slide 23 — Digression: Connection to Multi-Task Policies](../images/13-meta-rl/slide-23.jpg)

No number printed in the text layer. "multi-task policy: $\pi_\theta(\mathbf{a} \mid \mathbf{s}, \mathbf{z}_ i)$". Two photographs/frames: a robot gripper holding a red Lego-style brick above a stack of yellow, blue and green bricks with red and green square outlines marking positions, captioned with $\mathbf{z}_ i$ then ": stack location"; and a simulated blue two-legged walker on a grey grid floor with a green target marker, captioned with $\mathbf{z}_ i$ then ": walking direction".

At right, a diagram of a recurrent chain of four boxes. The first three take inputs $(s_1, a_1, r_1)$, $(s_2, a_2, r_2)$, $(s_3, a_3, r_3)$, under a brace labelled "task identifier $\mathbf{z}_ i$"; the fourth takes $s_4$ and outputs $a_4$; a brace on the right labels the whole chain "policy". Caption: "Multi-task policy with experience as task identifier."

Bottom: "What about **goal-conditioned policies**?" with two dashes: "rewards are a strict generalization of goals"; "meta-RL objective is to *adapt* new tasks vs. *generalize* to new goals (**k-shot** vs. **0-shot**)".

## Slide 24 — Another way to look at it

No number printed in the text layer. A recurrent chain like slide 16: inputs $\mathbf{s}_ 1, 0$; $\mathbf{s}_ 2, r_ 1$; $\mathbf{s}_ 3, r_ 2$; ...; $\mathbf{s}_ t, r_ {t-1}$, outputs $\mathbf{a}_ 1$, $\mathbf{a}_ 2$, $\mathbf{a}_ 3$, $\mathbf{a}_ t$, a brace "policy". A magenta curved arrow points at the arrow between the dots and the last box, labelled in magenta "What information is passed here?"

Text: "When $\mathbf{z}_ i$ isn't in the state, there's ambiguity about the MDP." "Policy needs to *infer* $\mathbf{z}_ i$ from its experience."

## Slide 25 — Another way to look at it

![Slide 25 — Another way to look at it](../images/13-meta-rl/slide-25.png)

No number printed in the text layer. Builds on slide 24 by answering its question: the chain is split in two. At left, the first three boxes (inputs $\mathbf{s}_ 1, 0$; $\mathbf{s}_ 2, r_ 1$; $\mathbf{s}_ 3, r_ 2$; outputs $\mathbf{a}_ 1$, $\mathbf{a}_ 2$, $\mathbf{a}_ 3$) end in an arrow to $\hat{\mathbf{z}}_ i$. At right, a single box takes $\hat{\mathbf{z}}_ i$ from the left and $\mathbf{s}_ t, r_ {t-1}$ from below, outputs $\mathbf{a}_ t$, and is braced "policy". The same two text lines as slide 24 follow, then "Meta-RL as exploring to infer the unknown task, and then performing the task." and "A special kind of **partially observed** Markov decision process (POMDP)".

## Slide 26 — Meta-RL Example #3

![Slide 26 — Meta-RL Example #3](../images/13-meta-rl/slide-26.jpg)

Citation: "Rakelly, Zhou, Quillen, Finn, Levine. *Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables*. ICML 2019." "**Experiment:** Continuous control problems", with four simulation stills on checkerboard floors captioned "Half Cheetah", "Humanoid", "Ant", "Walker". Dashes: "different directions, velocities", "different physical dynamics". Then "Meta-RL algos are very efficient at new tasks." and "But, what about **meta-training efficiency**?" Teal: "**Question**: Do you expect off-policy meta-RL to be more or less efficient than on-policy meta-RL?"

Right, four line charts (a 2x2 grid), four series each, legend at bottom: "Black-box:" PEARL (blue) and RL2 (olive green); "Opt-based:" ProMP (purple) and MAML (orange). Each panel shows the learning curves as solid lines, with dashed horizontal lines in the colours of the methods; the slide does not say what the dashed lines mark, and the lecture does not mention them. Approximate readings:

- "Half-Cheetah-Vel" (x axis 0.0 to 2.0, scale 1e6; y axis about -400 to 0): PEARL rises steeply from about -320 at 0.2e6 to about -30 by 0.4e6 and stays near -30 (the top of the chart); RL2 climbs slowly from about -400 to about -250 at 2.0e6; ProMP and MAML curves sit around -400 to -350 and never rise above about -350. Dashed lines: olive near -25 at the top, at about the level of PEARL's plateau; orange near -120 and purple near -145.
- "Humanoid-Direc-2D" (x 0.0 to 1.0, scale 1e6; y 100 to 1000): PEARL rises from about 250 to about 800–900 with a noisy band; RL2, ProMP and MAML stay near 100–150. Dashed: purple near 520, olive near 280, orange near 200.
- "Ant-Goal-2D" (x 0.0 to 1.0, scale 1e7; y -1000 to -200): PEARL rises quickly to about -250 by 0.2e7 and levels near -200; RL2 rises from about -850 to about -650; ProMP and MAML stay around -900 to -800. Dashed lines: orange near -375, purple near -420, olive near -500.
- "Walker-2D-Params" (x 0 to 5, scale 1e6; y 0 to 800): PEARL rises from about 150 to about 730 at 5e6; RL2 reaches about 200; ProMP about 100; MAML about 30. Dashed lines: olive near 535, orange near 490, purple near 380.

The left tick labels of the first and third panels are partly cut off at the left edge (a stray "5" and "0"). In every panel the blue PEARL curve rises far above the other three solid curves within the plotted time steps.

## Slide 27 — Summary So Far

Left, in blue: "Black-box neural net" with "(Transformer, NN with memory)" and $\mathbf{a} \sim \pi^{\text{task}}(\cdot \mid \mathbf{s}, D_{\text{train}})$. Right: the recurrent-chain diagram of slide 16 without its brace and labels: boxes with inputs $\mathbf{s}_ 1, 0$; $\mathbf{s}_ 2, r_ 1$; $\mathbf{s}_ 3, r_ 2$; ...; $\mathbf{s}_ t, r_ {t-1}$ and outputs $\mathbf{a}_ 1$, $\mathbf{a}_ 2$, $\mathbf{a}_ 3$, $\mathbf{a}_ t$.

Pros and cons list, coloured: green "+ general & expressive", green "+ a variety of design choices in architecture", red "-- hard to optimize", orange "~ inherits sample efficiency from outer RL optimizer".

## Slide 28 — How should we think about exploration in meta-RL?

No number printed in the text layer. Section-divider slide: the single centred line "How should we think about exploration in meta-RL?"

## Slide 29 — How Do We Learn to Explore?

No number printed in the text layer. "Solution #1: Optimize for Exploration & Exploitation *End-to-End* w.r.t. Reward" (End-to-End in orange italics). Small print: "(Duan et al., 2016, Wang et al., 2016, Mishra et al., 2017, Stadie et al., 2018, Zintgraf et al., 2019, Kamienny et al., 2020)". Green: "+ simple", "+ in principle, it yields optimal exploration-exploitation trade-off". Red: "-- challenging optimization when exploration is hard".

## Slide 30 — A simple, running example

![Slide 30 — A simple, running example](../images/13-meta-rl/slide-30.jpg)

No number printed in the text layer. Three identical photographs of a dim hotel-style corridor with wooden floor, doors on both sides and ceiling lights receding to a vanishing point, labelled "Hallway 1", "Hallway 2" and "Hallway N", with "..." between the second and third. A cartoon robot (green body with a red circular badge, red-and-yellow joints) is drawn in front of the middle photograph, labelled "agent". Bottom left: a clip-art stack of text documents with "information on where to go". Bottom right: "Different tasks: navigating to the ends of different hallways".

## Slide 31 — How Do We Learn to Explore?

No number printed in the text layer. "Solution #1: Optimize for Exploration & Exploitation *End-to-End* w.r.t. Task Reward" (printed here with "Task Reward", where slide 29 prints "Reward"), with the same small-print citation list as slide 29. At top right, the unrolled two-episode recurrent diagram of slide 17 (Episode 1, Episode 2, then a dashed line and the start of a new task), without the algorithm list.

"Example episodes during meta-training:" with three episodes, each paired with a comment at right:

- "agent goes to the end of the correct hallway": red, "- gets positive reward for current task, but $\mathscr{D}_ i^{\text{tr}}$ won't be different than for any other task".
- "agent goes to wrong hallway then correct hallway": grey, "+/- provides signal on a **suboptimal** exploration + exploitation strategy".
- "agent looks at the instructions": red, "- good exploratory behavior, but won't get any reward for this behavior".

Bottom, italic: "It's hard to learn exploration & exploitation at the same time!"

## Slide 32 — Another Example of a Hard Exploration Meta-RL Problem

![Slide 32 — Another Example of a Hard Exploration Meta-RL Problem](../images/13-meta-rl/slide-32.jpg)

No number printed in the text layer. Left, "Learned cooking tasks in previous kitchens": five photographs of different kitchens of varied styles (white island with stools and beams; white kitchen with pendant lights; blue cabinets and red tile floor with hanging pots; wood cabinets with a blue ceiling; marble island with stools and a wooden floor), captioned "meta-training". Right, "**Goal**: Quickly learn tasks in a new kitchen.": one larger photograph of a kitchen with a white brick wall, a hood, shelves with plants and utensils, a wooden counter, a gas hob and a white oven, captioned "meta-testing".

## Slide 33 — Why is End-to-End Training Hard in This Example?

![Slide 33 — Why is End-to-End Training Hard in This Example?](../images/13-meta-rl/slide-33.png)

No number printed in the text layer. "**End-to-end approach**: optimize exploration and execution episode behaviors end-to-end to maximize reward of execution" (the first phrase in orange).

Left of a vertical dashed divider: an egg emoji and a chicken emoji joined by two curved arrows, one from egg to chicken and one back, with the egg labelled "Learning to cook (execution)" and the chicken "Learning to find ingredients (exploration)". Right of the divider, two arrow rows: "Ingredient not found (bad exploration)" arrow to "Cannot learn to cook (bad execution)"; "Cannot cook (bad execution)" arrow to "Low reward for **any** exploration" (the bad-labelled parts in red).

"**Coupling problem**: learning exploration and execution depend on each other" then "—> can lead to poor local optima, poor sample efficiency". Footer: "Liu, Raghunathan, Liang, Finn. *Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices*. ICML 2021".

## Slide 34 — How Do We Learn to Explore?

No number printed in the text layer. Repeats slide 31's title ("Solution #1: Optimize for Exploration & Exploitation *End-to-End* w.r.t. Task Reward"), the same citation list and the two-episode diagram, then a summary: green "+ simple", green "+ leads to optimal strategy in principle", red "-- challenging optimization when exploration is hard".

## Slide 35 — Solution #2: Leverage Alternative Exploration Strategies

![Slide 35 — Solution #2: Leverage Alternative Exploration Strategies](../images/13-meta-rl/slide-35.jpg)

No number printed in the text layer. Title "Solution #2: Leverage Alternative Exploration Strategies" (the last three words in violet).

"2a. Use posterior sampling" with "(also called Thompson sampling)" and, at right, "PEARL (Rakelly, Zhou, Quillen, Finn, Levine. ICML '19)". Sub-steps: "i. Learn distribution over latent task variable $p(\mathbf{z})$, $q(\mathbf{z} \mid \mathscr{D}_ {\text{tr}})$ and corresponding task policies $\pi(\mathbf{a} \mid \mathbf{s}, \mathbf{z})$" and "ii. Sample $\mathbf{z}$ from current *posterior* and sample from policy $\pi(\mathbf{a} \mid \mathbf{s}, \mathbf{z})$".

Three panels, each a semicircular arc of blue discs (possible goal locations, with one darker disc at the top) around a start point at the bottom centre, with purple/magenta trajectories running from the start to goal discs, each ending in an x. Captions:

- Caption $\mathbf{z} \sim p(\mathbf{z})$: four trajectories, each ending at a different disc on the left and right, none reaching the top disc.
- Caption $\mathbf{z} \sim q_\phi(\mathbf{z} \mid c_{1:10})$: more trajectories (about eight) to discs around the arc, including one toward the top dark disc.
- Caption $\mathbf{z} \sim q_\phi(\mathbf{z} \mid c_{1:30})$: the same fan plus a bundle of orange trajectories going straight to the top dark disc, i.e. the sampled tasks have concentrated on that goal.

Bottom: "When might posterior sampling be bad?" and "Eg. Goals far away & sign on wall that tells you the correct goal."

## Slide 36 — Solution #2: Leverage Alternative Exploration Strategies

![Slide 36 — Solution #2: Leverage Alternative Exploration Strategies](../images/13-meta-rl/slide-36.jpg)

No number printed in the text layer. Builds on slide 35: item 2a's text is greyed out, and two items are new.

"2b. Use intrinsic rewards" with "MAME (Gurumurthy, Kumar, Sycara. CoRL '19)". "2c. Task dynamics & reward prediction" with "MetaCURE (Zhang, Wang, Hu, Chen, Fan, Zhang. '20)"; sub-steps "i. Train model $f(\mathbf{s}', r \mid \mathbf{s}, \mathbf{a}, \mathscr{D}_ {\text{train}})$" and "ii. Collect $\mathscr{D}_ {\text{train}}$ so that model is accurate."

Bottom left, a small arc of blue discs (a dark disc at far left and fainter discs along the arc) with a purple trajectory arching over the top toward the right, a magenta one to the right, and a yellow-orange one to the dark disc at the left, each ending in an x. Beside it: "When might this be bad?" and "Lots of distractors, or complex, high-dim state dynamics".

## Slide 37 — Solution #2: Leverage Alternative Exploration Strategies

No number printed in the text layer. Same as slide 36 with 2a no longer greyed (2a, 2b, 2c all in full) and the trajectory figure and its "When might this be bad?" text replaced by a summary: green "+ easy to optimize", green "+ many based on principled strategies", red "-- suboptimal by arbitrarily large amount in some environments."

## Slide 38 — Plan for Today

The slide 5 agenda again: "1. Meta-RL problem statement", "2. Black-box meta-RL methods", "Next time: Exploration + Learning to explore.", bracket "part of HW4", and "Lecture goals:" with "Understand the meta-RL problem statement & set-up" and "Understand the basics & challenges of black-box meta RL algorithms", coloured as on slide 5.

## Slide 39 — Next time

Two blocks. "Next time": "**Today**: meta-RL basics" and "**Friday**: exploration & meta-exploration". "Reminders": "Homework 3 due **Friday**" and "Project milestone due **next Friday**".
