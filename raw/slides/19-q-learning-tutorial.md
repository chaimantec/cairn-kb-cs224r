---
title: "Lecture 19 — CS 224R Tutorial: Review of Q-Learning (slide deck)"
lecture: 19
slides: 35
printed_numbers: "1 only (the title page)"
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/section_q_learning_tutorial.pdf
numbering: 1:1 with PDF pages. The deck prints a number only on its title page ("1", bottom centre); every other label is the page index
figure_audit: Transcribed by Claude Sonnet 5.5 (slides 1–17 and 18–35 by two agents). Six figure slides (4, 15, 19, 24, 27, 31) were checked by Claude Opus 5.5 against 1400px renders, and all six agree, including every cell of slide 15's four grids, slide 19's bar heights and slide 31's average errors. The grid builds on slides 8–14 were also checked arithmetically against the update on slide 7 with discount 0.9, and every number is consistent with it. Slide 20's equations were checked against a render.
images: 21 slides are rendered and shown under their headings (3, 4, 5, 7–15, 18, 19, 21, 22, 24, 27, 28, 30, 31). Not rendered: 1 (title page, with two game screenshots) and 2 (outline); 20, 26 and 29 (equation slides reproduced exactly here); 23 (a table reproduced here); 33 and 34 (pseudocode reproduced here); 6, 16, 17, 25, 32 and 35 (text only)
---

# Lecture 19 — CS 224R Tutorial: Review of Q-Learning: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025) Q-learning tutorial deck, transcribed from the PDF. It is a TA session, given by Anikait Singh, whose name the title slide prints; the course site lists it as the "Extra section on Q-learning" in week 3, and the catalog places it last. Slide 2 thanks Pieter Abbeel, David Silver, Aviral Kumar and Justin Fu. Cite these as "slide N". Diagrams, grids and plots are described in prose since the KB is read as text.

The recording is about 51 minutes and **ends on slide 31**, while Singh is still discussing critic ensembles (≈49:49). It follows the deck in order: the outline and MDP review (slides 1–5, ≈0:05–3:56), the Bellman equation and fitted Q iteration on the grid world (slides 6–12, ≈3:56–10:54), the optimal value, Q-value, policy and advantage and the suboptimal policy (slides 13–15, ≈10:54–17:47), the colab pointer (slide 16, ≈17:47), parametric value functions (slides 18–19, ≈17:47–21:39), the Monte Carlo, TD and n-step derivation, worked on the whiteboard with slide 20's equations (≈21:39–28:59), the backup diagrams, n-step returns and bias-variance table (slides 21–23, ≈28:59–32:05), stitching (slide 24, ≈32:05), semi-gradients and target networks (slide 26, ≈32:51–38:14), gradient clipping and the Huber loss (slide 27, ≈38:14–39:47), replay buffers (slide 28, ≈39:47–42:06) and overestimation (slide 29, ≈42:06–46:47). Singh then shows slide 31's figure (≈46:47) before slide 30's ensembles (≈47:33) and returns to slide 31 (≈48:19). **Slides 32–35 are not in the recording**: the walkthrough of the homework algorithms, which slide 2 lists as "Q-Learning/Actor Critic Algorithm Walkthrough (Preview of Homework 2)", with its DQN and SAC pseudocode, is only in this file. Slide 17's pause for questions passes without one in the recording; slide 25's is "Any questions about this?" (≈32:51).

Some students are audible in this recording: three questions are captioned (≈7:47, 10:09 and 12:26), and five more are inaudible and answered where they arise (≈19:20, 30:32, 37:28, and two at 41:19).

## Slide numbers vs PDF pages

The deck is 1:1 with its page index: PDF page N is slide N, for all 35 pages. Only the title page prints a number ("1", at bottom centre, read from the text layer); `slide_number_map.py`, which reads the corners, found none, and no other page carries one. Every label after slide 1 is therefore the page index rather than a printed number. Slides 3–4 and 7–12 are builds: 4 replaces 3's agent-environment loop with the grid world, and 7–12 step through the iterations $Q^{\ast}_ 1$ to $Q^{\ast}_ 6$.

Companion pages: [wiki page for this lecture](../../wiki/19-q-learning-tutorial.md) · [transcript](../transcripts/19-q-learning-tutorial.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title, outline of the tutorial |
| 3–5 | Review of MDPs: the formulation, the grid world and its goal, value, Q and advantage |
| 6–17 | Exact (tabular) MDPs: the Bellman equation and Q-value iteration, fitted Q iteration on the grid world ($Q^{\ast}_ 1$ to $Q^{\ast}_ 6$), the optimal value, Q-value, policy and advantage, $Q^{\ast}$ against $Q^{\pi}$, the colab, a pause for questions |
| 18–25 | Parametric Q-learning: value and Q networks, discrete and continuous Q-values, the regression derivation (Monte Carlo, TD, n-step), backup diagrams, the bias-variance table, stitching, a pause for questions |
| 26–31 | Practical details: semi-gradients and target networks, gradient clipping and the Huber loss, replay buffers, overestimation, critic ensembling, Q-learning against double Q-learning |
| 32–35 | Understanding the homework algorithms: DQN and soft actor-critic pseudocode (not in the recording), thanks |

## Notation in this deck

The deck mixes notations. Slide 5 and slide 14's formula use bold $\mathbf{s}$, $\mathbf{a}$, as Lecture 6 does; every other slide writes plain $s_t$, $a_t$, and slide 20 writes the conditioning state as a capital $S_t$ and the expectation as $E_\pi$. The Bellman and Q-value-iteration equations on slides 6–12 take the maximum under $\max_{a_t}$ while its argument is $a_{t+1}$ (text layer: "max | at | Q*(st+1, at+1)"). Slide 3 labels the loop's observation $o_t$ "(state $s_t$)" and writes the environment's transition as $P(s' \mid s, a)$ beside the list's $p(s_{t+1} \mid s_t, a_t)$. Slide 26's update rules call the weights $w$ and $w'$, while its equations above write $Q_\theta$ and $Q_{\text{target}}$. The grid plots on slides 7–15 number their cells from 0 (x = 0–3, y = 0–2), while slide 4's cartoon numbers them from 1 (columns 1–4, rows 1–3).

In the grid plots each cell carries five numbers in a cross, which Singh reads as one per action: the top number is the value of moving up, the left number of moving left, and so on (≈9:20–10:09), with the centre for staying, the fifth of the five actions (≈6:14). Below, each cell is listed as top; middle-row left, centre, right; bottom.

## Slide 1 — CS 224R Tutorial

Title slide. Title: "CS 224R Tutorial". Subtitle: "Review of Q-Learning". At lower left: "Anikait Singh". At bottom centre the printed number "1".

At right, two Atari game screenshots stacked vertically. The upper one shows a maze of blue corridors with salmon-pink walls, small dashes along the corridors, a small yellow character near the right edge, and a black strip below with the score "2060" in yellow and a small cherry icon. The lower one shows a black screen with the score "03875" in blue at the top and a pyramid of orange cubes (six along the bottom row, narrowing to one at the top), each cube with a yellow or blue top face, and a small orange creature at the bottom left corner.

## Slide 2 — Outline of Tutorial

- Review of Markov Decision Processes (MDP)
- Exact MDPs
  - Fitted Q Iteration
- Parametric Q-Learning
  - Bias Variance Tradeoff (TD vs MC)
- Practical Details
  - Replay Buffer, Overestimation, TD Gradients
- Q-Learning/Actor Critic Algorithm Walkthrough (Preview of Homework 2)

At lower right: "Many thanks to Pieter Abbeel, David Silver, Aviral Kumar, and Justin Fu!"

## Slide 3 — MDP Formulation

![Slide 3 — MDP Formulation](../images/19-q-learning-tutorial/slide-3.jpg)

Left, in a black-bordered box: "An MDP is defined by:"

- Set of states $s_t$
- Set of actions $a_t$
- Transition function $p(s_{t+1} \mid s_t, a_t)$
- Reward Function $r(s_t, a_t)$
- Start state $s_1$
- Discount Factor $\gamma$
- Horizon $T$

Right: an agent-environment loop diagram in black serif text. A rounded box "Agent" at the top with the label "based on a policy" above it, and a rounded box "Environment" at the bottom with the label "stochastic with $P(s' \mid s, a)$" below it. An arrow runs from the right side of Agent down the right edge into Environment, labelled "action $a_t$". An arrow runs from the left side of Environment up the left edge into Agent, labelled "observation $o_t$ (state $s_t$)". A vertical arrow runs from Environment up to Agent in the middle, labelled "reward $r_t$".

## Slide 4 — MDP Formulation

![Slide 4 — MDP Formulation](../images/19-q-learning-tutorial/slide-4.jpg)

Left, the same black-bordered box as the previous slide: "An MDP is defined by:" with the bullets Set of states $s_t$; Set of actions $a_t$; Transition function $p(s_{t+1} \mid s_t, a_t)$; Reward Function $r(s_t, a_t)$; Start state $s_1$; Discount Factor $\gamma$; Horizon $T$.

Right, top: a cartoon grid world drawn on brown stone, four columns wide and three rows tall, with column numbers 1 to 4 along the bottom and row numbers 1 (bottom) to 3 (top) down the left side. The word "START" is printed in the bottom-left cell (column 1, row 1). A blue gem labelled "+1" is in the top-right cell (column 4, row 3). An orange explosion labelled "-1" sits in the cell below it (column 4, row 2), on a darker reddish background. A grey stone block fills the cell at column 2, row 2. A small round robot with a red eye stands in column 3, row 1, with three green arrows around it pointing up, left and right.

Right, middle: "Goal:" followed by

$$\max_{\pi} \mathbb{E}\left[\sum_{t=1}^{T} \gamma^{t} r(s_t, a_t) \mid \pi\right]$$

Right, bottom: the label $\pi$ with a colon, followed by a cartoon of two robot hands holding a paper map of the same grid. The map shows green arrows in each cell. Top row, left to right: an arrow pointing right, an arrow pointing right, an arrow pointing right, and the blue gem. Middle row: an arrow pointing up, a hatched (blocked) cell, an arrow pointing up, and an orange skull-and-crossbones. Bottom row: an arrow pointing up, an arrow pointing right, an arrow pointing up, and an arrow pointing left.

## Slide 5 — Revisiting Some Useful Objects

![Slide 5 — Revisiting Some Useful Objects](../images/19-q-learning-tutorial/slide-5.jpg)

- *value function* $V^\pi(\mathbf{s})$ - future expected rewards starting at $\mathbf{s}$ and following $\pi$
- *Q-function* $Q^\pi(\mathbf{s}, \mathbf{a})$ - future expected rewards starting at $\mathbf{s}$, taking $\mathbf{a}$, then following $\pi$

Two diagrams side by side. Left: a blue dot labelled "s" (in blue) at the bottom, with five black wavy trajectory arrows fanning out upward to the right, labelled "run $\pi$" at the left. A thin red double-headed arrow running alongside, parallel to the trajectories, is labelled "expected sum of these rewards". To the right, the label $V^\pi(\mathbf{s})$ with the s in blue.

Right: the same blue dot "s" with a similar fan of trajectories, grey except for two drawn in dark green, with a short green arrow at the start. The label reads "take action **a**, then run $\pi$" with the a in green. A thin red double-headed arrow with green text "expected sum of these rewards" runs alongside. To the right, the label $Q^\pi(\mathbf{s}, \mathbf{a})$ with s in blue and a in green.

"Useful relation:"

$$V^\pi(\mathbf{s}) = \mathbb{E}_ {\mathbf{a} \sim \pi(\cdot \mid \mathbf{s})}\left[Q^\pi(\mathbf{s}, \mathbf{a})\right]$$

- *advantage* $A^\pi(\mathbf{s}, \mathbf{a})$ - how much better it is to take $\mathbf{a}$ than to follow policy $\pi$ at state $\mathbf{s}$

$$A^\pi(\mathbf{s}, \mathbf{a}) = Q^\pi(\mathbf{s}, \mathbf{a}) - V^\pi(\mathbf{s})$$

## Slide 6 — Tabular Learning - Fitted Q Iteration

"Q-Function $Q^\pi(\mathbf{s}, \mathbf{a})$ - future expected rewards starting at state $\mathbf{s}$, taking action $\mathbf{a}$, then following $\pi$"

"Bellman Equation"

$$Q^{\ast}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}(s_{t+1}, a_{t+1})\right)$$

"Q-Value Iteration"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

## Slide 7 — Fitted Q Iteration with the MDP

![Slide 7 — Fitted Q Iteration with the MDP](../images/19-q-learning-tutorial/slide-7.jpg)

"Discount = 0.9"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

Left: the cartoon grid world from slide 4 (START at column 1 row 1, blue gem "+1" at column 4 row 3, "-1" explosion at column 4 row 2, grey stone at column 2 row 2, robot with three green arrows at column 3 row 1; column numbers 1 to 4 below, row numbers 1 to 3 at left).

Right: a plot titled $Q_1^{\ast}(s,a)$. Both axes have gridlines; the horizontal axis is labelled -1 to 4 and the vertical axis -1 to 3. A pale-yellow rectangle covers the grid from x = -0.5 to 3.5 and y = -0.5 to 2.5. It is divided into 12 cells, x = 0, 1, 2, 3 across and y = 0, 1, 2 up. Each cell carries five numbers printed in a cross: one at the top, three in a row across the middle (left, centre, right), and one at the bottom. Cells are listed below as top; middle-row left, centre, right; bottom. The cell at x = 3, y = 2 is filled dark blue and the cell at x = 3, y = 1 is filled dark red; all other cells are pale yellow. The wall cell (x = 1, y = 1) carries the same numbers as the others.

- Row y = 2: x = 0: 0.0; 0.0, 0.0, 0.0; 0.0. x = 1: 0.0; 0.0, 0.0, 0.0; 0.0. x = 2: 0.0; 0.0, 0.0, 0.0; 0.0. x = 3 (dark blue): 1.0; 1.0, 1.0, 1.0; 1.0.
- Row y = 1: x = 0, 1 and 2: every number 0.0. x = 3 (dark red): -1.0; -1.0, -1.0, -1.0; -1.0.
- Row y = 0: every number 0.0 in all four cells.

## Slide 8 — Fitted Q Iteration with the MDP

![Slide 8 — Fitted Q Iteration with the MDP](../images/19-q-learning-tutorial/slide-8.jpg)

"Discount = 0.9"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

Left: the same cartoon grid world as slide 7. Right: the plot, now titled $Q_2^{\ast}(s,a)$, with the same layout (cells x = 0 to 3, y = 0 to 2; five numbers per cell: top; middle left, centre, right; bottom).

- Row y = 2: x = 0: all 0.0. x = 1: all 0.0. x = 2: 0.0; 0.0, 0.0, 0.9; 0.0 (the middle-right number 0.9 sits in a medium-blue triangle at the cell's right). x = 3 (dark blue): all 1.0.
- Row y = 1: x = 0 and x = 1: all 0.0. x = 2: 0.0; 0.0, 0.0, -0.9; 0.0 (the -0.9 sits in a red triangle at the right). x = 3 (dark red): all -1.0.
- Row y = 0: x = 0, 1, 2: all 0.0. x = 3: -0.9; 0.0, 0.0, 0.0; 0.0 (the top number -0.9 sits in a red triangle).

Changed from the previous slide: the plot is Q_2 rather than Q_1; the new non-zero entries are 0.9 at (x = 2, y = 2) right, -0.9 at (x = 2, y = 1) right and -0.9 at (x = 3, y = 0) top.

## Slide 9 — Fitted Q Iteration with the MDP

![Slide 9 — Fitted Q Iteration with the MDP](../images/19-q-learning-tutorial/slide-9.jpg)

"Discount = 0.9"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

Left: the same cartoon grid world. Right: the plot titled $Q_3^{\ast}(s,a)$, same layout.

- Row y = 2: x = 0: all 0.0. x = 1: 0.0; 0.0, 0.0, 0.8; 0.0 (the 0.8 sits in a blue triangle at the right). x = 2: 0.8; 0.0, 0.8, 0.9; 0.0 (blue triangles at the top, centre and right). x = 3 (dark blue): all 1.0.
- Row y = 1: x = 0 and x = 1: all 0.0. x = 2: 0.8; 0.0, 0.0, -0.9; 0.0 (blue triangle at the top, red at the right). x = 3 (dark red): all -1.0.
- Row y = 0: x = 0, 1, 2: all 0.0. x = 3: -0.9; 0.0, 0.0, 0.0; 0.0.

Changed from the previous slide: plot is Q_3; new entries 0.8 at (x = 1, y = 2) right, 0.8 at (x = 2, y = 2) top and centre, and 0.8 at (x = 2, y = 1) top.

## Slide 10 — Fitted Q Iteration with the MDP

![Slide 10 — Fitted Q Iteration with the MDP](../images/19-q-learning-tutorial/slide-10.jpg)

"Discount = 0.9"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

Left: the same cartoon grid world. Right: the plot titled $Q_4^{\ast}(s,a)$, same layout (five numbers per cell: top; middle left, centre, right; bottom).

- Row y = 2: x = 0: 0.0; 0.0, 0.0, 0.7; 0.0. x = 1: 0.7; 0.0, 0.7, 0.8; 0.7. x = 2: 0.8; 0.7, 0.8, 0.9; 0.7. x = 3 (dark blue): all 1.0.
- Row y = 1: x = 0: all 0.0. x = 1: all 0.0. x = 2: 0.8; 0.7, 0.7, -0.9; 0.0. x = 3 (dark red): all -1.0.
- Row y = 0: x = 0 and x = 1: all 0.0. x = 2: 0.7; 0.0, 0.0, 0.0; 0.0. x = 3: -0.9; 0.0, 0.0, 0.0; 0.0.

Changed from the previous slide: the plot is Q_4; 0.7 now appears at (x = 0, y = 2) right, at (x = 1, y = 2) centre, top and bottom, at (x = 2, y = 2) bottom, at (x = 2, y = 1) left and centre, and at (x = 2, y = 0) top; 0.8 appears at (x = 2, y = 1) top and at (x = 1, y = 2) right.

## Slide 11 — Fitted Q Iteration with the MDP

![Slide 11 — Fitted Q Iteration with the MDP](../images/19-q-learning-tutorial/slide-11.jpg)

"Discount = 0.9"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

Left: the same cartoon grid world. Right: the plot titled $Q_5^{\ast}(s,a)$, same layout.

- Row y = 2: x = 0: 0.7; 0.7, 0.7, 0.7; 0.0. x = 1: 0.7; 0.7, 0.7, 0.8; 0.7. x = 2: 0.8; 0.7, 0.8, 0.9; 0.7. x = 3 (dark blue): all 1.0.
- Row y = 1: x = 0: 0.7; 0.0, 0.0, 0.0; 0.0. x = 1: all 0.0. x = 2: 0.8; 0.7, 0.7, -0.9; 0.7. x = 3 (dark red): all -1.0.
- Row y = 0: x = 0: all 0.0. x = 1: 0.0; 0.0, 0.0, 0.7; 0.0. x = 2: 0.7; 0.0, 0.7, 0.0; 0.7. x = 3: -0.9; 0.7, 0.0, 0.0; 0.0.

Changed from the previous slide: the plot is Q_5; new 0.7 values appear in row y = 2 at x = 0 (top, left, centre), at (x = 0, y = 1) top, at (x = 1, y = 0) right, at (x = 2, y = 1) bottom, at (x = 2, y = 0) top, centre and bottom, and at (x = 3, y = 0) left.

## Slide 12 — Fitted Q Iteration with the MDP

![Slide 12 — Fitted Q Iteration with the MDP](../images/19-q-learning-tutorial/slide-12.jpg)

"Discount = 0.9"

$$Q^{\ast}_ {k+1}(s,a) = \sum_{s_{t+1}} p(s_{t+1} \mid s_t, a_t)\left(r(s_t, a_t) + \gamma \max_{a_t} Q^{\ast}_ {k}(s_{t+1}, a_{t+1})\right)$$

Left: the same cartoon grid world. Right: the plot titled $Q_6^{\ast}(s,a)$, same layout. Only the wall cell (x = 1, y = 1) and parts of the cells at x = 0 remain pale yellow.

- Row y = 2: x = 0: 0.7; 0.7, 0.7, 0.7; 0.6. x = 1: 0.7; 0.7, 0.7, 0.8; 0.7. x = 2: 0.8; 0.7, 0.8, 0.9; 0.7. x = 3 (dark blue): all 1.0.
- Row y = 1: x = 0: 0.7; 0.6, 0.6, 0.6; 0.0. x = 1 (pale yellow): all 0.0. x = 2: 0.8; 0.7, 0.7, -0.9; 0.7. x = 3 (dark red): all -1.0.
- Row y = 0: x = 0: 0.6; 0.0, 0.0, 0.6; 0.0. x = 1: 0.6; 0.0, 0.6, 0.7; 0.6. x = 2: 0.7; 0.6, 0.7, 0.6; 0.7. x = 3: -0.9; 0.7, 0.6, 0.6; 0.6.

Changed from the previous slide: the plot is Q_6; new 0.6 values appear at (x = 0, y = 2) bottom, in (x = 0, y = 1) left, centre and right, and in row y = 0 at x = 0 (top, right), x = 1 (top, centre, bottom) and x = 2 (left, right), x = 3 (centre, right, bottom); the 0.7 values at (x = 1, y = 0) right, (x = 2, y = 0) centre and (x = 3, y = 0) left are unchanged.

## Slide 13 — Optimal Value, Q-Value, Policy

![Slide 13 — Optimal Value, Q-Value, Policy](../images/19-q-learning-tutorial/slide-13.jpg)

Three plots side by side, each with horizontal axis -1 to 4 and vertical axis -1 to 3, gridlines, and the 4 by 3 cell grid (x = 0 to 3, y = 0 to 2).

Left plot, titled $V^{\ast}(s)$: one number per cell, each cell shaded.

- Row y = 2: 0.7, 0.8, 0.9, 1.0 (x = 0 to 3; shades of blue, darkest at x = 3).
- Row y = 1: 0.7, 0.0 (pale yellow, the wall cell), 0.8, -1.0 (dark red).
- Row y = 0: 0.6, 0.7, 0.7, 0.7 (lighter blues).

Middle plot, titled $Q^{\ast}(s,a)$: five numbers per cell (top; middle left, centre, right; bottom).

- Row y = 2: x = 0: 0.7; 0.7, 0.7, 0.7; 0.6. x = 1: 0.7; 0.7, 0.7, 0.8; 0.7. x = 2: 0.8; 0.7, 0.8, 0.9; 0.7. x = 3: all 1.0.
- Row y = 1: x = 0: 0.7; 0.6, 0.6, 0.6; 0.5. x = 1 (pale yellow): all 0.0. x = 2: 0.8; 0.7, 0.7, -0.9; 0.7. x = 3 (dark red): all -1.0.
- Row y = 0: x = 0: 0.6; 0.5, 0.5, 0.6; 0.5. x = 1: 0.6; 0.5, 0.6, 0.7; 0.6. x = 2: 0.7; 0.6, 0.7, 0.6; 0.7. x = 3: -0.9; 0.7, 0.6, 0.6; 0.6.

Right plot, titled $\pi^{\ast}(s,a)$: five numbers per cell in the same layout, on a dark-red background, with a dark-blue triangle (or small square at the centre) behind each 1.0.

- Row y = 2: x = 0: 0.0; 0.0, 0.0, 1.0; 0.0. x = 1: 0.0; 0.0, 0.0, 1.0; 0.0. x = 2: 0.0; 0.0, 0.0, 1.0; 0.0. x = 3: 0.0; 0.0, 1.0, 0.0; 0.0.
- Row y = 1: x = 0: 1.0; 0.0, 0.0, 0.0; 0.0. x = 1: 0.0; 0.0, 1.0, 0.0; 0.0. x = 2: 1.0; 0.0, 0.0, 0.0; 0.0. x = 3: 0.0; 0.0, 1.0, 0.0; 0.0.
- Row y = 0: x = 0: 1.0; 0.0, 0.0, 0.0; 0.0. x = 1: 0.0; 0.0, 0.0, 1.0; 0.0. x = 2: 1.0; 0.0, 0.0, 0.0; 0.0. x = 3: 0.0; 1.0, 0.0, 0.0; 0.0.

## Slide 14 — Advantage (Optimal Policy)

![Slide 14 — Advantage (Optimal Policy)](../images/19-q-learning-tutorial/slide-14.jpg)

Three plots side by side, as on the previous slide. The left plot is titled $V^{\ast}(s)$ and the middle plot $Q^{\ast}(s,a)$; both carry exactly the numbers listed on the previous slide. The right plot is titled $A^{\ast}(s,a)$, with five numbers per cell (top; middle left, centre, right; bottom), mostly dark blue with a red triangle at (x = 2, y = 1) right and a bright-red triangle at (x = 3, y = 0) top.

- Row y = 2: x = 0: -0.1; -0.1, -0.1, 0.0; -0.1. x = 1: -0.1; -0.2, -0.1, 0.0; -0.1. x = 2: -0.1; -0.2, -0.1, 0.0; -0.2. x = 3: 0.0; 0.0, 0.0, 0.0; 0.0.
- Row y = 1: x = 0: 0.0; -0.1, -0.1, -0.1; -0.1. x = 1: 0.0; 0.0, 0.0, 0.0; 0.0. x = 2: 0.0; -0.1, -0.1, -1.7; -0.2. x = 3: 0.0; 0.0, 0.0, 0.0; 0.0.
- Row y = 0: x = 0: 0.0; -0.1, -0.1, 0.0; -0.1. x = 1: -0.1; -0.1, -0.1, 0.0; -0.1. x = 2: 0.0; -0.1, -0.1, -0.1; -0.1. x = 3: -1.6; 0.0, -0.1, -0.1; -0.1.

At the bottom:

$$A^{\pi}(\mathbf{s}_ t, \mathbf{a}_ t) = Q^{\pi}(\mathbf{s}_ t, \mathbf{a}_ t) - V^{\pi}(\mathbf{s}_ t)\text{: how much better } \mathbf{a}_ t \text{ is}$$

(The printed sentence ends after "is" and runs no further.)

## Slide 15 — $Q^{\ast}$ vs $Q^{\pi}$

![Slide 15 — Q* vs Q-pi](../images/19-q-learning-tutorial/slide-15.jpg)

Title printed as "Q* vs Q^π" in italic serif type with a bold "vs". Left labels: "Optimal Policy (Deterministic for MDP)" beside the top row of plots, and "Suboptimal Policy" beside the bottom row. Four plots in a 2 by 2 arrangement, each with axes -1 to 4 and -1 to 3 and the 4 by 3 grid of cells (five numbers per cell: top; middle left, centre, right; bottom).

Top left, titled $Q^{\ast}(s,a)$: the same numbers as the middle plot of slide 13.

- Row y = 2: x = 0: 0.7; 0.7, 0.7, 0.7; 0.6. x = 1: 0.7; 0.7, 0.7, 0.8; 0.7. x = 2: 0.8; 0.7, 0.8, 0.9; 0.7. x = 3: all 1.0.
- Row y = 1: x = 0: 0.7; 0.6, 0.6, 0.6; 0.5. x = 1: all 0.0. x = 2: 0.8; 0.7, 0.7, -0.9; 0.7. x = 3: all -1.0.
- Row y = 0: x = 0: 0.6; 0.5, 0.5, 0.6; 0.5. x = 1: 0.6; 0.5, 0.6, 0.7; 0.6. x = 2: 0.7; 0.6, 0.7, 0.6; 0.7. x = 3: -0.9; 0.7, 0.6, 0.6; 0.6.

Top right, titled $\pi^{\ast}(s,a)$: the same numbers as the right plot of slide 13 (dark red with dark-blue triangles behind each 1.0).

- Row y = 2: x = 0, 1, 2: 0.0; 0.0, 0.0, 1.0; 0.0. x = 3: 0.0; 0.0, 1.0, 0.0; 0.0.
- Row y = 1: x = 0: 1.0; 0.0, 0.0, 0.0; 0.0. x = 1: 0.0; 0.0, 1.0, 0.0; 0.0. x = 2: 1.0; 0.0, 0.0, 0.0; 0.0. x = 3: 0.0; 0.0, 1.0, 0.0; 0.0.
- Row y = 0: x = 0: 1.0; 0.0, 0.0, 0.0; 0.0. x = 1: 0.0; 0.0, 0.0, 1.0; 0.0. x = 2: 1.0; 0.0, 0.0, 0.0; 0.0. x = 3: 0.0; 1.0, 0.0, 0.0; 0.0.

Bottom left, titled $Q^{\pi}(s,a)$ (mostly pale yellow with light-blue triangles):

- Row y = 2: x = 0: 0.0; 0.0, 0.0, 0.3; 0.0. x = 1: 0.0; 0.0, 0.0, 0.4; 0.0. x = 2: 0.0; 0.0, 0.0, 0.6; 0.1. x = 3 (dark blue): all 1.0.
- Row y = 1: x = 0: 0.3; 0.0, 0.0, 0.0; 0.0. x = 1: all 0.0. x = 2: 0.4; 0.0, 0.0, -0.1; 0.0. x = 3 (dark red): all -1.0.
- Row y = 0: x = 0: 0.2; 0.0, 0.0, 0.0; 0.0. x = 1: 0.0; 0.0, 0.0, 0.2; 0.0. x = 2: 0.3; 0.0, 0.0, 0.0; 0.0. x = 3: -0.2; 0.2, 0.0, 0.0; 0.0.

Bottom right, titled $\pi(s,a)$ (cells in shades of red, orange and blue):

- Row y = 2: x = 0: 0.1; 0.0, 0.0, 0.9; 0.0. x = 1: 0.0; 0.1, 0.0, 0.8; 0.0. x = 2: 0.0; 0.0, 0.0, 0.6; 0.3. x = 3: 0.1; 0.1, 0.8, 0.0; 0.0.
- Row y = 1: x = 0: 0.8; 0.1, 0.0, 0.0; 0.1. x = 1: 0.0; 0.1, 0.9, 0.0; 0.0. x = 2: 0.7; 0.0, 0.1, 0.1; 0.1. x = 3: 0.1; 0.0, 0.8, 0.0; 0.0.
- Row y = 0: x = 0: 0.8; 0.1, 0.0, 0.0; 0.1. x = 1: 0.1; 0.1, 0.0, 0.7; 0.1. x = 2: 0.9; 0.0, 0.0, 0.0; 0.1. x = 3: 0.3; 0.7, 0.0, 0.0; 0.0.

## Slide 16 — [text only, no title]

"To play more with the Tabular Environment, checkout the colab" with the word "colab" underlined (a hyperlink).

Singh points to this colab as "the collab in the PDF that I had shared" (≈17:47, as captioned). The page prints no URL; the link target is not recorded in this file.

## Slide 17 — [text only, no title]

"Pause for Questions."

## Slide 18 — Limitations of Tabular Environment

![Slide 18 — Limitations of Tabular Environment](../images/19-q-learning-tutorial/slide-18.jpg)

Subtitle: "Moving towards parametric value functions".

Bullets: "A tabular value function intractable as state/action space grows!"; "Learning a parametric value function can also allow for generalization"; sub-bullet "Related states have similar q-values (smoothness)"; "Can use regression to learn value functions with a neural network".

Right side: a diagram of three trapezoid "networks" in a row, each with an input label beneath and an output label above.

- Left, in blue, containing $V^\pi_\phi$. Above it "(scalar)" and $V^\pi_\phi(s_t)$; below it the input $s_t$ and the caption "Value".
- Middle, in green, containing $Q^\pi_\phi$. Above it "(scalar)" and $Q^\pi_\phi(s_t, a_t)$; below it the two inputs $s_t$ and $a_t$ and the caption "Continuous Q-Value".
- Right, in orange, containing $Q^\pi_\phi$. Above it "(vector)" and $Q^\pi_\phi(s_t, a_1) \ldots Q^\pi_\phi(s_t, a_m)$; below it the input $s_t$ and the caption "Discrete Q-Value".

## Slide 19 — Discrete vs Continuous Q-values

![Slide 19 — Discrete vs Continuous Q-values](../images/19-q-learning-tutorial/slide-19.jpg)

Text, as dash bullets: "Different formulations for continuous and discrete action spaces:"; sub-bullet "In **continuous (a)**, typically condition on an action and output a scalar, where you sample actions from your policy for an expectation calculation"; sub-bullet "In **discrete (b)**, typically learn a value for each bin (output) and can enumerate over all actions for an expectation" (the words "continuous (a)" and "discrete (b)" are in bold).

Right side, two figures stacked.

Top (panel a, not lettered on the slide): a strip of five robot camera frames showing a gripper opening a cabinet or oven door in a kitchen, the door swinging progressively open from left to right. Below the strip, a line chart of dots joined by a dashed line. Y-axis $Q(s, a_t)$ (rotated), ticks 0, -300, -600, -900, -1200, -1500; x-axis "Timestep", ticks 0, 10, 20, 30, 40, 50, 60. One series (blue dots): about -1180 at timestep 0, about -1200 near timestep 5, about -1050 at 15, flat at about -1020 from 15 to 20, jumping to about -870 at 22, rising to about -700 at 27, about -520 at 33, about -300 at 36, about -330 at 39, about -150 at 42, about -190 at 46, about 0 at 49-50, about -150 at 52, and about -30 at 59.

Bottom, labelled "b": four frames numbered with black circles 1 to 4, each a brown playing field with a small orange bar at the left, a small green bar at the right and a white ball with a dashed trajectory line; a score strip at the top of each reads an orange "0" and a green "2". Below each frame is a three-bar chart, y-axis "Action-Values (Q)", ticks -1, -0.5, 0, 0.5, 1, with a legend at the right: light-blue "NO-OP", purple "UP", pink "DOWN". Three series (bars) per panel:

- Frame 1: NO-OP about 0.75, UP about 0.75, DOWN about 0.8.
- Frame 2: NO-OP about -0.9, UP about 0.7, DOWN about -0.9.
- Frame 3: NO-OP about -1, UP about 0.8, DOWN about -1.
- Frame 4: NO-OP, UP and DOWN all about 1.

## Slide 20 — Q-Learning as a Regression Problem

Subtitle: "Whiteboard".

The slide marks a whiteboard segment. Singh works this derivation on the board (≈21:39–28:59): the definition, the Monte Carlo sum, the expansion and the TD form line by line, then the n-step form, read from an iPad "so I don't make a notation mistake" (≈27:49). The board is not visible in the recording; the printed equations are the derivation's lines.

Printed equations, each with a label at the right:

$$V^\pi(s) = E_\pi \left[ \mathcal{R}_ t \mid S_t = s \right] \qquad \text{(Definition)}$$

$$= E_\pi \left[ \sum_{k=0}^{\infty} \gamma^k \thinspace r(s_{t+k}, a_{t+k}) \mid S_t = s \right] \qquad \text{(Monte–Carlo Rollout, MC)}$$

$$= E_\pi \left[ r(s_t, a_t) + \gamma \sum_{k=0}^{\infty} \gamma^k \thinspace r(s_{t+k+1}, a_{t+k+1}) \mid S_t = s \right] \qquad \text{(Expansion)}$$

$$= E_\pi \left[ r(s_t, a_t) + \gamma V^\pi(s_{t+1}) \mid S_t = s \right] \qquad \text{(Dynamic Programming, TD)}$$

$$= E_\pi \left[ r(s_t, a_t) + \gamma \thinspace r(s_{t+1}, a_{t+1}) + \cdots + \gamma^{k+1} V^\pi(s_{t+k+1}) \mid S_t = s \right] \qquad \text{(N-Step)}$$

Below, centred: "Bias vs Variance Tradeoff!"

## Slide 21 — MC vs TD Visualization

![Slide 21 — MC vs TD Visualization](../images/19-q-learning-tutorial/slide-21.jpg)

Two backup-tree diagrams side by side, each a tree of open circles (states) and small filled black dots (actions) branching downward from a root state labelled $S_t$, with green squares marked "T" for terminal states and dashed lines trailing off at the bottom where the tree continues.

- Left, caption "Monte-Carlo Rollouts" and, beneath it, "(Regression)": a single path from the root $S_t$ down through one action, one state, one action, one state, one action, ending at a green "T" square at the bottom, is highlighted with a thick red band. The rest of the tree is unhighlighted.
- Right, caption "TD Backup" and, beneath it, "(Dynamic Programming)": a red band covers only the top of the tree, from the root $S_t$ through its actions to the states one step below. One action is labelled "a", the edge below it "r", and the state it reaches "s'" (at the right end of the band). Everything below that first level is unhighlighted.

## Slide 22 — N-Step Returns

![Slide 22 — N-Step Returns](../images/19-q-learning-tutorial/slide-22.jpg)

Text: "Let TD target look $n$ steps into the future".

A diagram of five vertical backup chains side by side, each starting with an open circle at the top and alternating filled black dots (actions) and open circles (states) downward. Underlined column labels, left to right:

- "TD (1-step)": one filled dot and two open circles in total (circle, dot, circle).
- "2-step": circle, dot, circle, dot, circle.
- "3-step": circle, dot, circle, dot, circle, dot, circle.
- "n-step": a longer chain of circle, dot, circle, dot, circle, dot, then a vertical dotted ellipsis, a dot, and a final open circle.
- "Monte Carlo": the longest chain, with a vertical dotted ellipsis near the bottom and a grey filled square at the very bottom.

Horizontal dotted ellipses ("• • •") sit between the 3-step and n-step chains and between the n-step and Monte Carlo chains.

## Slide 23 — Bias vs Variance Tradeoff

Subtitle: "TD, MC, N-Step Return".

| Method | Bootstrap Depth | Bias | Variance |
| --- | --- | --- | --- |
| TD | 1 Steps | High | Low |
| N-Step | N Steps | Medium | Medium |
| Monte Carlo | Full Episode | Low | High |

The header row is orange and the method column is dark yellow.

## Slide 24 — Why Dynamic Programming?

![Slide 24 — Why Dynamic Programming?](../images/19-q-learning-tutorial/slide-24.jpg)

Subtitle: "Aside on Stitching".

Two maze pictures side by side, each an orange maze with black walls. Bottom-right footer text: "Should I Use Offline RL or Imitation Learning?".

- Left, heading "Our maze navigation task". The maze has labelled points A (bottom left, a blue dot marking the start), B (top right, a purple square), C (top middle), D (a yellow square, the goal, in the middle-left) and E (middle). A red arrow path runs from A up and along to the right, then up and right to B. A white arrow path runs from C down through E to the bottom and left, then up to D. A grey legend box beneath reads: blue dot "Start (A)", yellow square "Goal (D)", red arrow "trajectory from A → B", white arrow "trajectory from C → D".
- Right, heading "“Stitching” enables finding optimal behavior". The same maze with the blue start dot at A and the yellow goal D; a single white arrow path runs from A up, across to the right, down along the right side, then left and up to D. A grey box beneath reads "Trajectory found by stitching segments of trajectories".

## Slide 25 — [Pause for Questions]

Centred text only: "Pause for Questions."

In the recording: "Any questions about this?" (≈32:51), with no question audible.

## Slide 26 — Practical Implementation Detail

Subtitle: "Semi-Gradient and Target Network".

"Going back to our formulation:"

$$Q(s, a) \leftarrow r(s, a) + \gamma \max_{\hat{a}} Q(s', \hat{a})$$

"Semi-Gradient"

$$Q_\theta(s, a) \leftarrow r(s, a) + \gamma \max_{\hat{a}} \text{stopgrad}\left( Q_\theta(s', \hat{a}) \right)$$

"Target Network"

$$Q_\theta(s, a) \leftarrow r(s, a) + \gamma \max_{\hat{a}} \text{stopgrad}\left( Q_{\text{target}}(s', \hat{a}) \right)$$

"Update Types"

On the left: $w' \leftarrow w$ when $\text{mod}\thinspace(n, N) = 0$, captioned "Hard Target Update".

On the right: $w' \leftarrow \tau \cdot w + (1 - \tau) \cdot w'$, captioned "Soft Target Update (Polyak)".

## Slide 27 — Practical Implementation Details

![Slide 27 — Practical Implementation Details](../images/19-q-learning-tutorial/slide-27.jpg)

Subtitle: "Dealing with TD Gradients".

Two figures side by side, each with a caption beneath.

- Left, caption "Gradient Clipping": two contour plots of an elongated, tilted loss surface, drawn as nested black closed curves with a small red cross near the centre-right marking the minimum. The left plot is titled "Without Gradient Clipping": a red arrow path starting at the lower left zigzags in large jumps, overshooting up and down across the contours and landing far from the minimum. The right plot is titled "With Gradient Clipping": a red arrow path from the same start zigzags in progressively smaller steps and ends at the red cross.
- Right, caption "Huber Loss": a line chart with no axis labels. X-axis ticks -4, -3, -2, -1, 0, 1, 2, 3, 4; y-axis ticks 0 to 8 in steps of 1. Two series. A blue curve is a parabola with its minimum of 0 at x = 0, about 0.5 at x = ±1, about 2 at ±2, about 4.5 at ±3 and 8 at ±4. A green curve coincides with the blue one near zero (to about |x| = 1), then continues as straight lines, about 3.5 at x = ±4.

## Slide 28 — Practical Implementation Details

![Slide 28 — Practical Implementation Details](../images/19-q-learning-tutorial/slide-28.jpg)

Subtitle: "Replay Buffers".

Bullets (bold words as printed):

- "Replay Buffers are **memory structures** that store past experiences encountered by the agent, allowing the **experiences to be reused** instead of discarded after a single update"
- "They **break temporal correlations** between consecutive training samples and prevents recency bias"
- "**Improves sample efficiency** in Q-learning algorithms, allowing an agent to reuse its experience effectively"

Right side: a black-and-white drawing of a database cylinder divided into horizontal slabs. From top to bottom the slabs read "<s_1, a_1, r_2, s_2>, 1", then "<s_2, a_2, r_3, s_3>, 2", then an empty slab with four vertical dashed lines (an ellipsis), then "<s_t, a_t, r_{t+1}, s_{t+1}>P" (the subscripts as drawn: $s_1, a_1, r_2, s_2$; $s_2, a_2, r_3, s_3$; $s_t, a_t, r_{t+1}, s_{t+1}$).

## Slide 29 — Q-Learning Overestimation

"Q-Learning is typically formulated as follows:"

$$Q(s, a) \leftarrow r(s, a) + \gamma \max_{\hat{a}} Q(s', \hat{a})$$

"When parameterized by a function approximator like a NN, there is some inherent noise in the q-value estimates due to modeling errors"

$$Q^{\text{approx}}(s', \hat{a}) = Q^{\text{target}}(s', \hat{a}) + Y^{\hat{a}}_ {s'}$$

"Due to the noise on the RHS ($Y^{\hat{a}}_ {s'}$), there results in error on the LHS, expressed as:"

$$Z_s := \gamma \left( \max_{\hat{a}} Q^{\text{approx}}(s', \hat{a}) - \max_{\hat{a}} Q^{\text{target}}(s', \hat{a}) \right)$$

"Even with zero mean ($Y^{\hat{a}}_ {s'}$), $Z_s$ may have positive mean:"

$$\mathbb{E}\left[ Y^{\hat{a}}_ {s'} \right] = 0 \thinspace \forall \hat{a} \implies \mathbb{E}\left[ Z_s \right] \gt 0 \quad \text{(often)}$$

Footer, bottom right: "Issues in Using Function Approximation for Reinforcement Learning".

Singh introduces this as "a derivation from using function approximation from RL" (≈42:52, as captioned); the slide's footer is the title it names. In the recording the noise term is called "this Y parameter" (≈43:40).

## Slide 30 — Practical Implementation Detail

![Slide 30 — Practical Implementation Detail](../images/19-q-learning-tutorial/slide-30.jpg)

Subtitle: "Critic Ensembling".

Bullets (bold words as printed):

- "One approach to tackle overestimation is to measure **epistemic uncertainty** with an ensemble of networks"
- "Since different model initializations may have **different modeling errors**, this can lead to ensembling with a **min reduction** to be effective"
- "Has been even shown to be effective in **offline RL** where distribution shift is a larger concern (Sac-N and EDAC)"

Right side: a diagram. A box "INPUT" at the top has three arrows down to three boxes side by side, labelled "Neural Network #1", "Neural Network #2" and "Neural Network #3". Each box holds the same small fully connected network drawn as circles with arrows between layers: 3 input circles, two hidden layers of 4 circles each, and 2 output circles. An arrow from the bottom of each box points into a dark grey box "Ensemble", and one arrow from "Ensemble" points down to a box "OUTPUT".

## Slide 31 — Q-Learning vs Double Q-Learning

![Slide 31 — Q-Learning vs Double Q-Learning](../images/19-q-learning-tutorial/slide-31.jpg)

A reproduced paper figure with three rows and three columns of plots, plus an "Average error" column at the right. All plots share x-axis label "state" (under the bottom row), ticks -6, -4, -2, 0, 2, 4, 6. Column titles: "True value and an estimate", "All estimates and max", "Bias as function of state", "Average error". A dashed grey horizontal line marks zero in each plot.

Left column (per row: a thick purple curve labelled $Q_\ast(s, a)$ for the true value, a green curve labelled $Q_t(s, a)$ for the estimate, and green dots on the sampled states; two curves and the dots per panel). The y-axis ticks are 2, 0, -2 in row 1, 2, 0 in row 2 and 4, 2, 0 in row 3.

- Row 1: the true value is a sine-like wave (about 1 at state -5 and 1.5, about -1 at about -2.5 and 4.5); the green estimate follows it closely through the green dots.
- Row 2: the true value is flat at 0 with a single peak of about 2 at state 0; the green estimate is a smoother, lower bump (about 1.5 at 0) with dips below zero at about -3 and 3.
- Row 3: the same single-peak true value (about 2 at 0); the green estimate is a high-degree fit that shoots off the chart (above 4) near states -4.5 and oscillates around zero elsewhere.

Middle column (green curves for all the estimates, and a dashed black curve labelled $\max_a Q_t(s, a)$ for their maximum; y-axis ticks 2, 0, -2 in row 1, 2, 0 in row 2, 4, 2, 0 in row 3). In every row the dashed black maximum lies above the true-value shape almost everywhere: row 1 a wave with extra peaks at the edges, row 2 a broad arch peaking at about 1.5 with a dip to 0 at about -3 and 3, and row 3 a W-shaped curve rising steeply (off the top of the chart) near the edges at about -5 and 5.

Right column (an orange curve labelled $\max_a Q_t(s, a) - \max_a Q_\ast(s, a)$ and a blue curve labelled "Double-Q estimate"; two series per panel; y-axis ticks 1, 0, -1 in rows 1 and 2, and 4, 2, 0 in row 3). In each row the orange curve is well above zero and the blue curve stays near zero. Average error column, to the right, in orange (for the orange curve) and in blue (for the blue curve):

| Row | Orange (max difference) | Blue (Double-Q estimate) |
| --- | --- | --- |
| 1 | +0.61 | −0.02 |
| 2 | +0.47 | +0.02 |
| 3 | +3.35 | −0.02 |

Printed caption: "Figure 2: Illustration of overestimations during learning. In each state (x-axis), there are 10 actions. The **left column** shows the true values $V_\ast(s)$ (purple line). All true action values are defined by $Q_\ast(s, a) = V_\ast(s)$. The green line shows estimated values $Q(s, a)$ for one action as a function of state, fitted to the true value at several sampled states (green dots). The **middle column** plots show all the estimated values (green), and the maximum of these values (dashed black). The maximum is higher than the true value (purple, left plot) almost everywhere. The **right column** plots shows the difference in orange. The blue line in the right plots is the estimate used by Double Q-learning with a second set of samples for each state. The blue line is much closer to zero, indicating less bias. The three **rows** correspond to different true functions (left, purple) or capacities of the fitted function (left, green). (Details in the text)"

In the recording Singh shows this figure first at ≈46:47, before slide 30, and returns to it at ≈48:19, saying it is "from the double Q-learning paper" (≈49:04); the slide itself prints no source. Singh calls the curve for standard Q-learning "the red one" (≈49:04); on the slide it is the orange curve.

## Slide 32 — [Understanding HW Algorithms]

Centred text only: "Understanding HW Algorithms".

Slides 32–35 are not in the recording, which ends on slide 31 (≈49:49).

## Slide 33 — Q-Learning : DQN

Subtitle: "Whiteboard".

The slide marks a whiteboard segment, which is not in the recording: the recording ends on slide 31.

A pseudocode box titled "Algorithm 1: DQN":

**Input:** Replay buffer $\mathcal{D}$, Q-network $Q_\theta$, target network $Q_{\theta'}$ with $\theta' \leftarrow \theta$, discount factor $\gamma$, batch size $N$, learning rate $\alpha$, target update frequency $C$, exploration schedule $\epsilon$

**for** each interaction step $t = 1, 2, \ldots$ **do** (the body below is inside the loop, marked by a vertical bar)

- Observe state $s_t$
- Choose action $a_t$ using $\epsilon$-greedy:

$$a_t = \begin{cases} \text{random action,} & \text{with probability } \epsilon, \cr \arg\max_a Q_\theta(s_t, a), & \text{otherwise} \end{cases}$$

- Execute $a_t$, observe $(r_t, s_{t+1})$, and store $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{D}$
- "// Sample a mini-batch"
- Sample $\lbrace (s_i, a_i, r_i, s_{i+1}) \rbrace_ {i=1}^{N} \sim \mathcal{D}$
- "// Compute targets"
- **for** $i = 1, \ldots, N$ **do**, with body $y_i \leftarrow r_i + \gamma \max_{a'} Q_{\theta'}(s_{i+1}, a')$
- "// Q-network update"
- $\theta \leftarrow \theta - \alpha \nabla_\theta \frac{1}{N} \sum_{i=1}^{N} \left[ Q_\theta(s_i, a_i) - y_i \right]^2$
- "// Periodic target update"
- **if** $t \bmod C = 0$ **then**, with body $\theta' \leftarrow \theta$

## Slide 34 — Actor Critic: SAC

Subtitle: "Whiteboard".

The slide marks a whiteboard segment, which is not in the recording: the recording ends on slide 31.

A pseudocode box titled "Algorithm 1: Soft Actor-Critic (no entropy term)":

**Input:** Replay buffer $\mathcal{D}$, critic nets $Q_{\theta_1}, Q_{\theta_2}$, actor net $\pi_\phi$, target critics $\theta'_ i \leftarrow \theta_i$, discount $\gamma$, smoothing $\tau$, batch size $N$

**for** each interaction step $t = 1, 2, \ldots$ **do** (the body below is inside the loop, marked by a vertical bar)

- Observe state $s_t$
- Sample action $a_t \sim \pi_\phi(\cdot \mid s_t)$
- Execute $a_t$, observe $(r_t, s_{t+1})$ and store in $\mathcal{D}$
- "// Sample a mini-batch"
- Sample $\lbrace (s_i, a_i, r_i, s_{i+1}) \rbrace_ {i=1}^{N} \sim \mathcal{D}$
- "// Compute targets (no entropy term)"
- **for** $i = 1, \ldots, N$ **do**, with body $a'_ {i+1} \leftarrow \pi_\phi(s_{i+1})$; then $y_i \leftarrow r_i + \gamma \min_{j=1,2} Q_{\theta'_ j}(s_{i+1}, a'_ {i+1})$
- "// Critic update"
- **for** $j = 1, 2$ **do**, with body $\theta_j \leftarrow \theta_j - \lambda_Q \nabla_{\theta_j} \frac{1}{N} \sum_i \left[ Q_{\theta_j}(s_i, a_i) - y_i \right]^2$
- "// Actor update (maximize Q only)"
- $\phi \leftarrow \phi + \lambda_\pi \nabla_\phi \frac{1}{N} \sum_i Q_{\theta_1}(s_i, \pi_\phi(s_i))$
- "// Target networks soft-update"
- **for** $j = 1, 2$ **do**, with body $\theta'_ j \leftarrow \tau \theta_j + (1 - \tau) \theta'_ j$

## Slide 35 — [Thanks!]

Centred text only: "Thanks!"
