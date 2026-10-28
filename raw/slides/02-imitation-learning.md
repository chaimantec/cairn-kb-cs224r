---
title: Lecture 2 — Imitation Learning (slide deck)
lecture: 2
slides: 27
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/02_cs224r_imitation_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–27). Pages 10 and 25 print no number; labels inferred from position.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (7, 8, 10, 12, 19, 21) were checked by Claude Opus 5.5 against 1400px renders. All six agreed on series, values, structure, arrows, labels and equations. Three descriptive fixes: slide 7's arrow directions (the traffic faces the camera, so the arrow to the image's right is the driver's left), the position of Mario on slide 8, and slide 12's right-hand photo (a wooden tabletop, not a floor).
images: 15 slides are rendered and shown under their headings (3, 5, 7–10, 12–14, 18–21, 23, 24). Not rendered — slides 1 and 15, whose only images are SpongeBob cartoon stills credited on the slides to Viacom International and described below; and the reminder, agenda, text and equation-only slides (2, 4, 6, 11, 16, 17, 22, 25, 26, 27).
---

# Lecture 2 — Imitation Learning: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 2 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 27 pages. Slides 10 and 25 print no number; their labels are inferred from position.

The recording covers slides 3–21. It opens on the recap (slide 3) without discussing slide 2's reminders, and it ends at about 67:05 in the questions after slide 21, so section 4 of the plan (slides 22–24, how to collect demonstrations), the summary (slide 25) and the closing slides (26–27) are never discussed. For those slides this file is the only source.

Companion pages: [wiki page for this lecture](../../wiki/02-imitation-learning.md) · [transcript](../transcripts/02-imitation-learning.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–4 | Title, course reminders, partial recap, plan for today |
| 5–9 | Goal of imitation learning, version 0 (deterministic regression), what could go wrong, learning distributions with neural networks |
| 10–16 | Generative models for policies, version 1, version 1 vs version 0, robotics and driving examples, summary so far |
| 17–21 | Compounding errors and interventions (DAgger, human-gated DAgger) |
| 22–24 | How to collect demonstrations, embodiment gap |
| 25–27 | Summary, plan recap, next time and course reminders |

## Slide 1 — Imitation Learning

Title slide. Top: a cartoon still from SpongeBob SquarePants showing, inside a restaurant kitchen, Patrick Star (pink, wearing a fake yellow moustache and an onion-ring-like ring over one eye, standing behind a grey cash register) on the left and SpongeBob (yellow, hands on hips, smiling, holding a yellow book or box behind him) on the right. A burger poster hangs at the upper left and a steaming burger sits on a ledge between them. The expression of both characters suggests Patrick is imitating SpongeBob.

Title: "Imitation Learning". Below: "CS 224R". Bottom right, small: "SpongeBob image rights & credits to Viacom International".

## Slide 2 — Course reminders

- Start forming **final project groups** (survey due Weds April 16)
- **Homework 1** out today, due Fri April 18
- PyTorch tutorial today at 1:30 pm in **Gates B1**

## Slide 3 — Partial Recap

![Slide 3 — Partial Recap](../images/02-imitation-learning/slide-3.jpg)

- *state* $\mathbf{s}_ t$ - the state of the "world" at time $t$
- *observation* $\mathbf{o}_ t$ - what the agent observes at time $t$
- *action* $\mathbf{a}_ t$ - the decision taken at time $t$
- *trajectory* $\tau$ - sequence of states/observations and actions

$$(\mathbf{s}_ 1, \mathbf{a}_ 1, \mathbf{s}_ 2, \mathbf{a}_ 2, \ldots, \mathbf{s}_ T, \mathbf{a}_ T)$$

- *reward function* $r(\mathbf{s}, \mathbf{a})$ - how good is $\mathbf{s}, \mathbf{a}$?

Figure (the policy network from Lecture 1, left to right): a photo labelled $\mathbf{s}$ at its right edge (a robot gripper above a pale yellow-green towel on a grey table next to a metal hook); an arrow into a convolutional network drawn as three stacks of blue squares followed by three narrow grey vertical bars (fully connected layers); an arrow to a label $\mathbf{a}$ and a second photo of the same scene with a red arrow drawn near the gripper over the towel. Layer labels are very small; legible ones are "7x7 conv stride 2 ReLU", "64 filters", "5x5 conv ReLU", "32 filters", "5x5 conv ReLU", "32 filters", "fully connected ReLU", "fully connected ReLU", "fully connected linear". Below the network: $\pi_\theta(\mathbf{a} \mid \mathbf{s})$.

- *policy* $\pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$ or $\pi(\mathbf{a}_ t \mid \mathbf{o}_ {t-m:t})$ - behavior, usually what we are trying to learn

Last line, centred: "can be represented using a generative model"

## Slide 4 — The plan for today

"Imitation Learning", then a numbered list:
1. Imitation learning basics
2. Learning expressive policy distributions
3. Learning from online interventions
4. Time permitting: how to collect demonstrations

A large right brace spans items 1–3 and is labelled "Topic of homework 1!".

"Key learning goals:"
- how to represent distributions with neural networks
- why expressive distributions matter for imitation learning
- what are compounding errors and how to address them

## Slide 5 — What is the goal of imitation learning?

![Slide 5 — What is the goal of imitation learning?](../images/02-imitation-learning/slide-5.jpg)

- **Data** (teal): Given trajectories collected by an expert "*demonstrations*"

$$\mathcal{D} := \lbrace(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T)\rbrace$$

(sampled from some unknown policy $\pi_{\text{expert}}$)

A teal database-cylinder icon sits to the right of the Data line.

- **Goal** (blue): Learn a policy $\pi_\theta$ that performs at the level of the expert policy, by mimicking it.

Pink box at the right, headed "Example", containing a photograph of a multi-lane highway with cars and a white box truck, and a red truck at the lower left (the same photo as Lecture 1), and the text:
- Dataset from human drivers
- Sensor readings + steering commands

## Slide 6 — Imitation learning - version 0

Subtitle: "Deterministic policy"

0. Given demonstrations collected by an expert $\mathcal{D} := \lbrace(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T)\rbrace$
1. For deterministic policy, supervised regression to the expert's actions

$$\min_\theta \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{s}, \mathbf{a}) \in \mathcal{D}} \lVert \mathbf{a} - \hat{\mathbf{a}} \rVert^2 \qquad \text{where } \hat{\mathbf{a}} = \pi_\theta(\mathbf{s})$$

2. Deploy learned policy $\pi_\theta$

## Slide 7 — What could go wrong?

![Slide 7 — What could go wrong?](../images/02-imitation-learning/slide-7.jpg)

Left: pink box headed "Example" with the highway photograph (multi-lane road, a white box truck, a silver car in the centre, a red truck at the lower left; the traffic is driving towards the camera). Two arrows start at the silver car: a yellow arrow pointing down and to the right, along its lane (staying straight), and a blue arrow curving off to the image's right, which is the driver's left (merging left). Text in the box:
- Dataset from human drivers
- Sensor readings + steering commands

Teal text: "Question: what might policy trained with $\ell_2$-regression do?"

Chart titled "Demonstration data distribution". X axis "steering command", ticks from -3.0 to 1.0 in steps of 0.5. Y axis "Probability Density", ticks 0.00 to 2.00 in steps of 0.25. Legend in the upper centre. Three series:
- "merge left data": solid blue curve, a Gaussian bump centred at about -2.0 with peak about 2.00.
- "stay straight data": solid yellow curve, a Gaussian bump centred at about 0.0 with peak about 2.00.
- "combined data": black dashed curve with grey fill beneath it, two bumps: about 0.50 at -2.0 and about 1.50 at 0.0 (the blue bump is weighted lower than the yellow one in the mixture).

A vertical teal dashed line sits at about -0.5 (between the two bumps, where the density is near zero); a teal curved arrow runs from the teal label "mean of data" (lower right of the chart) to this line. The label is an annotation, not an additional series.

Bold italic, lower right: "How can we represent more than the mean?"

Bottom: "How often does this happen in practice?   All the time!" then "Esp. when data collected by multiple people."

## Slide 8 — Learning distributions with neural networks

![Slide 8 — Learning distributions with neural networks](../images/02-imitation-learning/slide-8.jpg)

Left: a hand-drawn style curve with three peaks (a tall left peak and two lower peaks to its right) on small axes, labelled in italics "parameters of distribution". Two upward arrows lead from the top of a small neural network (a diamond-shaped graph: one grey input node at the bottom with an arrow from below, a row of four orange hidden nodes, and two grey output nodes at the top) up to that label.

Middle column, bold heading "1D discrete actions". Label $\mathbf{s}$ above a Super Mario game screenshot (blue sky, brick blocks with question marks, small Mario in mid-air beside a block, a Goomba on the ground, green hills). Label $p(\mathbf{a})$ above a bar chart with four peach bars of decreasing height (about 1.0, 0.75, 0.6, 0.2 of the tallest), x labels rotated: "up arrow", "down arrow", "press A", "press B". Text: "Neural net outputs $p(\text{up}), p(\text{down}), \ldots$ represent categorical distribution." Green text: "Maximally expressive".

Right column, bold heading "Continuous actions". Label $\mathbf{s}$ above a photograph from a driver's seat (steering wheel and dashboard, a mountain road through the windscreen). Label $p(\mathbf{a})$ above a single grey bell curve with a teal dashed vertical line at the peak labelled $\mu$ (teal, above) and a green horizontal bar to the right of the peak labelled $\sigma$ (green); x-axis caption "next steering angle". Text: "Neural net outputs $\mu, \sigma$ to represent Gaussian distribution." Red text: "Not very expressive!"

## Slide 9 — Learning distributions with neural networks

![Slide 9 — Learning distributions with neural networks](../images/02-imitation-learning/slide-9.jpg)

Subtitle, with a light-bulb emoji: "Can we use generative modeling?"

Two columns:
- Left, "image diffusion models": a square image of coloured random noise (a diffusion starting image). Caption below: "learning $p(\text{image} \mid \text{text description})$".
- Right, "autoregressive models": grey italic placeholder text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat." Caption below: "learning $p(\text{next word} \mid \text{words so far})$".

Bottom: "**Our goal**: learning $p(\text{action} \mid \text{observations})$"

## Slide 10 — Generative models for policies, approximating $p(\mathbf{a} \mid \mathbf{s})$

![Slide 10 — Generative models for policies](../images/02-imitation-learning/slide-10.jpg)

(No number is printed on this page; the label is inferred from position.)

Three columns, each with a small neural network diagram (grey input node at bottom with an upward arrow, four orange hidden nodes, two grey output nodes with upward arrows):

1. Bold heading "Mixture of Gaussians". A plot labelled $p(\mathbf{a}_ t)$: a black curve with a tall central peak and a smaller peak on the right and a low bump on the left, with dotted coloured component curves (red, blue, green) beneath. Text: "output $\mu_1, \sigma_1, w_1, \mu_2, \sigma_2, w_2, \ldots$". Network input labelled $\mathbf{s}_ t$.
2. Bold heading "Discretize + Autoregressive". A histogram labelled $p(\mathbf{a}_ {t,d})$ with many pale green bars forming three tall clusters. Text: "output $p(\mathbf{a}_ {t,1}), p(\mathbf{a}_ {t,2} \mid \hat{\mathbf{a}}_ {t,1}), p(\mathbf{a}_ {t,3} \mid \hat{\mathbf{a}}_ {t,1:2}), \ldots$". Network input labelled $\mathbf{s}_ t$.
3. Bold heading "Diffusion". A photo of a robot arm above a white table with an orange T-shaped block (a push-T style task), with the italic text "iteratively denoise" at its left. Text: "output $\epsilon_n$". Network inputs labelled $\mathbf{s}_ t$ and $\mathbf{a}_ t + \sum_{i=1}^{n} \epsilon_i$. A circular arrow runs from the network's right side around and back, labelled $n = N \ldots 1$.

Teal text: "Question: how are these different from $\ell_2$ loss with larger network?"

"**Important Note**: Neural network expressivity is often distinct from distribution expressivity!"

## Slide 11 — Imitation learning - version 1

Subtitle: "Expressive policies"

0. Given demonstrations collected by an expert $\mathcal{D} := \lbrace(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T)\rbrace$
1. Train **generative model** of the expert's actions

$$\min_\theta - \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}} \left[ \log \pi_\theta(\mathbf{a} \mid \mathbf{s}) \right] \qquad \text{with expressive distribution } \pi(\cdot \mid \mathbf{s})$$

2. Deploy learned policy $\pi_\theta$

Annotation under the equation: "maximize the log probability of the demo actions under the policy", with coloured words and matching coloured underlines under the equation. "log probability" is pink (underlining $\log \pi_\theta$), "demo actions" is blue (underlining $(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}$ and $\mathbf{a}$), "policy" is teal (underlining $\pi_\theta$).

## Slide 12 — Imitation learning - version 1 vs version 0

![Slide 12 — Imitation learning - version 1 vs version 0](../images/02-imitation-learning/slide-12.jpg)

Two bar charts, both with y axis "success rate" (ticks 0, 0.25, 0.5, 0.75, 1) and x-axis caption "demo dataset type".

Left chart, titled "Simulated transport task". Legend: dark green "diffusion", light green "GMM". X groups: "single human", "multi human". Four bars (two series):
- diffusion (dark green): single human about 1.0; multi human about 0.88.
- GMM (light green): single human about 0.87; multi human about 0.43.

Right chart, titled "Real shirt hanging task". Legend: green "diffusion", blue "L1". One x group "multi-human data". Two bars:
- diffusion (green): about 0.70.
- L1 (blue): about 0.25.

Below the left chart: a rendered simulation image of two robot arms at tables on a wooden tiled floor. Below the right chart: a photograph of a lab workspace: a square wooden tabletop inside a metal frame, with a robot arm at each corner.

Bottom left, URLs: "https://diffusion-policy.cs.columbia.edu/" and "https://aloha-unleashed.github.io/"

## Slide 13 — Robotics: Imitation learning + expressive policies

![Slide 13 — Robotics: Imitation learning + expressive policies](../images/02-imitation-learning/slide-13.jpg)

Four image panels, each with a title and a small caption naming the method:
- "Physical Intelligence $\pi_0$": a room with a washer and dryer and a kitchen beyond; a two-armed robot at the left foreground, with the overlay text "autonomous, 2x speed" at the lower left. Caption: "diffusion". Appears to be a video frame.
- "NVIDIA Gr00t N1": two humanoid robots with white hard hats at a shelf in a dim warehouse, with yellow bins; a small banner reads "Autonomous | Real" and "Fourier GR-1" with "1X Speed". Caption: "diffusion". Appears to be a video frame.
- "Figure Helix": two dark humanoid robots by a refrigerator, and a man holding a paper bag at a black counter with a wooden tray and groceries. Caption: "diffusion". Appears to be a video frame.
- "OpenVLA": a 2 by 2 grid of four robot-view photos: a toy sink with a red cup, a steel pot and toy vegetables with a gripper above; a robot arm at a black table with a pink bowl, a steel pot and a white bowl; a tan table with an orange and a soda can; a tabletop with gold tinsel, a dustpan and a small brick-pattern surface. Caption: "discretize + autoregressive prediction".

## Slide 14 — Autonomous driving: Imitation learning + expressive policies

![Slide 14 — Autonomous driving: Imitation learning + expressive policies](../images/02-imitation-learning/slide-14.jpg)

Two panels:
- "Waymo EMMA": a street-level photograph of a tree-lined residential road with parked cars at the right outlined in cyan 3D boxes, a small pedestrian box in the distance, and a translucent teal band over the lane showing the planned path. Caption: "discretize + autoregressive prediction".
- "Wayve LINGO-2": a dashcam frame of a city street with a glass-fronted building and traffic lights, a motorcyclist ahead; a Wayve logo at the upper right; small green text at the upper left reading "Autonomous: True" and "Speed: 0.00 mph" (partly legible); overlay text "the first language model to drive on public roads"; a black caption bar at the bottom reading "Staying stopped due to the red light." Caption: "discretize + autoregressive prediction". Appears to be a video frame.

## Slide 15 — Summary so far

Two columns, each with a SpongeBob cartoon still and a caption.
- Left: heading "Data from one consistent demonstrator". Image: the same still as slide 1 (Patrick Star in a fake moustache behind a grey register, SpongeBob standing with hands on hips). Caption: "Unimodal policy distribution is enough".
- Right: heading "Multimodal data, e.g. from multiple demonstrators". Image: a cartoon of one large SpongeBob body whose arms and neck are stretched out to several other characters' heads: a small SpongeBob head in a glass bubble (left), Plankton (small, green, on the left shoulder), a second SpongeBob head at the top, Mr. Krabs (red, eyes on stalks, upper right) and Squidward (pale green, drooping eyelids, right) pulling on the body, in a room with a red armchair, a life ring and green mattresses. Caption: "Need expressive generative model for policy".

Bottom right, small: "SpongeBob image rights & credits to Viacom International"

## Slide 16 — Summary so far

- **Key idea**: Train expressive policy class via generative modeling on dataset of demonstrations.
- Algorithm is fully *offline*

Red-outlined box at the right, headed "Definitions." (red):
- *offline*: using only an existing dataset, no new data from learned policy
- *online*: using new data from learned policy

Green lines:
- \+ no need for data from policy (online data can be unsafe, expensive to collect) — the parenthetical is printed in smaller type
- \+ no need to define a reward function

Red line:
- \- may need **a lot** of data for reliable performance

## Slide 17 — The plan for today

Same agenda as slide 4, with items 1, 2 and 4 greyed out and item 3 ("Learning from online interventions") in black:
1. Imitation learning basics
2. Learning expressive policy distributions
3. Learning from online interventions
4. Time permitting: how to collect demonstrations

A right brace spans items 1–3, labelled "Topic of homework 1!". Heading line "Imitation Learning". The key learning goals block of slide 4 is not shown.

## Slide 18 — What can go wrong in imitation learning?

![Slide 18 — What can go wrong in imitation learning?](../images/02-imitation-learning/slide-18.jpg)

Bold subtitle: "Compounding errors"

Two panels.

Left, heading "Supervised learning". Figure: a light-blue tilted ellipse labelled $p(\mathbf{x})$ (blue italic, at its lower right) containing three black dots labelled $\mathbf{x}_ 1$, $\mathbf{x}_ 2$, $\mathbf{x}_ 3$. Text: "Inputs independent of predicted labels $\hat{\mathbf{y}}$"

Right, heading "Supervised learning of behavior". Figure: four teal dots labelled $\mathbf{s}_ 1$ (bottom left), $\mathbf{s}_ 2$, $\mathbf{s}_ 3$, $\mathbf{s}_ 4$ (top right) along a rising curved path, with black arrows $\mathbf{s}_ 1 \to \mathbf{s}_ 2 \to \mathbf{s}_ 3 \to \mathbf{s}_ 4$. Beside each of the first three teal dots, a grey dot slightly offset from the next teal dot, reached by a thin grey arrow labelled $\hat{\mathbf{a}}_ 1$, $\hat{\mathbf{a}}_ 2$, $\hat{\mathbf{a}}_ 3$ (the predicted action's resulting state); a grey dot also sits beside $\mathbf{s}_ 4$. The grey endpoints land slightly off the teal points, so the grey path drifts away from the teal path.

Text to the right of the figure:
- Predicted actions affect next state.
- Errors can lead to drift away from the data distribution!
- Errors can then compound!

Equation: $p_{expert}(\mathbf{s}) \neq p_\pi(\mathbf{s})$, with a teal underline under $p_{expert}(\mathbf{s})$ labelled in teal "states visited by expert" and a grey underline under $p_\pi(\mathbf{s})$ labelled in grey "states visited by learned policy $\pi$". Last line: "covariate shift" (in quotation marks).

## Slide 19 — What can go wrong in imitation learning?

![Slide 19 — What can go wrong in imitation learning?](../images/02-imitation-learning/slide-19.jpg)

Bold subtitle: "Compounding errors"

Left, heading "Supervised learning of behavior": the same drift figure as slide 18 (teal states $\mathbf{s}_ 1$ to $\mathbf{s}_ 4$ joined by black arrows, grey predicted points reached by grey arrows labelled $\hat{\mathbf{a}}_ 1$, $\hat{\mathbf{a}}_ 2$, $\hat{\mathbf{a}}_ 3$), with the text:
- Predicted actions affect next state.
- Errors can lead to drift away from the data distribution!
- Errors can then compound!

and $p_{expert}(\mathbf{s}) \neq p_\pi(\mathbf{s})$.

Right, heading "Solutions?":
1. Collect A LOT of demo data & hope for the best.
2. Collect (orange) corrective behavior data

Figure below item 2: the same drift figure, now with orange arrows added from the grey drifted points ($\hat{\mathbf{a}}_ 2$ and $\hat{\mathbf{a}}_ 3$ positions) pointing along the teal path to the next teal point, showing corrective actions pointing back toward the expert's states. Labels $\mathbf{s}_ 1$ to $\mathbf{s}_ 4$ and $\hat{\mathbf{a}}_ 1$ to $\hat{\mathbf{a}}_ 3$ as before.

## Slide 20 — Addressing Compounding Errors with Interventions

![Slide 20 — Addressing Compounding Errors with Interventions](../images/02-imitation-learning/slide-20.png)

Subtitle: "Collect" followed by orange "corrective behavior data".

Figure at the left: the same drift figure as slide 19 with orange corrective arrows from the grey points back toward the teal path, labels $\mathbf{s}_ 1$ to $\mathbf{s}_ 4$ and $\hat{\mathbf{a}}_ 1$ to $\hat{\mathbf{a}}_ 3$.

A grey curved arrow runs up the left side of the numbered list, from step 4 back to step 1, indicating the loop repeats. Steps (the printed numerals are shown as 1.–4.):
1. Roll out learned policy $\pi_\theta$: $\mathbf{s}'_ 1, \hat{\mathbf{a}}_ 1, \ldots, \mathbf{s}'_ T$ (the trajectory is printed in grey)
2. Query expert action at visited states $\mathbf{a}^{\ast} \sim \pi_{expert}(\cdot \mid \mathbf{s}')$ ($\mathbf{a}^{\ast}$ is orange)
3. Aggregate corrections with existing data $\mathcal{D} \leftarrow \mathcal{D} \cup \lbrace(\mathbf{s}', \mathbf{a}^{\ast})\rbrace$ ($\mathbf{a}^{\ast}$ is orange)
4. Update policy $\min_\theta \mathcal{L}(\pi_\theta, \mathcal{D})$

"dataset aggregation" (DAgger) — "dataset aggregation" is in quotation marks.

- \+ data-efficient way to learn from an expert (green)
- \- can be challenging to query expert when agent has control (red)

Bottom: "Is there another way to collect corrective data?"

Bottom right: a green clip-art car viewed from the front, with a smiling girl in pink at the left window and a small grey robot with a star on its chest at the steering wheel on the right.

## Slide 21 — Addressing Compounding Errors with Interventions

![Slide 21 — Addressing Compounding Errors with Interventions](../images/02-imitation-learning/slide-21.png)

Subtitle: "Collect" then orange "corrective behavior data", then "while *taking full control*".

Figure at the left: teal dots $\mathbf{s}_ 1$ to $\mathbf{s}_ 5$ along a rising path joined by black arrows, a teal star labelled $\mathbf{s}_ 6$ at the top right. Grey dots with grey arrows labelled $\hat{\mathbf{a}}_ 1$, $\hat{\mathbf{a}}_ 2$, $\hat{\mathbf{a}}_ 3$ show the policy's drifting predictions for the first three steps. From the grey point after $\hat{\mathbf{a}}_ 3$, an orange arrow leads to an orange dot, and a further orange arrow from that dot leads to the star $\mathbf{s}_ 6$: the expert's takeover, steering back to the goal. A grey curved arrow runs up the left of the list from step 5 to step 1.

Steps:
1. Start to roll-out learned policy $\pi_\theta$: $\mathbf{s}'_ 1, \hat{\mathbf{a}}_ 1, \ldots, \mathbf{s}'_ t$ (grey)
2. Expert intervenes at time $t$ when policy makes mistake
3. Expert provides (partial) demonstration $\mathbf{s}'_ t, \mathbf{a}^{\ast}_ t, \ldots, \mathbf{s}'_ T$ (orange)
4. Aggregate new demos with existing data $\mathcal{D} \leftarrow \mathcal{D} \cup \lbrace(\mathbf{s}'_ i, \mathbf{a}^{\ast}_ i)\rbrace; i \geq t$ ($\mathbf{s}'_ i$ and $\mathbf{a}^{\ast}_ i$ are orange)
5. Update policy $\min_\theta \mathcal{L}(\pi_\theta, \mathcal{D})$

"human gated DAgger" (in quotation marks)

- \+ (much) more practical interface for providing corrections (green)
- \- can be hard to catch mistakes quickly in some application domains (red)

Teal text, lower right: "Question: could you automatically detect when intervention is needed?"

## Slide 22 — The plan for today

Same agenda as slide 4, with items 1–3 greyed out and item 4 ("Time permitting: how to collect demonstrations") in black. Heading "Imitation Learning". A right brace spans items 1–3, labelled "Topic of homework 1!".

## Slide 23 — How to collect demonstrations?

![Slide 23 — How to collect demonstrations?](../images/02-imitation-learning/slide-23.jpg)

**In some domains**: People already collect demonstrations that can be recorded e.g. driving cars, writing text messages

"What about robotics?" Three columns:
- "Kinesthetic teaching": a photo of a person's hands physically guiding the grey arm of a PR2 robot (the label "PR2" is visible on the robot), with a wooden dish rack at the bottom. Green: "+ easy interface". Red: "- human visible in scene".
- "Remote controllers": a white Xbox-style game controller at the upper left, a black and white 3D-mouse puck at the upper right, and a white VR hand controller with a ring at the lower right. Orange: "~ interface ease varies, can have high latency".
- "Puppeteering": a photo of a smiling woman in a brown jacket holding two small handheld leader arms in a lab, with larger robot arms in the foreground and another person partly visible at the right. Green: "+ easy interface". Red: "- requires double hardware".

Bottom: "**In other domains**: It may not be viable to collect demos! (e.g. quadruped robot)"

## Slide 24 — Can robots directly use videos of people, animals?

![Slide 24 — Can robots directly use videos of people, animals?](../images/02-imitation-learning/slide-24.jpg)

**Embodiment gap**:
- difference in appearance
- difference in physical capabilities, degrees of freedom

"Hard to directly imitate human & animal data, but can potentially guide exploration."

Figure: a filmstrip of 10 frames in two rows. Top row, five frames of a person on grass doing a cartwheel (standing with arms raised, bending sideways, handstand-like, legs apart, landing), then five frames of the same or another person doing a backflip in a field (crouch, in the air tucked, upside down, landing crouched, standing). Bottom row, matching simulated characters: a blue-grey stick-figure humanoid doing the cartwheel on a bumpy grey terrain, then a boxy grey humanoid robot doing the flip on a flat tiled floor. Each simulated frame matches the video frame above it.

Citation: "Peng, Kanazawa, Malik, Abbeel, Levine. SFV: Reinforcement Learning of Physical Skills from Videos. SIGGRAPH Asia 2018."

## Slide 25 — Summary

(No number is printed on this page; the label is inferred from position.)

**Part 1**: Train policy to mimic offline demonstrations
- best if policy is an expressive generative model over actions
- algorithm is fully *offline*

At the right: "often referred to as **behavior cloning** (**BC**)"

**Part 2**: Improve policy using online interventions
- requires interface for human or expert intervention
- algorithm involves running policy *online*

At the right: "often referred to as **DAgger**, or **HG-DAgger**"

Green lines:
- \+ **offline BC**: simple, no need for data from policy (online data can be unsafe, expensive) — the parenthetical is in smaller type; "offline BC" is bold
- \+ DAgger: possible path to reliable performance, more data-efficient than offline BC
- \+ no need to define a reward function

Red lines:
- \- may need **impractically large amount** of data for reliable performance
- \- doesn't provide a framework for improving on own (from "practicing")

Bottom: "Many successful methods **combine** imitation learning and reinforcement learning!"

## Slide 26 — The plan for today

Same as slide 4: heading "Imitation Learning", items 1–4 all in black (1. Imitation learning basics; 2. Learning expressive policy distributions; 3. Learning from online interventions; 4. Time permitting: how to collect demonstrations), a right brace spanning items 1–3 labelled "Topic of homework 1!", and "Key learning goals:" with the three bullets:
- how to represent distributions with neural networks
- why expressive distributions matter for imitation learning
- what are compounding errors and how to address them

## Slide 27 — Next time

"Start of *reinforcement learning algorithms*"

Second heading: "Course reminders"
- Start forming **final project groups** (survey due Weds April 16)
- **Homework 1** out today, due Fri April 18
- PyTorch tutorial today at 1:30 pm in **Gates B1**
