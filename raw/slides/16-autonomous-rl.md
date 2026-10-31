---
title: "Lecture 16 — RL for Robots: Autonomous Learning (slide deck)"
lecture: 16
slides: 46
printed_numbers: "28, 30 and 32 only"
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/16_cs224r_autonomy_2025.pdf
numbering: 1:1 with PDF pages. Only pages 28, 30 and 32 print a number in the text layer (bottom right), each equal to its page index; the other 43 pages print none, and because the printed numbers leave no gap, the labels are the page index
figure_audit: Transcribed by Claude Sonnet 5.5. Seven figure slides (13, 16, 17, 24, 25, 28, 30) were checked by Claude Opus 5.5 against 1400px renders. Slides 13 (both equations and the three arrows), 16 (every series against its label and its values) and 28 (every arrow of the task graph) agree. Corrections: slide 24's early values for VaPRL, MEDAL, FBRL and the oracle, and SAC's last value; slide 25's values for MEDAL at 8.0e5, the unlabelled green series and SAC; slide 17's partly covered green line is a curve, not a blob; slide 30's hand is described without a finger count. The transcriber flagged slides 8, 26, 30–33, 39 and 42–44 as possible video frames; the transcript confirms videos on all of them except 32, whose half-empty frame is most likely the first frame of a video or animation that Finn talks over, and 37–38 hold the first frame of the video that plays on 39. What each video showed is under its slide, from the transcript.
images: 25 slides are rendered and shown under their headings (5, 6, 7, 8, 11, 12, 13, 16, 17, 19, 22, 23, 24, 25, 26, 28, 30, 31, 32, 33, 39, 40, 42, 43, 44). Not rendered: build steps or repeats whose content a rendered slide shows (10, slide 6's reset loop with its "Problem" label, transcribed here; 18, slide 16's chart repeated small; 29, slide 28's task graph beside the algorithm transcribed here; 37 and 38, the door before the hand of slide 39 enters; 41, slide 40's single-life panel and walker images beside text); text slides (14, 20, 21, 27, 35, 36); the title, reminder, plan and summary slides (1, 2, 3, 4, 9, 15, 34, 45); and slide 46, whose only picture is a portrait of the next guest lecturer.
---

# Lecture 16 — RL for Robots: Autonomous Learning: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025) Lecture 16 deck, transcribed from the PDF. The title slide (slide 1) prints "RL for Robots" followed by a robot emoji, then ": Autonomous Learning", with "CS224R: Deep Reinforcement Learning" beneath it, and names no lecturer. Cite these as "slide N". Diagrams, photographs and plots are described in prose since the KB is read as text. Most slides credit "Slide adapted from Archit Sharma".

The recording (about 66 minutes) opens with the plan for the day (slide 3, ≈0:05), without discussing slide 2's reminders, follows the deck through slide 44 and closes on slide 45's summary (≈1:05:07, ending at about 1:05:41); slide 46's reminders and "next time" are not discussed, so for them this file is the only source. Finn writes on the slides where the recording does not show it: an agent's reset-free experience, drawn as a red trajectory (≈9:16–11:35), the continuing-evaluation formula (≈13:53–14:40), the drift and collapse sketch (≈19:19–21:38), the forward-backward algorithm (≈23:14–25:33), a step "negative one" for the demonstrations (≈30:12) and the task-cycle algorithm (≈44:02–45:35); the transcript is the only source for those. Slides 8, 26, 30, 31, 33, 37–39 and 42–44 hold one frame each of videos that play in the lecture, and slide 32 most likely the first frame of a video or animation; what they showed is under those slides.

## Slide numbers vs PDF pages

The deck is 1:1 with its page index: PDF page N is slide N, for all 46 pages. Only pages 28, 30 and 32 print a number in the text layer (bottom right), each equal to its page index. The other 43 pages print no number; because the printed numbers leave no gap, their labels are the page index.

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
| 40 | 40 |
| 41 | 41 |
| 42 | 42 |
| 43 | 43 |
| 44 | 44 |
| 45 | 45 |
| 46 | 46 |

## Sections of the deck

| Section | Slides |
| ------- | ------ |
| Title, reminders and plan | 1–4 |
| Why aren't robots autonomous already? | 5–8 |
| Defining the problem: autonomous RL | 9–14 |
| Developing the algorithms: learning policies without human intervention | 15–33 |
| Single-life RL | 34–44 |
| Summary and reminders | 45–46 |

## Slide 1 — RL for Robots: Autonomous Learning

Title slide. Centred title "RL for Robots 🤖: Autonomous Learning", where the robot-face emoji (a blue-grey robot head with an orange light on top) sits between "Robots" and the colon. Near the bottom, centred: "CS224R: Deep Reinforcement Learning". No lecturer or date is printed.

## Slide 2 — Reminders

Title "Reminders". Two lines: "- Project milestone due tonight" and "- HW4 due next Friday".

## Slide 3 — Plan for Today

Title "Plan for Today". Outline: "- Why aren't robots autonomous already?"; "- Defining the problem: **autonomous RL**" (the words "autonomous RL" in bold); "- Developing the algorithms"; then two indented sub-items, "- Learning policies without human intervention" and "- Single life RL". Below, centred: "**Goal**: Build autonomous agents that can learn and interact in the real world" ("Goal" in bold).

## Slide 4 — Plan for Today

The same outline as slide 3, with every item greyed out except the first, "- Why aren't robots autonomous already?", which is in black. The Goal line is absent.

## Slide 5 — Reinforcement Learning = Trial-and-Error

![Slide 5 — Reinforcement Learning = Trial-and-Error](../images/16-autonomous-rl/slide-5.jpg)

Title "Reinforcement Learning = Trial-and-Error". Credit at bottom left: "Slide adapted from Archit Sharma".

On the left is a diagram captioned "Learn a policy $\pi$ to go from $\rho_0$ to $\rho_g$". A blue circle labelled $\rho_0$ at the upper left and a green circle labelled $\rho_g$ at the upper right are joined by a broad pale-purple curved band that sags below them. A dark-red curve runs from $\rho_0$ to $\rho_g$ inside the band. Two brown curves also leave $\rho_0$: one wanders up and to the right and ends in the air above the band, the other drops down and ends at the lower right far from $\rho_g$. A vertical column of three brown dots between them stands for more trials.

On the right: "Repeat:" followed by three bullets with grey triangle markers: "Execute actions from the policy $\pi$", "Observe data from the environment", "Update the policy $\pi$".

## Slide 6 — Standard Reinforcement Learning

![Slide 6 — Standard Reinforcement Learning](../images/16-autonomous-rl/slide-6.jpg)

Title "Standard Reinforcement Learning". Credit: "Slide adapted from Archit Sharma".

At the upper left is a trajectory $s_0, a_0, s_1, a_1 \ldots s_H$. A curved grey arrow leaves the end ($s_H$) and loops back to the start of a second line, $s_ 0', a_ 0', s_ 1', a_ 1' \ldots$. Beneath it, in italics: "How does this happen?".

On the right is a screenshot of a code snippet in a dark grey frame (the right edge of the code is cut off). Its text reads: `import gym`; `env = gym.make("CartPole-v1")` (the string "CartPole-v1" in red); `observation = env.reset()`; `for _ in range(1000):`; `env.render()`; `action = env.action_space.sample() # y...` (cut off); `observation, reward, done, info = env.s...` (cut off); then, inside a red rectangle, `if done:` and `observation = env.reset()`; and `env.close()`. Under it: "[Code snippet from https://gym.openai.com/]". A dark arrow runs from the red rectangle down and to the left to the bold text "Only in simulation!".

## Slide 7 — The Continual Real World

![Slide 7 — The Continual Real World](../images/16-autonomous-rl/slide-7.jpg)

Title "The Continual Real World". Credit: "Slide adapted from Archit Sharma".

Three illustrated examples. Top left is an aerial photograph of a road, parking lot and trees with a path drawn on it: a green line runs along the road, turns and heads down to the lower right, while two red arrows head off in other directions (one up, one along the road to the right). Caption: "Navigate to the basketball court" (in quotation marks).

Top right is a pair of cartoon robot arms with an orange gripper. In the left picture a purple mug stands on a brown shelf; a dark-red arrow points to the right picture in which the mug has fallen off the end of the shelf, tipped over. Caption: "Grasp the mug".

At the bottom is a cartoon robot arm facing a book (green cover) and a yellow bookshelf. Dashed arrows from the book go to different places: two red dashed arrows end with the book off the shelf (above and below the shelf), a red arrow ends with a book lying flat in front of a shelf row, and green dashed arrows end with a book correctly among the others on shelf rows. A curly brace to the right of the shelf is labelled "Several thousands of trials!". Caption by the robot: "Learn how to shelf a book".

## Slide 8 — (photograph collage of robot learning examples)

![Slide 8 — (photograph collage of robot learning examples)](../images/16-autonomous-rl/slide-8.jpg)

No title. Credit: "Slide adapted from Archit Sharma". Three photographs, each with a citation below (may be a video frame for each).

- Left: a white wheeled humanoid robot (a PR2-style robot with a grey torso and an orange gripper) in a cluttered lab, with a stack of equipment behind it. The text overlay at the top reads "Hockey" and at the bottom "PILQR: 50 samples". A small orange object (a puck) is on the floor in front of the robot. Caption: "[Combining model-based and model-free updates for trajectory-centric reinforcement learning, Chebotar et al. 2017]" (may be a video frame).
- Top right: a person with a beard in a striped blue and white shirt stands at a table next to two wooden door frames, each holding a white door panel with a handle. Several robot arms stand behind the doors. The person reaches towards a door handle. A game controller lies on the table. Caption: "[Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search, Yahya et al. 2016]" (may be a video frame).
- Bottom right: a view into a dark box where two robot grippers hang from above over a pale surface holding a crumpled yellow-green cloth next to a metal circular fixture with an upright hook. The word "Nest" is in the top right corner. Caption: "[Self-Improving Robots: End-to-End Autonomous Visuomotor Reinforcement Learning, Sharma et al. 2023]" (may be a video frame).

In the lecture (≈6:58–7:44) these are three videos. On the left a robot learns to hit a hockey puck into a goal, and Finn's friend Yevgen, who resets it, "is probably doing more work than the robot"; at the top right Ali closes the doors so that the robot can learn to open them; at the bottom right Archit resets the robot so that it can attempt to fold the towel. The page holds one frame of each.

## Slide 9 — Plan for Today

The same outline as slide 3 without the Goal line, with every item greyed out except the second, "- Defining the problem: autonomous RL", in black.

## Slide 10 — (Reset Environment; increasing H)

No title. Credit: "Slide adapted from Archit Sharma".

At the top is the trajectory $s_0, a_0, s_1, a_1 \ldots s_H$ with a curved grey arrow from its end looping back to the start of $s_ 0', a_ 0', s_ 1', a_ 1' \ldots$ below. The label "Reset Environment" sits at the bend of the arrow. A black arrow points at that label from a red flag emoji followed by "Problem: this requires human supervision".

Lower left: "What if we increase $H$?" and beneath it a green check mark followed by "Fewer environment resets, less human supervision".

## Slide 11 — Autonomous RL: Definition

![Slide 11 — Autonomous RL: Definition](../images/16-autonomous-rl/slide-11.jpg)

Title "Autonomous RL: Definition". Credit: "Slide adapted from Archit Sharma".

A robot-face emoji at the upper left has two teal dashed arrows pointing to the elements of the sequence $s_0, a_0, s_1, a_1 \ldots$ (one at $a_0$, one at $a_1$). The label in teal reads "Agent interacts autonomously". The first state $s_0$ is highlighted in a pale orange box, and below it: "Initialize *once* at the beginning" (the word "once" in orange italics). To the right of the sequence: "No environment resets\*". Footnote at the bottom right: "\*can relax this constraint to reset occasionally, or at low frequency".

## Slide 12 — Autonomous RL: Evaluation

![Slide 12 — Autonomous RL: Evaluation](../images/16-autonomous-rl/slide-12.jpg)

Title "Autonomous RL: Evaluation". Credit: "Slide adapted from Archit Sharma".

A diagram like slide 5's. A blue circle labelled $\rho_0$ at the upper left and a gold star on a green circle at the upper right (the goal) are joined by a pale-purple curved band, along which a teal dashed arrow runs from $\rho_0$ to the goal star. A long orange curve leaves $\rho_0$, goes down, and winds through the rest of the picture (looping to the right, back up past the goal, then to a loop at the far right), passing through four more small gold stars (one low on the left, two close together in the middle, one at the lower right). The orange curve is the whole lifetime of the agent and the stars are the places where it received reward.

Text: "We might care about two different things:", then "- The amount of reward recovered over the course of its life (ex: Mars rover)" with "over the course of its life" in orange, and "- The quality of the learned policy (ex: robot chef)" with "learned policy" in teal.

## Slide 13 — Autonomous RL: Evaluation

![Slide 13 — Autonomous RL: Evaluation](../images/16-autonomous-rl/slide-13.jpg)

Title "Autonomous RL: Evaluation". Credit: "Slide adapted from Archit Sharma". Footnote at the bottom right: "\*average reward RL".

Two columns under green-highlighted headings.

Left, heading "Deployed Policy Evaluation", then "= Quality of policy learned" and the equation

$$J(\pi) = \mathbb{E}_ {\mathbf{s}_ 1 \sim p(\mathbf{s}_ 1), \mathbf{a}_ t \sim \pi} \left[ \sum_ t \gamma^t r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

Three orange arrows point from the parts of the equation to labels: from the subscript's $\mathbf{s}_ 1 \sim p(\mathbf{s}_ 1)$ to "Start from initial state distribution"; from $\mathbf{a}_ t \sim \pi$ to "Take actions according to $\pi$"; from the bracketed sum to "Total reward over the episode".

Right, heading "Continuing Policy Evaluation\*", then "= Reward accumulated over lifetime" and the equation

$$\lim_{h \to \infty} \mathbb{E} \left[ \frac{1}{h} \sum_{t=0}^{h} r(s_t, a_t) \right]$$

with the label beneath, "Average over reward accumulated in the lifetime". At the bottom centre: "We'll look at algorithms for both!". (The left equation prints bold $\mathbf{s}_ t, \mathbf{a}_ t$; the right one prints plain $s_t, a_t$.)

## Slide 14 — Why is autonomous RL important?

Title "Why is autonomous RL important?". Credit: "Slide adapted from Archit Sharma".

Text: "Robotics <-> Autonomy"; two indented bullets "- We want robots to operate with minimal human supervision" and "- We want robots to *train* with minimal human supervision" (asterisks printed around "train"). Then: "RL on real robots is promising path towards high success rate, reliability. Autonomy unlocks more data, more data may unlock greater generalization."

## Slide 15 — Plan for Today

The same outline as slide 3 without the Goal line, with the third item "- Developing the algorithms" and its first sub-item "- Learning policies without human intervention" in black; the others, including "- Single life RL", are greyed out.

## Slide 16 — What about standard RL algorithms?

![Slide 16 — What about standard RL algorithms?](../images/16-autonomous-rl/slide-16.jpg)

Title "What about standard RL algorithms?". Subtitle: "What happens when episode length increases?".

On the left is a rendered image of a simulated orange fish with a blue eye and orange fins in a dark blue starry-sky scene, beside a red ball, with the fish's shadow on a grey floor below. On the right is a line chart titled "Fish". The x axis is "Environment Steps", with ticks 0.0e0, 1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6. The y axis has no title; its ticks are 1.0e2, 2.0e2, 3.0e2, 4.0e2, 5.0e2. There are five series, each a line with a shaded band around it, and each is labelled by a coloured arrow at its right end with an episode length. All start near 80 to 100 at step 0 and begin to rise at about 0.5e6. Approximate values at 1.0e6, 2.0e6, 3.0e6 and 5.0e6 steps:

- "1000 steps", in green: about 250, 400, 460 and 490 (the highest series, with the widest band).
- "2000 steps", in red: about 190, 340, 370 and 345.
- "10,000 steps", in cyan: about 140, 230, 235 and 210.
- "50,000 steps", in purple: about 130, 200, 205 and 180.
- "100,000 steps", in olive yellow: about 95, 130, 125 and 130 (the lowest series).

Below the chart: "Note: this measures the **deployed policy evaluation**." (the last two words in bold). The longer the episode, the lower the performance.

## Slide 17 — The Challenge of Learning Autonomously

![Slide 17 — The Challenge of Learning Autonomously](../images/16-autonomous-rl/slide-17.jpg)

Title "The Challenge of Learning Autonomously". Credit: "Slide adapted from Archit Sharma".

Two diagrams side by side, each with a blue circle $\rho_0$ at the upper left, a green circle $\rho_g$ at the upper right and a pale-purple band between them labelled $\rho\ast$ in pale purple.

Left, headed "Episodic Learning": a dark-red curve runs inside the band from $\rho_0$ to $\rho_g$, with two brown curves that leave $\rho_0$, one going up and ending in the air and one going down and ending far below, and a column of three brown dots. Caption below: "Can always retry the task from initial state distribution".

Right, headed "Non-Episodic Learning": a thick dark-red curve inside the band joins $\rho_0$ and $\rho_g$, and a red arrow points down from it to a red warning-sign banner (a triangle with an exclamation mark) with the bold red text "The agent never learns a good policy". A brown curve leaves $\rho_0$ downward and wanders off at the bottom, where the banner partly covers it and a green curve. Beneath this, the caption "Challenge 1: exploration can cause the agent to drift far away". At the upper right, a blue scribble loops around $\rho_g$ and a grey dashed curve leaves the goal downwards; the caption at its right reads "Challenge 2: state distribution collapse".

## Slide 18 — How to autonomously learn an effective policy?

Title "How to autonomously learn an effective policy?". Credit: "Slide adapted from Archit Sharma".

On the left is the "Fish" chart of slide 16 again, at a smaller size and without the colour labels or the series' episode lengths (same axes: Environment Steps 0.0e0 to 5.0e6, y ticks 1.0e2 to 5.0e2, the same five series). A green arrow from the text on the right points at the top green series. Text: "Learns successfully if allowed to frequently retry *from the initial state distribution*" (the last four words in italics). Below it: "**Key idea**: learn a policy to reset?" ("Key idea" in bold).

## Slide 19 — Algorithm: Forward-Backward RL

![Slide 19 — Algorithm: Forward-Backward RL](../images/16-autonomous-rl/slide-19.jpg)

Title "Algorithm: Forward-Backward RL". Footnotes at bottom left: "[Han et al. Learning Compound Multi-Step Controllers under Unknown Dynamics]" and "[Eysenbach et al. Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning]".

On the left is a diagram with a blue circle $\rho_0$ at the upper left and a green circle $\rho_g$ at the upper right, a pale-purple band labelled $\rho\ast$ between them. A pink-purple wavy curve labelled $\pi_f$ ("Forward Policy" in pink-purple beneath it) goes from $\rho_0$ along the top to $\rho_g$ and another pink-purple curve goes down from $\rho_0$ and ends below. A blue curve labelled $\pi_b$ ("Backward Policy" in blue) goes from $\rho_g$ back to $\rho_0$ and another blue curve from $\rho_0$ ends at the lower middle. At the upper left, in italics: "Define $r_b(\mathbf{s}, \mathbf{a})$ for reaching $\rho_0$" (the reward symbol in light blue).

On the right, in bold centred: "Forward-Backward RL (FBRL)". Then:

1. Initialize forward policy $\pi_f$ and backward policy $\pi_b$
2. Repeat:
   1. Roll out $\pi_f$ for $H$ steps
   2. Update $\pi_f$ using $r_f(\mathbf{s}, \mathbf{a})$
   3. Roll out $\pi_b$ for $H$ steps
   4. Update $\pi_b$ using $r_b(\mathbf{s}, \mathbf{a})$
   5. *No reset or very infrequent reset*

($\pi_f$ in pink-purple, $\pi_b$ in blue, $r_f$ in green, $r_b$ in light blue.) Then "**Test time:** Discard $\pi_b$. Deploy $\pi_f$" and, in green, "+ simple" followed by, in red, "- backward policy is ultimately discarded".

## Slide 20 — Alternatives to Forward-Backward RL?

Title "Alternatives to Forward-Backward RL?". A numbered list:

1. Learn to reset to expert state distribution (forward-backward with modified backward reward) (the parenthesis in grey)
2. Practice *cycle* of different tasks
3. Set up task that can be done repetitively ("forward-forward RL"?) (the parenthesis in grey)

## Slide 21 — Alternatives to Forward-Backward RL?

The same list as slide 20, with item 1, "Learn to reset to expert state distribution", in bold.

## Slide 22 — Algorithm: Modified backward policy reward

![Slide 22 — Algorithm: Modified backward policy reward](../images/16-autonomous-rl/slide-22.jpg)

Title "Algorithm: Modified backward policy reward", with "Modified" in teal. Subheading in bold: "Matching Expert Distributions for Autonomous Learning (MEDAL)". Beneath, in italics: "*Modified reward* $r_b(\mathbf{s}, \mathbf{a})$**:** learned reward for matching state distribution in demos." ("Modified reward" in italics and the reward symbol in light blue).

On the left is a diagram with $\rho_0$ (blue circle) and $\rho_g$ (green circle) joined by the pale-purple band labelled $\rho\ast$; a pink-purple curve $\pi_f$ ("Forward Policy") goes down from $\rho_0$ and wanders to the middle-bottom of the band; another pink-purple curve rises from there to $\rho_g$; a blue curve $\pi_b$ ("Backward Policy") runs inside the band from the lower middle up to $\rho_g$.

On the right:

0. Given a small set of demonstrations (in teal)
1. Initialize forward policy $\pi_f$ and backward policy $\pi_b$
2. Repeat:
   1. Roll out $\pi_f$ for $H$ steps
   2. Update $\pi_f$ using $r_f(\mathbf{s}, \mathbf{a})$
   3. Roll out $\pi_b$ for $H$ steps
   4. Update $\pi_b$ using $r_b(\mathbf{s}, \mathbf{a})$
   5. *No reset or very infrequent reset*
   6. Update learned reward $r_b(\mathbf{s}, \mathbf{a})$ (in teal)

Then "**Test time:** Discard $\pi_b$. Deploy $\pi_f$".

## Slide 23 — Autonomous Learning via MEDAL

![Slide 23 — Autonomous Learning via MEDAL](../images/16-autonomous-rl/slide-23.jpg)

Title "Autonomous Learning via MEDAL". Footnote at bottom left: "[1] Kakade & Langford. *Approximately Optimal Approximate Reinforcement Learning*. ICML 2002."

In the centre, a diagram headed in bold "Matching Expert Distributions for Autonomous Learning": $\rho_0$ and $\rho_g$ joined by the pale-purple band labelled $\rho\ast$; a pink-purple curve $\pi_f$ ("Forward Policy") goes down from $\rho_0$ and ends mid-bottom; a blue curve $\pi_b$ ("Backward Policy") goes from there up through two dark-blue dots on the band to $\rho_g$. A thin green line runs diagonally across the picture from the upper left to the lower right through the dots. At the right is a black database (stack of cylinders) icon captioned "demonstrations".

Annotations: at the left, "Addressing challenge 2: backward policy avoids collapse of state distribution"; at the lower right, "Addressing challenge 1: agent doesn't drift away". Beneath the diagram: "Pro: Forward policy tries the task from wide set of initial states, both easy and hard, improving the sample efficiency [1]".

## Slide 24 — Forward backward results

![Slide 24 — Forward backward results](../images/16-autonomous-rl/slide-24.jpg)

Title "Forward backward results". Left: "**EARL Benchmark**" then "**Training**: reset every 200k steps" and "**Evaluation**: policy performance from $\rho_0$" (bold "EARL Benchmark", "Training", "Evaluation"). Beneath it is a rendered image of a simulated tabletop, viewed from above: a wooden table with a dark red border, with a red object at the left, a brown mug at the right, a small black disc, and three small square white and grey pads (one at the top, one at the left, one at the bottom).

Right is a line chart. The y axis "Deployed Policy Evaluation" has ticks 0.2, 0.4, 0.6, 0.8, 1.0; the x axis has ticks 0.0e0, 5.0e5, 1.0e6, 1.5e6, 2.0e6, 2.5e6 (no x title). There are six series, each with a shaded band, labelled by coloured text with arrows. All start at 0 and begin to rise from about 3e5 to 1e6. Approximate values at 1.0e6, 1.5e6, 2.0e6 and 2.5e6:

- "VaPRL (goal curriculum)", in green: about 0.92, 0.93, 0.94 and 0.95 (rises first, reaching about 0.9 by 1.0e6, dips slightly and flattens near 0.95).
- "MEDAL", in thick blue: about 0.35, 0.92, 0.94 and 0.96 (rises steeply between 1.0e6 and 1.5e6 and ends highest).
- "FBRL (reach $\rho_0$)", in orange: about 0.5, 0.72, 0.83 and 0.9.
- "R3L (state novelty)", in magenta: about 0.1, 0.4, 0.7 and 0.9 (the latest to rise).
- "oracle SAC (episodic)", in grey dashed: about 0.7, 0.78, 0.8 and 0.82.
- "SAC (non-episodic, no backward policy)", in teal: about 0.3, 0.47, 0.5 and 0.35 (rises to about 0.5 then declines, with a very wide band).

References at the bottom: "EARL: Sharma\*, Xu\* et al. Autonomous Reinforcement Learning: Formalism and Benchmarking, ICLR 2022."; "VaPRL: Sharma et al. *Autonomous Reinforcement Learning via Subgoal Curricula*. NeurIPS 2021."; "FBRL: Han et al. Learning Compound Multi-Step Controllers under Unknown Dynamics. IROS 2015."; "R3L: Zhu et al. The Ingredients of Real-World Robotic Reinforcement Learning. ICLR 2020."

## Slide 25 — Forward backward results

![Slide 25 — Forward backward results](../images/16-autonomous-rl/slide-25.jpg)

Title "Forward backward results". Left: the same "**EARL Benchmark**", "**Training**: reset every 200k steps", "**Evaluation**: policy performance from $\rho_0$" text as slide 24. Beneath is a rendered image of a red robot arm in a simulated workspace, reaching into a wooden tray with a black box standing on it, captioned "Door Closing".

Right is a line chart. The y axis "Deployed Policy Evaluation" has ticks 0.2, 0.4, 0.6, 0.8, 1.0; the x axis has ticks 0.0e0, 8.0e5, 1.6e6, 2.4e6, 3.2e6, 4.0e6. There are six series (five labelled), with bands. Approximate values at 8.0e5, 1.6e6, 2.4e6 and 4.0e6:

- "MEDAL", in thick blue: about 0.75, 1.0, 1.0 and 1.0 (rises steeply from about 5e5).
- "oracle SAC (episodic)", in grey dashed: about 0.9, 1.0, 1.0 and 1.0 (rises earliest, a little ahead of MEDAL).
- "FBRL (reach $\rho_0$)", in orange: about 0.1, 0.45, 0.75 and 0.95.
- "R3L (state novelty)", in magenta: about 0.05, 0.25, 0.4 and 0.5.
- "SAC (non-episodic, no backward policy)", in teal: about 0.15, 0.13, 0.17 and 0.02 (never above about 0.2, ending near 0).
- An unlabelled green series: about 0.4, 0.6, 0.65 and 0.63 (no label on this slide; the same green is "VaPRL" on slide 24).

References: "EARL: Sharma\*, Xu\* et al. Autonomous Reinforcement Learning: Formalism and Benchmarking, ICLR 2022."; "FBRL: Han et al. Learning Compound Multi-Step Controllers under Unknown Dynamics. IROS 2015."; "R3L: Zhu et al. The Ingredients of Real-World Robotic Reinforcement Learning. ICLR 2020."

## Slide 26 — Forward-backward RL on real robots

![Slide 26 — Forward-backward RL on real robots](../images/16-autonomous-rl/slide-26.jpg)

Title "Forward-backward RL on real robots". Two photographs (may be video frames).

- Left, in a black frame: a low-resolution view from above of a silver metal tray on a light wooden table, with a few small packages (white, green and red wrapped items) lying in it. A robot gripper hangs at the top above a small brown card. Caption: "Vanilla FBRL" and "Demo initialized, VLM-provided rewards". (may be a video frame)
- Right: a dark enclosure with a grey surface, a crumpled pale yellow-green cloth on it, a round metal base with an upright metal hook stand, and a white robot arm at the upper right. Caption: "MEDAL" and "Demo initialized, learned rewards". (may be a video frame)

In the lecture (≈39:23–40:55) these are time-lapse videos of the algorithms on real robots. On the left, vanilla forward-backward RL learns a forward and a backward policy; both make mistakes, and it improves through a fully autonomous process. On the right is MEDAL, where, for example, when the forward policy misses, the backward policy does not have much to do. In both, the rewards came from models rather than from humans, so the whole process is autonomous and can run overnight.

## Slide 27 — Alternatives to Forward-Backward RL?

The same list as slide 20, with item 2, "Practice *cycle* of different tasks", in bold.

## Slide 28 — Autonomous RL with task cycles

![Slide 28 — Autonomous RL with task cycles](../images/16-autonomous-rl/slide-28.jpg)

Title "Autonomous RL with task cycles". Printed slide number "28" at the bottom right.

On the left are black pictogram icons: a coffee machine with a round button, a cup being moved under its spout by a grey curved arrow from one cup position to another, and below it a person figure crossed out by a large red X (no person involved). In the centre is a graph of four green circle nodes: "Task 1" at the top, "Task 2" at the left, "Task 3" at the right and "Task 4" at the bottom. Black arrows: from Task 1 to Task 3 labelled "succeed"; from Task 1 to Task 2 labelled "fail"; from Task 2 a loop to itself and an arrow back to Task 1; from Task 3 to Task 4 labelled "fail"; from Task 3 back to Task 1; from Task 4 back to Task 1.

On the right, with the labels in bold: "Task 1: put cup in coffee machine", "Task 2: pick up cup", "Task 3: replace cup", "Task 4: clean up spill from cup…".

## Slide 29 — Algorithm: Autonomous RL with task cycles

Title "Algorithm: Autonomous RL with task cycles". On the left is the same task-cycle graph as slide 28 (Task 1 to Task 4 with "succeed" and "fail" arrows) without the icons. On the right:

1. Initialize task-conditioned policy $\pi(\cdot \mid \mathbf{s}, \mathbf{z})$
2. Repeat:
   1. Propose task $\mathbf{z}_ i$ to practice based on $\mathbf{s}$
   2. Roll out $\pi(\cdot \mid \mathbf{s}, \mathbf{z}_ i)$ for $H$ steps
   3. Update $\pi$ using $r(\mathbf{s}, \mathbf{a}, \mathbf{z})$
   4. *No reset or very infrequent reset*

(The policy is in pink-purple, $\mathbf{z}_ i$ in teal, the reward in orange. The printed conditioning bar in the policy is a vertical bar, shown here as $\mid$.) Then "**Test time:** Deploy $\pi$".

## Slide 30 — Example: autonomous RL with task cycles

![Slide 30 — Example: autonomous RL with task cycles](../images/16-autonomous-rl/slide-30.jpg)

Title "Example: autonomous RL with task cycles". Printed slide number "30" at the bottom right. Citation at the bottom left: "Gupta, Yu, Zhao, Kumar, Rovinsky, Xu, Devlin, Levine. **Reset-Free Reinforcement Learning via Multi-Task Learning: Learning Dexterous Manipulation Behaviors without Human Intervention**. 2021." (the title in bold).

On the left is a photograph of a black multi-fingered robotic hand on a red robot arm over a wooden tray with a white three-segment rounded object (a white three-lobed valve-like object); a wall clock is in the lower left corner (may be a video frame). On the right is a diagram of four small photographs in a square, each titled in its corner: "Re-center" (top left), "In-Hand" (top right), "Lift" (bottom left), "Flip" (bottom right). Purple arrows join them: from In-Hand to Re-center (leftward), between Re-center and Lift (double-headed, vertical), from Lift to Flip (rightward), from Flip to In-Hand (upward), and a diagonal arrow from Flip to Re-center. A small legend at the right reads "Reset" beside a dark purple arrow and "Train" beside a hollow arrow. Each photograph has a small circular reset icon at one corner, and pink arrows within photographs show the motion.

In the lecture (≈45:35–47:09) the left photograph is a video: the robot must pick up the object, flip it over and rotate it. At first it cannot flip or rotate it and mostly practises picking it up and putting it back down; once it is better at picking it up, it starts to flip and rotate it, practising whatever makes sense from the state it is in.

## Slide 31 — Example: autonomous RL with task cycles

![Slide 31 — Example: autonomous RL with task cycles](../images/16-autonomous-rl/slide-31.jpg)

Title "Example: autonomous RL with task cycles". Two photographs, each with a citation (both may be video frames).

- Left: a small black quadruped robot walking on a lawn in a garden, with a fence and shrubs with orange flowers behind it, and a thin white line (a trail) drawn on the grass behind the robot. Overlay text at the upper left: "Initial Pre-Trained Policy" and "5x speed"; at the lower left: "Skill: Backward Pace". Caption: "Laura Smith, J. Chase Kew, Xue Bin Peng, Sehoon Ha, Jie Tan, Sergey Levine. **Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World**. 2021." (the title in bold). (may be a video frame)
- Right: a white two-armed PR2 robot (label "PR2" on its torso) standing in front of a green backdrop at a wooden table, with a small blue box with a lid on the table and a blue wrench-like object. Overlay text: "7-stage wrench task" at the top right, "forward controller 1" at the right, "1x real time" at the lower left and "autonomous execution" at the lower right. Caption: "Han, Levine, Abbeel. **Learning Compound Multi-Step Controllers under Unknown Dynamics.**" (the title in bold). (may be a video frame)

In the lecture (≈47:09–47:55) both are videos. On the left the robot learns to walk forward and backward and, at the same time, to get back up, switching between going forward and going backward: when it has gone too far back it practises going forward. On the right is a seven-stage task (pick the wrench up, position it, screw repeatedly), running different controllers depending on the state it is in, and eventually a backward policy that puts the wrench back on the table.

## Slide 32 — How to identify which task to collect data for?

![Slide 32 — How to identify which task to collect data for?](../images/16-autonomous-rl/slide-32.jpg)

Title "How to identify which task to collect data for?". Printed slide number "32" at the bottom right. Below the title, two phrases side by side: "In general: an open research question" and "One idea: Ask a vision language model.". Citation at the bottom left: "Zhou\*, Atreya\*, Lee, Walke, Mees, Levine. **SOAR: Autonomous Improvement of Instruction Following Skills via Foundation Models.**" (the title in bold).

In the middle is a large white frame (a video, shown as one frame): the left part holds a small photograph of a robot's workspace, a toy microwave with an open door at the upper left, a silver bowl holding a yellow banana-like object and a purple object, a dark wooden table, and a black gripper at the top. Overlaid on it, the word "Observation". Beneath the photograph, the text "initial frame of a trajectory." The rest of the white frame, to the right, is empty. (may be a video frame; what the video showed is not on this page)

In the lecture (≈48:42–51:06) Finn talks over this figure, which most likely plays as a video or animation (the page holds only its first frame, and its right part is empty): from this frame there are things the robot can do (open the toy microwave, pick up the banana, push the eggplant, move the pot) and things it cannot (pick up a carrot, which is not in the scene). A vision-language model is asked what could be done in this environment and predicts, for example, "take the banana out of the silver pot", and the robot practises that task. The work then feeds the instruction into an image-editing diffusion model to generate a goal image and trains the policy to reach it.

## Slide 33 — Alternatives to Forward-Backward RL?

![Slide 33 — Alternatives to Forward-Backward RL?](../images/16-autonomous-rl/slide-33.jpg)

Title "Alternatives to Forward-Backward RL?". The list reads: "1. Learn to reset to expert state distribution (forward-backward with modified backward reward)" (the parenthesis in grey); "2. Practice cycle of different tasks" (no italics here); and, in bold, "3. Set up task that can be done repetitively ("forward-forward RL"?)" (the parenthesis in bold grey).

Beneath the list is a video frame (may be a video frame): an overhead view of two black robot arms with red glowing light strips in an office-like workspace with a light wooden table, a white sheet or cloth on it, a white shelf or wall behind with red scribbles, and a clock on a stand at the upper right. Overlaid text at the lower left: "DYNA-1 autonomous, un-cut 24 hours"; at the lower right: "DYNA" in spaced letters. Source line beneath: "Source: x.com/JasonMa2020/status/1917255026751553629".

In the lecture (≈52:39) this is a video of a task set up to be done repeatedly: folding napkins, for businesses that need lots of napkins folded.

## Slide 34 — Plan for Today

The same outline as slide 3 without the Goal line, with the third item "- Developing the algorithms" and its second sub-item "- Single life RL" in black; the others, including "- Learning policies without human intervention", are greyed out.

## Slide 35 — So far…

Title "So far…". Text: "We've looked at learning policies without resets:"; "- Can improve the quality of the policy through *autonomous practice*" (the last two words in italics); "What happens after we deploy the policy?"; "*Obviously*, everything works perfectly and nothing ever goes wrong." followed by a green check mark ("Obviously" in italics).

## Slide 36 — So far…

The same as slide 35, except the line "*Obviously*, everything works perfectly and nothing ever goes wrong." is struck through and the check mark is gone, and a new line appears beneath: "The natural world is complex, and something likely will go wrong, despite preparation."

## Slide 37 — Even humans make mistakes and adapt

Title "Even humans make mistakes and adapt". A close-up photograph of a glass door in a metal frame: a silver door handle at the top and, beneath it, a keyhole in a lock plate, with a blurred reflection of a street and a bicycle in the glass. No person is in the frame. This is the first frame of the video that plays on slide 39.

## Slide 38 — Even humans make mistakes and adapt

The same page as slide 37: the title and the same photograph, with no change that is visible in the export (a build step of slide 39).

## Slide 39 — Even humans make mistakes and adapt

![Slide 39 — Even humans make mistakes and adapt](../images/16-autonomous-rl/slide-39.jpg)

The same photograph and title, now with a hand coming in from the right holding a ring of keys, one key inserted in the keyhole. The hand wears a dark band bracelet on its wrist. (may be a video frame)

In the lecture (≈54:13–54:59) this is a video of a person opening the door, played twice. The person makes a mistake and adapts in a split second: pushing, they find the key is not correctly in the hole, draw it back out and put it back into the keyhole.

## Slide 40 — Single-Life Reinforcement Learning (SLRL)

![Slide 40 — Single-Life Reinforcement Learning (SLRL)](../images/16-autonomous-rl/slide-40.jpg)

Title "Single-Life Reinforcement Learning (SLRL)". Citation at the bottom left: "Chen, Sharma, Levine, Finn. *You Only Live Once: Single-Life Reinforcement Learning*, NeurIPS '22" (the title in italics).

Three panels compare settings, each with a blue running-person icon and an orange star for the goal, joined by grey dotted path arrows.

- "Episodic reinforcement learning" (left; a blue rounded box labelled "train" on its side, and a green box labelled "test" below it). In the train box, two rows each hold a person and a star; the first row's dotted path makes a loop and ends with an arrow pointing back down before reaching the star, and the second row's path makes a loop and then reaches the star. Three dots below the second row stand for more episodes. In the test box, one person's dotted path goes straight to the star.
- "Reset-free reinforcement learning" (middle; the same "train" blue box and "test" green box). In the train box there is a single person and star, with many overlapping dotted paths, some faded, looping around and back to the start and ending at the star. The test box shows one person with one path to the star.
- "Single life reinforcement learning" (right; one teal box). A database icon labelled "prior training data" at the upper left; a person at the lower left with one long dotted path that loops and winds many times to end at the star at the right, with an arrow at the end.

Text at the bottom: "Given prior data in train env, agent has one "life" to autonomously complete task in novel scenario." ("one "life"" in blue) and "Can we run RL in a single episode at test time?". At the lower right are two small rendered images: "training experience", a pink-brown legged walker on a green and white checkerboard floor, and "test environment", the same walker on the same floor with a blue upright block at its right.

## Slide 41 — Single-Life Reinforcement Learning (SLRL)

Title "Single-Life Reinforcement Learning (SLRL)". Same citation as slide 40: "Chen, Sharma, Levine, Finn. *You Only Live Once: Single-Life Reinforcement Learning*, NeurIPS '22".

On the left is the single-life panel from slide 40, headed in bold "Single life reinforcement learning": a teal box with the "prior training data" database icon, a person at the lower left and a long looping dotted path to an orange star. On the right: "Given prior data in train env(s), agent has one "life" to autonomously complete task in **novel, OOD scenario**." ("one "life"" in teal; "novel, OOD scenario" in bold). Beneath, the two rendered images as on slide 40, "training experience" and "test environment" (the walker on the checkerboard, the second with a blue block), and the line "Can we fine-tune with RL in a single episode at test time?".

## Slide 42 — How to approach single-life RL?

![Slide 42 — How to approach single-life RL?](../images/16-autonomous-rl/slide-42.jpg)

Title "How to approach single-life RL?". Citation at the bottom left: "Chen, Sharma, Levine, Finn. *You Only Live Once: Single-Life Reinforcement Learning*, NeurIPS '22".

Text: "Can we fine-tune with vanilla reinforcement learning, but at test time?". Beneath it, a wide dark teal-grey frame (one frame of a video) with a small stick-figure legged walker in orange and pink standing near the centre, with a pale blue reflection of it below it (may be a video frame). Under the frame: "Agent fails to handle or recover from out-of-distribution states without interventions."

In the lecture (≈56:35–57:20) this is a video of a policy fine-tuned with reinforcement learning at test time, on a course with obstacles. It gets over the first obstacle, which it has never seen, but then gets stuck flipped over; though it is still running reinforcement learning updates and is somewhat trying to get up, it cannot recover without interventions.

## Slide 43 — How to approach single-life RL?

![Slide 43 — How to approach single-life RL?](../images/16-autonomous-rl/slide-43.jpg)

Title "How to approach single-life RL?". Text: "A few useful ideas:"; "- Guide the learning process towards good, familiar states to avoid getting stuck"; "- Leverage pre-training"; two indented sub-bullets "- Pre-trained skills can allow for adaptation in *skill* space rather than in action space" (the word "skill" in italics) and "- Pre-trained LLMs may be able to reason through how to adapt". Then "Example of guiding policy towards good states in prior data:" above a dark teal-grey video frame with the same small orange and pink walker and its pale reflection (may be a video frame). The frame is overlapped by a white box behind the end of the sentence. Citation below: "Chen, Sharma, Levine, Finn. *You Only Live Once: Single-Life Reinforcement Learning*, NeurIPS '22".

In the lecture (≈1:00:25) the video shows the agent, guided towards familiar states rather than towards the goal, on the same example: it still gets stuck on its back, but by trying to get back to a familiar state, such as one on its feet, it gets back up after a few attempts and reaches the goal.

## Slide 44 — How to approach single-life RL?

![Slide 44 — How to approach single-life RL?](../images/16-autonomous-rl/slide-44.jpg)

Title "How to approach single-life RL?". Two photographs under headings (both may be video frames).

- Left, heading "Adaptation in low-level action space" ("low-level" in orange): a black quadruped robot at the right of a living-room scene with a stone fireplace and bookshelves behind it, walking over a pile of cardboard boards on a grey foam-tile floor. Caption: "Kumar, Fu, Pathak, Malik. RMA: Rapid Motor Adaptation for Legged Robots. RSS 2021". (may be a video frame)
- Right, heading "Adaptation in high-level skill space" ("high-level" in teal): a view from low down on a dark carpeted floor among orange, red patterned and pale green soft chairs and cushions, with a red object on the floor. Overlay text: "Left Small" at the top left and "15x Speed" at the bottom left. A small inset at the upper right shows a camera view of the legs of a person in dark clothing near an orange cushion. Caption: "Chen, Lessing, Tang, Chada, Smith, Levine, Finn. Commonsense Reasoning for Legged Robot Adaptation with Vision-Language Models. ICRA 2025". (may be a video frame)

In the lecture (≈1:00:25–1:02:00) both are videos. On the left, a system trained to adapt in the low-level space from recent history handles small out-of-distribution changes, such as different terrains. On the right, a legged robot has to reach a goal; a policy run without context keeps trying to go forward, while one that tries different strategies (moving left, moving backwards) with context and adaptation problem-solves its way through obstacles it has never seen. Answering a question (≈1:04:20), Finn adds that this work used a vision-language model to indicate, from past experience, which strategies are good to try.

## Slide 45 — Summary

Title "Summary". Outline: "- Why aren't robots autonomous already?"; "- Defining the problem: autonomous RL"; "- Developing the algorithms"; and two indented sub-items "- Learning policies without human intervention" and "- Single life RL".

## Slide 46 — Reminders

Title "Reminders" (centred). Two centred lines: "Project milestone due tonight" and "Homework 4 due next Friday". Below, the heading "Next time" and the lines "Guest lecture from Ashish Kumar" and "(Tesla Optimus)". At the lower right is a portrait photograph of Ashish Kumar (named by the text beside it, not by a caption): a person with short dark hair and a beard, smiling, in a dark jacket over a white open-collared shirt, with blurred lights behind.
