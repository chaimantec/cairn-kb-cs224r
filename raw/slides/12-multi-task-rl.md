---
title: "Lecture 12 — Multi-Task RL (slide deck)"
lecture: 12
slides: 29
printed_numbers: 1–29
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf
numbering: 1:1 with PDF pages; every page prints its number, bottom centre on most pages and bottom right on pages 12–19, 21 and 23 (slide_number_map.py misses the bottom-right numbers, takes a stray "1" in page 14's content for that page's number, and so reports gaps that do not exist; the raw spans show every page's number)
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (6, 14, 19, 20, 24, 27) were checked by Claude Opus 5.5 against 1400px renders, with slide 27's chart and slide 18's equations checked on crops. Slides 6 and 20 agree, and slide 27's series count, colours and legend agree. Corrections: slide 27's values at epoch 50 had been read about 10 epochs early (pushing's red curve is about 55% at 50 and 78% at 60; pick-and-place's curves reach about 95% at 60, not by 30–50; its red curve dips at 140, not 150), and two sliding values were re-read; slide 19's two encoder arrows both end at the blue dots on the embedding sphere, which the transcription had split into a video side and a language side; slide 24's large silhouette was said to stand for the player in either task, which the slide does not say; slide 14's mannequin has both arms at its sides.
images: 9 slides are rendered and shown under their headings (6, 14, 17, 19, 20, 21, 22, 24, 27). Not rendered: slide 5, a build step whose content slide 6 shows; equation, algorithm and labelled-formula slides whose arrows, highlights, colours and labels are transcribed here (7, 9, 13, 15, 16, 18, 23, 25, 26); and the title, reminder, agenda and text slides (1, 2, 3, 4, 8, 10, 11, 12, 28, 29).
---

# Lecture 12 — Multi-Task RL: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 12 deck, "Model-Based RL and Multi-Task and Goal-Conditioned Reinforcement Learning", transcribed from the PDF. The title slide (slide 1) prints "Multi-Task and Goal-Conditioned Reinforcement Learning" and "CS 224R", with the words "Model-Based RL and" added above it in pink (see slide 1), and names no lecturer. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

The recording (about 70 minutes) opens by stating the plan (slide 3) and recapping planning (slide 4), without discussing slide 2's reminders. It follows the deck through slide 25, Finn's "lastly is the goal-conditioned version" (≈1:09:21) begins slide 26, and the captions end at about 1:10:21, partway through it. So slide 26's relabeling with the last state, its other strategies and its result line, slide 27's example and chart, and slides 28–29 are not discussed: for them this file is the only source. Finn works several things on a board the recording does not show: the model-based policy optimization algorithm, with its generated data called D gen (≈7:47–9:20); the aggregated MDP's reward and dynamics (≈38:31–40:48); a one-hot vector (≈43:55); the multi-task imitation objective (≈44:40–46:11); and the relabeling algorithm (≈1:00:54–1:02:26). The transcript is the only source for those.

## Slide numbers vs PDF pages

The deck is 1:1 with its printed numbers: PDF page N is printed slide N, for all 29 pages. Every page prints its number: at bottom centre on most pages, and at bottom right on pages 12–19, 21 and 23.

| PDF page | Printed slide |
| -------- | ------------- |
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

## Sections

| Section | Slides |
| ------- | ------ |
| Title, course reminders and the plan | 1–3 |
| Model-based RL: recap of planning, using a learned model via synthetic data generation (model-based policy optimization), when to use model-based RL, other kinds of models | 4–9 |
| Plan (multi-task section highlighted) | 10 |
| Multi-task imitation and RL: what multi-task RL is, why, formalizing tasks, task identifiers, task conditioning for imitation learning and in practice, multi-task RL algorithms | 11–23 |
| Goal-conditioned RL and hindsight relabeling (the hockey example, relabeling algorithm, the goal-conditioned version, simulated-manipulation results); goal-conditioned RL is first introduced on slide 17 | 24–27 |
| Plan (closing) and reminders, next time | 28–29 |

## Slide 1 — Multi-Task and Goal-Conditioned Reinforcement Learning

Title slide. Top left, in pink: "Model-Based RL and". Below and to its right, a small pink caret "^" sits under the end of "and", an insertion mark pointing at the black title that follows. The black centred title reads "Multi-Task and Goal-Conditioned Reinforcement Learning" over two lines, with "CS 224R" below it. Read with the insertion, the title is "Model-Based RL and Multi-Task and Goal-Conditioned Reinforcement Learning". No lecturer or affiliation is printed.

## Slide 2 — Course reminders

- Homework 3 due next Friday, May 16
- Project milestone due Friday, May 23

## Slide 3 — The plan for today

Agenda, numbered list:

1. Model-based reinforcement learning
   - a. Using a learned model via synthetic data generation
   - b. When to use model-based RL
2. Multi-task imitation and reinforcement learning
   - a. Task conditioning
   - b. Goal-reaching tasks
   - c. Hindsight relabeling

A large right brace spans items 2a–2c, labelled in bold "Part of homework 4!".

## Slide 4 — Recap: Planning with learned models

A numbered list, then a pros and cons pair, then a question.

1. Can *plan* $\mathbf{a}_ 1, \ldots, \mathbf{a}_ H$ with gradient-based (blue) or sampling-based (magenta) optimization
2. *Update the model* using data collected with planning
3. *Replan* periodically to help account for mistakes.

Two columns below. Left, in green: "+ Simple"; "+ Easy to plug in different goals / rewards (possibly even at test time!)". Right, in red: "- Compute intensive at test time"; "- Only practical for short-horizon problems (or very shaped reward functions)".

"Why only short horizons?" with two answers: "(a) too compute expensive to make long plans" and "(b) model is not accurate for long horizons".

Closing line at the bottom: "Can we *train a policy* using a learned model?"

## Slide 5 — Model-based policy optimization

Subtitle: "Key idea: augment data with model-simulated roll-outs."

Left, captioned "Example real trajectory": six dots joined by black arrows in a rising chain from lower left to upper right, labelled $\mathbf{s}_ 1$ (blue dot, at the bottom left) then $\mathbf{s}_ 2$, $\mathbf{s}_ 3$, $\mathbf{s}_ 4$, $\mathbf{s}_ 5$, $\mathbf{s}_ 6$ (teal dots). The labels are printed as $s_1$ to $s_6$ in plain (non-bold) type.

Right, "How to augment?" with a list in which "initial states" is blue and "partial trajectories" is magenta italic:

- generate full trajectories from initial states?
  - model may not be accurate for long horizons
- generate *partial trajectories* from initial states?
  - may not get good coverage of later states

This is the first of two builds; the next slide adds the augmented data to the diagram and a third option.

## Slide 6 — Model-based policy optimization

![Slide 6 — Model-based policy optimization](../images/12-multi-task-rl/slide-6.png)

Build of slide 5 with the same title, subtitle, real trajectory and first two options. It adds, on the diagram, short magenta arrows leaving real states to pale pink dots (the model-simulated states), captioned "Augmented data" in magenta at the bottom. From $s_1$ a magenta arrow goes to a pink dot, then on to a second pink dot. From $s_3$ a magenta arrow goes up to a pink dot, which is joined by a further arrow to a second pink dot near the top. From $s_4$ a magenta arrow goes right to a pink dot, then another arrow up to a second one. So each short pink branch starts at a real state and runs about two model steps.

The list gains a third option: "generate *partial trajectories* from *all states* in the data" ("all states" in teal), followed by a light-bulb emoji at the right edge.

## Slide 7 — Model-based policy optimization

Subtitle: "Key idea: augment data with model-simulated roll-outs."

"Full algorithm", a four-step loop; a thick grey curved arrow on the left runs from the end of step 4 back up to step 1.

1. Collect data using current policy $\pi_\theta$, add to $D_{env}$
2. Update model $p_\psi(s' \mid s, a)$ using $D_{env}$
3. Collect short synthetic roll-outs using $\pi_\theta$ in model $p_\psi$ from states in $D_{env}$; add to $D_{model}$
4. Update policy $\pi_\theta$ (and critic $Q_\phi$) using $D_{env} \cup D_{model}$

"Notes:"

- compatible with variety of model-free RL methods (step 4)
- could additionally use $D_{env}$ in policy update

The slide prints $s$, $a$, $s'$ in plain italic here, not bold; step 3's line runs to the right edge of the page.

## Slide 8 — When to use model-based RL?

Subtitle in bold: "Big upsides and big downsides".

In green:

- \+ Immensely useful, far more data efficient if model is easy to learn
- \+ Model can be trained on data without reward labels (fully self-supervised)
- \+ Model is somewhat task-agnostic (can sometimes be transferred across rewards)

In dark red:

- \- Models don't optimize for task performance
- \- Sometimes harder to learn than a policy
- \- Another thing to train, more hyperparameters, more compute intensive

Closing line: "Whether to use a model depends on how hard it is to learn!"

## Slide 9 — Other kinds of models

"So far: modeling $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$"

"Many alternatives":

- inverse model $p(\mathbf{a}_ t \mid \mathbf{s}_ t, \mathbf{s}_ {t+1})$
- multi-step inverse model $p(\mathbf{a}_ t \mid \mathbf{s}_ t, \mathbf{s}_ {t+n})$ or $p(\mathbf{a}_ {t:t+n} \mid \mathbf{s}_ t, \mathbf{s}_ {t+n})$
- future prediction without actions $p(\mathbf{s}_ {t+1:t+n} \mid \mathbf{s}_ t)$
- video interpolation $p(\mathbf{s}_ {t+1:t+n} \mid \mathbf{s}_ t, s_{t+n+1})$
- transition distribution $p(\mathbf{s}_ t, \mathbf{a}_ t, \mathbf{s}_ {t+1})$

Closing line: "These kinds of models can be useful too! Have various purposes."

The video-interpolation formula prints its last conditioning state as a non-bold $s_{t+n+1}$ while the others are bold.

## Slide 10 — The plan for today

The agenda of slide 3 repeated, with item "2. Multi-task imitation and reinforcement learning" in bold to mark the new section. The brace and "Part of homework 4!" label beside 2a–2c are unchanged. Items 1, 1a and 1b are in regular weight.

## Slide 11 — What is multi-task RL?

Subtitle: "Can we train generalist policies to do many "tasks", not just one?"

"Examples:" followed by five lines, each with a bold-leading category:

- Generalist LLM assistant: travel booking, grocery shopping, ...
- Legged robot: walk, run, dance, crouch, ...
- Mobile manipulator: hang up a towel, unload a dishwasher, wipe a spill, ...
- Music recommender system: personalization to many distinct users
- Game player: play Flappy bird, Pokemon, Pinball, ...

Then: "May have different reward functions, different dynamics, different action spaces, ...!"

Last line: "Key idea will be to condition on the task." ("condition on the task" underlined).

## Slide 12 — Why multi-task RL?

Subtitle: "Can we train generalist policies to do many "tasks", not just one?"

A pale-green rounded box reads "Fundamentals of CS224r so far:" with a bulleted list: Imitation; On-policy, off-policy and offline RL; Model-free and model-based RL; Reward functions. At the right of the box: "Biggest challenge so far?" and below it, underlined, "Data efficiency".

Below the box: "Can we **amortize the data complexity** across many tasks & scenarios?"

"Plenty of learnings can be shared (blue) across tasks!"

A row of three examples, with the shared item in blue: "LLM assistant: grammar", "Legged robots: balance", "Personal recommenders: user similarities".

At the bottom left, a teal box in bold, "There's also a deeper motivation.", with beside it: "Generalist ML systems are often more reliable, performant than specialists."

## Slide 13 — How do we formalize what different "tasks" are?

Subtitle: "Different tasks can just be different MDPs!"

"A task:" followed by

$$\mathcal{T}_ i \triangleq \lbrace \mathcal{S}_ i, \mathcal{A}_ i, p_i(\mathbf{s}_ 1), p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}), r_i(\mathbf{s}, \mathbf{a}) \rbrace$$

Grey arrows with labels point at the terms: from above, "action space" points at $\mathcal{A}_ i$ and "dynamics" at $p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$; from below, "state space" points at $\mathcal{S}_ i$, "initial state distribution" at $p_i(\mathbf{s}_ 1)$ and "reward" at $r_i(\mathbf{s}, \mathbf{a})$.

Bottom: "**Note**: This allows for much more variety than the semantic meaning of "task"!"

## Slide 14 — A Few Examples Task Distributions

![Slide 14 — A Few Examples Task Distributions](../images/12-multi-task-rl/slide-14.jpg)

Title as printed ("A Few Examples Task Distributions"). Top line repeats the task tuple: "A task: $\mathcal{T}_ i \triangleq \lbrace \mathcal{S}_ i, \mathcal{A}_ i, p_i(\mathbf{s}_ 1), p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}), r_i(\mathbf{s}, \mathbf{a}) \rbrace$". Then "Personalized recommendations: $p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$, $r_i(\mathbf{s}, \mathbf{a})$ vary across tasks" (no picture).

"Character animation:" has two rows.

- "across maneuvers, $r_i(\mathbf{s}, \mathbf{a})$ vary": four filmstrips in a 2 by 2 grid, each of six frames of a grey humanoid figure on a grey floor, captioned beneath in small type: top left "Spinkick", top right "Cartwheel", bottom left "Backflip", bottom right "Run". Each shows the same figure performing that motion across the frames.
- "across garments & initial states, $p_i(\mathbf{s}_ 1)$, $p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$ vary": a pale-blue panel with four simulation frames of a bald orange mannequin figure, arms hanging at its sides, with a blue cloth garment draped over its head and shoulders; the garment's crumple differs slightly between frames.

"Multi-robot RL:" shows three different robot arms side by side (a small black multi-joint arm on a round base, a red arm with a screen face showing eyes, and a white arm with a grey base), captioned with the formula $\mathcal{S}_ i$, $\mathcal{A}_ i$, $p_i(\mathbf{s}_ 1)$, $p_i(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$ followed by the word "vary".

The slide has no source credits for the images.

## Slide 15 — How do we formalize what different "tasks" are?

Build of slide 13: same subtitle, task tuple and labelled arrows (shifted slightly left), with a new block below.

Bold question: "How can do we identify one task from another?" (the slide prints "can do we", as quoted).

Three lines: "Assign task index (e.g. task 0, task 1, ...)"; "Language description"; "Video illustration of what to do". A brace on their right reads "These are task identifiers, $\mathbf{z}_ i$".

The slide drops slide 13's closing "Note" line.

## Slide 16 — How do we formalize what different "tasks" are?

Build of slide 15 (the slide moves the block down slightly): same subtitle, task tuple and labelled arrows, with the identification block of slide 15 removed and a new line at the bottom.

"**An alternative view:** A task identifier is part of the state: $\mathbf{s} = (\bar{\mathbf{s}}, \mathbf{z}_ i)$", with an arrow from $\bar{\mathbf{s}}$ labelled "original state".

## Slide 17 — The goal of multi-task reinforcement learning

![Slide 17 — The goal of multi-task reinforcement learning](../images/12-multi-task-rl/slide-17.jpg)

Title with "multi-task" in teal. A reinforcement-learning loop diagram at the top: at left a bold $\mathbf{s}$ with a black arrow curving up into a convolutional network drawing (a stack of boxes labelled with layer sizes such as 224, 55, 27, 13, with "Max pooling" and "Stride of 4" annotations, ending in "dense 4096", "dense 4096" and a final thin layer labelled "1000" cut off at the edge), captioned $\pi_\theta(\mathbf{a} \mid \mathbf{s})$. An arrow labelled bold $\mathbf{a}$ leaves the network and points at a picture of the Earth globe, captioned $p(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$. A long black arrow runs from under the globe back along the bottom to the $\mathbf{s}$ at the left. The picture is the standard policy-and-environment loop; the network drawing is a generic image-classifier network standing for the policy.

Left block, bold heading "Multi-task RL":

"The same as before, except: a task identifier is part of the state: $\mathbf{s} = (\bar{\mathbf{s}}, \mathbf{z}_ i)$"

"e.g. one-hot task ID; language description; desired goal state, $\mathbf{z}_ i = \mathbf{s}_ g$" with a left arrow from that line to "goal-conditioned RL" (with "goal-conditioned" in magenta, in quotation marks).

Right block, bold heading "What is the reward?":

"The same as before". "Or, for goal-conditioned RL:" (magenta words "goal-conditioned") followed by

$$r(\mathbf{s}) = r(\bar{\mathbf{s}}, \mathbf{s}_ g) = -d(\bar{\mathbf{s}}, \mathbf{s}_ g)$$

"Distance function $d$ examples:" with two dashed items, "Euclidean $\ell_2$" and "sparse 0/1".

Bottom line: "If it's still a standard Markov decision process (grey words), then, why not apply standard RL algorithms (grey words)? You can! You can do better in some cases."

## Slide 18 — Let's start with multi-task imitation learning

"Single task imitation learning":

$$\min_ \theta \thinspace -E_ {(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}} \log \pi_\theta(\mathbf{a} \mid \mathbf{s})$$

with the expectation-and-log part highlighted in a light-blue box. A second line, offset to the right, shows $\min_ \theta \thinspace \mathcal{L}(\theta, \mathcal{D})$ with the loss $\mathcal{L}(\theta, \mathcal{D})$ highlighted in the same blue, so the blue box on the first line is being abbreviated as $\mathcal{L}$. A black arrow leads from it to

$$\min_ \theta \sum_ {i=1}^{T} \mathcal{L}(\theta, \mathcal{D}_ i)$$

captioned "Multi-task imitation learning" (the image of this formula has a slightly noisy background). The sum runs to $T$; in the recording Finn speaks of $n$ tasks, with $i$ running from 1 to $n$ (≈45:26).

Below: "One useful trick from multi-task supervised learning: **stratified sampling**". "Construct each minibatch with data from each task." "-> lower variance gradients".

## Slide 19 — How to condition on the task in practice?

![Slide 19 — How to condition on the task in practice?](../images/12-multi-task-rl/slide-19.jpg)

Citation at bottom left: "Jang et al. BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning CoRL 2021."

A network diagram reproduced from that paper. At left, an "RGB image (512, 640, 3)" photo of a robot arm over a wooden table with a bowl and objects, with an arrow labelled "Random crop (450, 450), Downsample (150, 150), Random photometric distortion" to "Preprocessed Images (150, 150, 3)". These feed a "Vision Network (ResNet-18)" drawn as two blue stacked-block layers, then "Mean Pool" (a tall column of cells), whose output fans out to three heads. Each head is a green-arrow chain of "fully connected ReLU (256)" columns, then a linear layer: "linear (3N)" feeding "Delta Tool XYZ sequence (residual)" with "Huber Loss"; "linear (4N)" feeding "Delta Axis-angle rotation sequence" with "Huber Loss"; "linear (N) sigmoid" feeding "Gripper Angle sequence" with "Log Loss". The three heads' output cells form the "commanded action", and a red box around the first column of each marks "actions taken by robot". Legend: blue square "network activations", green line "policy parameters", orange-red line "supervision signal", red square outline "actions taken by robot".

Below the image input, the task conditioning: a "Human Demo Video" still (a hand with objects on a teal cloth) goes through a "Video Encoder" (a blue trapezoid), and the text "Place bottle in the ceramic bowl" goes through a "Language Encoder" (a blue trapezoid). Both arrows point into a sphere, a shared embedding space, and both end at a small cluster of blue dots near its top; three orange dots sit lower on the sphere. The figure does not label what the two colours of dot stand for. From a point on the sphere, green arrows labelled "FiLM-Conditioned Task Embedding" go into the two vision-network layers.

Equation at the bottom:

$$\min \sum_ {\text{task } i} \thinspace \sum_ {(s,a) \sim \mathcal{D}_ e^i, \thinspace w_h \sim \mathcal{D}_ h^i \cup \mathcal{D}_ e^i} \underbrace{-\log \pi(a \mid s, z^i)}_ {\text{behavior cloning}}, \text{ where } \underbrace{z_h^i \sim q(\cdot \mid w_h)}_ {\text{video encoder}}, \underbrace{z_\ell^i \sim q(\cdot \mid w_\ell^i)}_ {\text{language encoder}}$$

The slide prints $s$ and $a$ in plain italics here, not bold, as in the paper's figure. Small text in the original figure is read at the page's resolution; the layer sizes above are as printed.

## Slide 20 — How to condition on the task in practice?

![Slide 20 — How to condition on the task in practice?](../images/12-multi-task-rl/slide-20.jpg)

Subtitle: "Modern policy architectures". Citation at bottom: "Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model. CoRL 2024." (the page number 20 prints just right of it).

Diagram from that paper inside a dashed box titled "OpenVLA". At left, an "Input Image" (a small photo of a blue sink with a yellow dish rack, a silver bowl, a purple object and a black robot gripper) and a "Language Instruction" box reading "Put eggplant in bowl". The image goes to a pair of boxes "DinoV2" (yellow) and "SigLIP" (blue), marked ①, then to an "MLP Projector" (yellow-to-blue gradient), marked ②, producing a row of five yellow-blue tokens. The instruction goes into the prompt "What should the robot do to {task}? A:" which goes through a green "Llama Tokenizer" producing a row of five green tokens. Both token rows feed a large grey box "Llama 2 7B", marked ③. Above it, three purple tokens feed an "Action De-Tokenizer" (purple), whose output goes to a bracketed column of $\Delta x$, $\Delta\theta$ and $\Delta \text{Grip}$, labelled "7D Robot Action", with an arrow down to a photo of a black robot arm at the lower right.

Bottom line: "Task identifier $\mathbf{z}_ i$ passed as prompt into a (fine-tuned) LLM".

## Slide 21 — Example of multi-task imitation learning

![Slide 21 — Example of multi-task imitation learning](../images/12-multi-task-rl/slide-21.jpg)

Citation at bottom left: "Pi Team et al. $\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization arXiv 2025." The number 21 prints at the bottom right.

A frame, appearing to be a video frame, of a kitchen with teal walls and cabinets, a black stove with a yellow kettle, a white checked towel crumpled on the stove top, and a mobile robot with black arms at the right edge. A caption bar at the top left of the frame reads "Human: flatten the towel", and a curved arrow from a bold $\mathbf{z}_ i$ outside the frame points at it. A bar at the bottom left reads "autonomous, 10x speed", and a white $\pi$ logo sits at the bottom right of the frame.

At right: "Two tasks:" then the lines $\mathbf{z}_ 1$ = "flatten the towel" and $\mathbf{z}_ 2$ = "hang the towel".

In the recording (≈53:11–53:56) Finn plays this as a video: the task identifier is shown in the top left, the robot first flattens the towel, and once the towel is more flattened the human switches the task and tells the robot to hang the towel, the instruction again passed into a transformer.

## Slide 22 — Example of multi-task imitation learning

![Slide 22 — Example of multi-task imitation learning](../images/12-multi-task-rl/slide-22.jpg)

Build of slide 21 with a different frame, again appearing to be a video frame: a top-down view of a home kitchen with beige tile floor, a stainless refrigerator, dark cabinets, a sink and a table with a plate and napkin; a mobile robot with two arms and a vertical torso stands in the middle. A bar at the bottom left reads "autonomous, 10x speed" and the $\pi$ logo is at the bottom right. A curved arrow from bold $\mathbf{z}_ i$ points at the frame's top left, as on slide 21.

At right, "Many tasks:" with

- $\mathbf{z}_ 1$ = "put the plate in the drawer"
- $\mathbf{z}_ 2$ = "close the microwave"
- $\mathbf{z}_ 3$ = "drive base backward"
- $\mathbf{z}_ 4$ = "rotate base left"
- $\mathbf{z}_ 5$ = "throw the napkin away"
- $\mathbf{z}_ 6$ = "put the cup in the sink"
- $\mathbf{z}_ 7$ = "lift torso up"
- $\mathbf{z}_ 8$ = "clean the spill"
- ...

Bottom line: "What about high-level, long-horizon tasks like "clean the kitchen"? -> Lecture on hierarchy".

In the recording (≈53:56–54:44) Finn plays this as a video of more than ten tasks, ten different language strings the robot follows, reading out "put the plate in the drawer", "close the microwave", "drive the base backward" and "rotate the base left", and notes that the tasks have different levels of abstraction. The slide has no citation line; slide 21's citation (Pi Team et al.) is not repeated.

## Slide 23 — Multi-Task RL Algorithms

Bold subtitle "The basics".

- Policy: $\pi_\theta(\mathbf{a} \mid \bar{\mathbf{s}}) \thinspace \rightarrow \thinspace \pi_\theta(\mathbf{a} \mid \bar{\mathbf{s}}, \mathbf{z}_ i)$ (printed with the arrow as "—>")
- Q-function: $Q_\phi(\bar{\mathbf{s}}, \mathbf{a}) \thinspace \rightarrow \thinspace Q_\phi(\bar{\mathbf{s}}, \mathbf{a}, \mathbf{z}_ i)$

"Consider per-task replay buffer for stratified sampling."

Lower block: left, "Can we share data *across tasks*?"; right, "If we collect policy data when conditioning on $\mathbf{z}_ 1$, can we reuse that data to learn something for task 2?" and, centred below, "Would require relabeling the data with $\mathbf{z}_ 2$!"

## Slide 24 — An example

![Slide 24 — An example](../images/12-multi-task-rl/slide-24.jpg)

Clip-art silhouettes of ice-hockey players. At left, "Task 1: passing" beside a dark-blue silhouette of a player with a stick across the body, and "Task 2: shooting goals" beside an orange silhouette of a player at a red hockey net. At right, a larger light-blue silhouette of a player skating with a stick and a puck.

Text below: "What if you accidentally perform a good pass when trying to shoot a goal?" Then "Store experience as normal. *and* Relabel experience with task 2 ID & reward and store." (the "*and*" is printed with asterisks). Last line: ""hindsight relabeling" "hindsight experience replay" (HER)" (the first quoted name sits under "Relabel experience", the second to its right). The number 24 prints under the word "hindsight" in the last line.

Printed slip, corrected aloud: the slide relabels the accidental pass "with task 2 ID", but passing is task 1. Finn relabels it "with the task one ID, specifically the passing ID" and says "There's a typo there" (≈1:00:08).

## Slide 25 — Multi-task RL with relabeling

Title with "Multi-task RL" in teal. A four-step algorithm with a green curved arrow at the left from step 4 back to step 1, labelled $k$++:

1. Collect data $\mathcal{D}_ k = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{z}_ i, r_{1:T}) \rbrace$ using some policy
2. Store data in replay buffer $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_ k$
3. Perform hindsight relabeling (words in orange):
   - a. Relabel experience in $\mathcal{D}_ k$ for task $\mathcal{T}_ j$: $\mathcal{D}'_ {k} = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{z}_ j, r'_ {1:T}) \rbrace$ where $r'_ t = r_j(\mathbf{s}_ t)$
   - b. Store relabeled data in replay buffer $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}'_ {k}$
4. Update policy using replay buffer $\mathcal{D}$

At right, "<— Which task $\mathcal{T}_ j$ to choose?" with two dashed answers: "randomly" and "task(s) in which the trajectory gets high reward".

Below: "When can we apply relabeling?" with three dashed items: "reward function form is known, evaluatable" (spelled as printed, "evaluatable"); "dynamics consistent across goals/tasks"; "using an off-policy algorithm*" (the asterisk has no matching footnote on the page).

Citations bottom left: "Kaelbling. Learning to Achieve Goals. IJCAI '93" and "Andrychowicz et al. Hindsight Experience Replay. NeurIPS '17".

## Slide 26 — Goal-conditioned RL with hindsight relabeling

Title with "Goal-conditioned RL" in magenta. The same algorithm layout and green $k$++ arrow as slide 25, specialized to goals:

1. Collect data $\mathcal{D}_ k = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{s}_ g, r_{1:T}) \rbrace$ using some policy
2. Store data in replay buffer $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_ k$
3. Perform hindsight relabeling (orange):
   - a. Relabel experience in $\mathcal{D}_ k$ using last state as goal: $\mathcal{D}'_ {k} = \lbrace (\mathbf{s}_ {1:T}, \mathbf{a}_ {1:T}, \mathbf{s}_ T, r'_ {1:T}) \rbrace$ where $r'_ t = -d(\mathbf{s}_ t, \mathbf{s}_ T)$
   - b. Store relabeled data in replay buffer $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}'_ {k}$
4. Update policy using replay buffer $\mathcal{D}$

At right: "<— Other relabeling strategies?" and "use any state (orange) from the trajectory".

Centre: "Result: exploration challenges alleviated". Bottom right in magenta: "You will implement hindsight relabeling in homework 4!". Citations bottom left as on slide 25 (Kaelbling IJCAI '93; Andrychowicz et al. NeurIPS '17).

The slide prints $\mathbf{s}_ T$ and $\mathbf{s}_ t$ without the $\bar{\ }$ of slide 17's notation and uses $d(\mathbf{s}_ t, \mathbf{s}_ T)$ for the distance.

## Slide 27 — Hindsight relabeling for goal-conditioned RL

![Slide 27 — Hindsight relabeling for goal-conditioned RL](../images/12-multi-task-rl/slide-27.jpg)

Subtitle: "Example: goal-conditioned RL, simulated robot manipulation". Citations bottom left as on slide 25.

Left: a 3 by 5 grid of simulation frames (each row a time sequence of five frames, left to right) from the Hindsight Experience Replay paper, with the caption "Figure 2: Different tasks: *pushing* (top row), *sliding* (middle row) and *pick-and-place* (bottom row). The red ball denotes the goal position." Top row: a blue robot arm above a grey table pushing a small box towards a red dot. Middle row: a grey table seen from above with a green puck sliding past a small red dot as the arm at the top left moves. Bottom row: the arm lifting a small box off the table, with a red dot at the goal.

Right: a chart panel of three side-by-side line plots, one per task, with a shared legend and shared axes. Legend: "DDPG" (red dashed line), "DDPG+count-based exploration" (green line), "DDPG+HER" (blue line), "DDPG+HER (version from Sec. 4.5)" (red solid line). Each plot's y axis is "success rate", 0% to 100% in steps of 20%; the x axis is "epoch number (every epoch = 800 episodes = 800x50 timesteps)", 0 to 200 in steps of 50. Each curve has a shaded band around it, in its own colour. There are four series in each plot; values read at epochs 50, 100, 200:

pushing:
- red dashed DDPG: about 0% throughout.
- green count-based exploration: about 0% throughout, rising to a few percent near 200.
- blue DDPG+HER: about 15% at 50 (8% at 40, 23% at 60), about 63% at 100, about 94% at 200.
- red solid DDPG+HER (Sec. 4.5): about 55% at 50 (33% at 40, 78% at 60), about 90% at 100, about 95% at 200.

sliding:
- red dashed DDPG: about 0% throughout.
- green count-based exploration: about 7% at 50, about 12% at 100, about 22% at 200, with a wide band.
- blue DDPG+HER: about 10% at 50, about 16% at 100, about 42% at 200.
- red solid DDPG+HER (Sec. 4.5): about 12% at 50, about 45% at 100, about 87% at 200.

pick-and-place:
- red dashed DDPG: about 0% throughout.
- green count-based exploration: about 0% throughout.
- blue DDPG+HER: about 78% at 40, about 97% at 60, about 96% at 100, dipping to about 80% at 160, about 91% at 200.
- red solid DDPG+HER (Sec. 4.5): about 79% at 40, about 95% at 60, about 93% at 100, dipping to about 86% at 140, about 95% at 200.

In pick-and-place the two HER curves rise together from a few percent at epoch 20 to about 95% by epoch 60. These values are read off the plot by eye; the shapes, not the exact numbers, are what the slide shows.

## Slide 28 — The plan for today

The agenda of slides 3 and 10 repeated, with no item in bold, marking the end of the lecture. Items: 1. Model-based reinforcement learning (a. Using a learned model via synthetic data generation; b. When to use model-based RL); 2. Multi-task imitation and reinforcement learning (a. Task conditioning; b. Goal-reaching tasks; c. Hindsight relabeling), with the brace and "Part of homework 4!" beside 2a–2c.

## Slide 29 — Course reminders

- Homework 3 due next Friday, May 16
- Project milestone due Friday, May 23

Then a second heading, "Next Time", with: "Can we quickly adapt to new tasks?" and "(e.g. in-context learning for RL)".

The reminders repeat slide 2's, including "next Friday, May 16".
