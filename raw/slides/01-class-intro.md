---
title: Lecture 1 — Class Intro (slide deck)
lecture: 1
slides: 45
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–45). Pages 25 and 29 print no number; labels inferred from position. Page 30's "30" is in the text layer but hidden under its image.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (24, 25, 30, 33, 36, 37) were checked by Claude Opus 5.5 against 1400px renders. All six agreed on structure, arrows, labels and equations. The audit added the lecture's context to slide 30 and corrected one slip note (slide 37 also prints the t+1 subscript in bold).
images: 16 slides are rendered and shown under their headings (19–21, 23–28, 30, 32–37). Not rendered — the title, welcome, agenda and text-only slides (1, 2, 4–7, 9–12, 14–18, 29, 31, 39, 41, 42, 44, 45); slide 3, photographs of course staff; slide 8, five clip-art icons whose captions are transcribed; slide 13, a meme; slide 22, product logos whose names are transcribed; and slides 38, 40 and 43, whose only images are equations and a brace reproduced exactly below.
---

# Lecture 1 — Class Intro: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 1 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 45 pages. Slides 25 and 29 print no number; their labels are inferred from position. Slide 30's number is present in the PDF's text layer but covered by the slide's image, so none is visible on the page.

The recorded lecture does not cover slides 5–6 and 10–15 (course logistics): the video passes from slide 9 to slide 17. Slides 1–3 and 45 are not discussed either. Their content is here and nowhere in the transcript.

Companion pages: [wiki page for this lecture](../../wiki/01-class-intro.md) · [transcript](../transcripts/01-class-intro.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and welcome |
| 3 | Introductions (course staff) |
| 4–15 | Plan, logistics, course goals, prerequisites, grading, coursework, initial steps |
| 16–30 | Why study deep RL: beyond supervised learning, legged robots, manipulation, games, language models, traffic control, image models, chip design, learning from practice, open problems, behind the scenes |
| 31–45 | Modeling behavior and RL: states, observations, actions, trajectories, rewards, Markov property, policy networks, the RL objective, stochastic policies, value and Q-functions, algorithm types, recap, course reminders |

---

## Slide 1 — Deep Reinforcement Learning

Title page. Large text "Deep Reinforcement Learning", below it "CS 224R". Slide number 1 at the bottom.

## Slide 2 — Welcome!

Single centred word: "Welcome!".

## Slide 3 — Introductions

Photo grid of course staff with printed names and roles (no further description of the people).

Left block, two rows of two photos:
- Prof. Chelsea Finn, Instructor
- Jubayer Ibn Hamid, Head Teaching Assistant
- Amelie Byun, Course Manager
- John Cho, Course Manager

Right block, a grid of 3 rows by 6 columns of photos, each captioned with a name only (no roles printed):
- Row 1: Annie Chen, Anikait Singh, Sergio Charles, Ashish Rao, Fengyu Li, Marcel Torne
- Row 2: Yash Kankariya, Andy Tang, Haoyi Duan, Shirley Wu, Zhen Wu, Jinny Chung
- Row 3: Sirui Chen, Pulkit Goel, Joy He Yueya, Daniel Shin, Sri Jaladi, Jensen Gao

## Slide 4 — The Plan for Today

1. Course goals & logistics
2. Why study deep reinforcement learning?
3. Intro to modeling behavior and reinforcement learning

Key learnings goals:
- what is deep reinforcement learning??
- how to represent behavior
- how to formulate a reinforcement learning problem

## Slide 5 — Information & Resources

- Course website: http://cs224r.stanford.edu/ (link) — annotation with arrow: "We have put a lot of info here. Please read it. :)"
- Ed, Gradescope: Connected to Canvas
- Staff mailing list: cs224r-staff-spr2425@cs.stanford.edu (link) — annotation with arrow: "Student liaison, course manager, head CA, me"
- Office hours: Course website & Canvas, start today.

Bottom: "OAE letters can be sent to staff mailing list or in private Ed post."

## Slide 6 — Lectures & Office Hours

Lectures
- In-person, livestreamed, & recorded
- A few guest lectures (Ashish Kumar from Tesla, Archit Sharma from Google DeepMind, one TBD)
- Aiming to make it interactive. I will ask you questions. Ask me questions too!

Office hours
- mix of in-person and remote

## Slide 7 — What do we mean by deep reinforcement learning?

Title with "deep" underlined in magenta and "reinforcement learning" underlined in blue.

Blue heading: Sequential decision-making problems
- A system needs to make *multiple* decisions based on stream of information.
- observe, take action, observe, take action, ...

Blue heading: AND the solutions to such problems

Three columns of bullets, with "and more!" to the right:
- imitation learning; model-free & model-based RL
- offline & online RL; multi-task & meta RL
- RL for LLMs; RL for robots

Magenta text: Emphasis on solutions that scale to deep neural networks

## Slide 8 — How does deep RL differ from other ML topics?

Two columns.

Supervised learning
- Given labeled data: $\lbrace (x_i, y_i) \rbrace$, learn $f(x) \approx y$
- directly told what to output
- inputs x are independently, identically distributed (i.i.d.)

Reinforcement learning
- Learn *behavior* $\pi(a \mid s)$.
- from experience, indirect feedback
- data **not** i.i.d.: actions $a$ affect the future observations.

Bottom: "Behavior can include:" followed by a row of five clip-art icons with captions:
- a small grey cartoon robot with antennae and a purple flower — "motor control"
- two overlapping speech bubbles (red and blue) above two person silhouettes — "chat bots"
- a game controller — "game playing"
- a green cartoon car — "driving"
- a blue globe with "www" banner — "web agents"

## Slide 9 — We can't cover everything in deep RL.

We'll focus on:
- core concepts behind deep RL methods
- implementation of algorithms
- examples in robotics, control, language models (but techniques generalize broadly)
- topics that we think are most useful & exciting!

Side note at right: "For more theory & other applications, see CS234!"

Bottom: **Core goal**: Able to understand and implement existing and emerging methods.

## Slide 10 — Pre-Requisites

Machine learning: CS229 or equivalent.
- e.g. we'll assume knowledge of SGD, cross-val, calculus, probability theory

Some familiarity with deep learning:
- We'll build on concepts like backpropagation, neural networks, sequence models
- Assignments will require training networks in PyTorch.
- Marcel will hold a PyTorch review session on Friday, 1:30 pm in Gates B1.

Some familiarity with reinforcement learning:
- We will go quickly over the basics.
- See Sutton & Barto or CS 221 for intro RL content

Side note at right: "Aiming to improve accessibility compared to Spring '23!"

## Slide 11 — Coursework and Grading

- **4 x 2-week assignments** (50% - lowest scoring is 5%, rest worth 15%)
- **Final default or custom course project** (1-3 people, 50%)
  - proposal (10%), milestone (10%), poster (10%), report (20%)
- **Late days**:
  - 6 free late days; afterwards, 2% of course grade per day late
  - maximum of 2 free late days per assignment unless advanced permission
- **Collaboration & AI tools**
  - Please read course website, honor code, AI tools policy (the last two are hyperlinks)
  - Document collaborators and write solutions on your own. Submit homework independently.
  - Employing AI tools (e.g. ChatGPT, Cursor) substantially is not allowed for homework and parts of default project.

## Slide 12 — Coursework

**Homeworks**: Implement different methods in PyTorch, run experiments in physics simulators, navigation environments
- Homework 1: Imitation learning
- Homework 2: Online reinforcement learning
- Homework 3: Offline reinforcement learning
- Homework 4: Goal-conditioned & meta reinforcement learning

**Project**:
- Custom project - propose your own topic, or
- Default project (*new this year!*, shown in red) - fine-tune an LLM with RL + open-ended extension
- Teams of 1-3 students, encouraged to use your research if applicable

## Slide 13 — A bit of advice

Deep RL methods take time to learn behavior!

- We try to make homeworks fast to train. (e.g. by using simple environments)
- *But*, they will still take some time & you may choose to be more ambitious in your project.

Right side: a three-panel meme image. Top panel (wide): a man sitting alone on a white swing seat in a park, with the caption "Me waiting for my neural network to finish training". Bottom left panel: a man seated at a kitchen table in a dim room, looking down. Bottom right panel: a man standing at the edge of an empty, dry swimming pool. The three panels convey waiting and boredom.

Bottom: We recommend that you don't start HWs/project deliverables the night before the deadline. :)

## Slide 14 — One more thing

- We have been working hard to develop a great course!
- But, we will probably make mistakes.
- We would **love** your feedback both for this iteration & future iterations.
  - —> high-resolution feedback form sent weekly to subset of students.

## Slide 15 — Initial Steps

1. Homework 1 coming out on Fri — due Fri 4/18 at 11:59 pm PT
2. Start forming final project groups if you want to work in a group

## Slide 16 — The Plan for Today

Same agenda as slide 4, with item 2 in bold:

1. Course goals & logistics
2. **Why study deep reinforcement learning?**
3. Intro to modeling behavior and reinforcement learning

## Slide 17 — Why study deep reinforcement learning?

1. Going **beyond supervised (x, y) examples**
   - AI model predictions have consequences!
   - When direct supervision isn't available
   - Side notes at right: "How can we take them into account?" and "Learn from *any* objective."
2. **Widely used and deployed** for performant AI systems
3. Learning from experience seems **fundamental to intelligence**
   - RL can **discover new solutions**
4. Plenty of exciting open research problems

## Slide 18 — Why study deep reinforcement learning? (Beyond supervised learning from (x, y) examples)

Subtitle (grey): Beyond supervised learning from (x, y) examples

Blue heading: Decision-making problems are everywhere!
- a. Any sort of AI agent: robots, autonomous vehicles, web assistants
- b. What if you want your AI system to interact with people? — teal annotation: "chatbots, recommenders"
- c. What if deploying your system affects future outcomes & observations? — teal annotation: "feedback loops" (in quotation marks)
- d. What if don't have labels or your objective isn't just accuracy? — teal annotation: "(and isn't differentiable!)"

## Slide 19 — Why study deep reinforcement learning? (Widely used for performant AI systems): legged robots

![Slide 19 — Why study deep reinforcement learning? (Widely used for performant AI systems): legged robots](../images/01-class-intro/slide-19.jpg)

Subtitle (grey): Widely used for performant AI systems

Heading: Learning complex physical tasks: legged robots

Two side-by-side photographs (each appears to be a still from an embedded video):
- Left: a small grey quadruped robot walking on a grassy dirt path with trees behind. Overlay text "Happy Puppy" at top left and "1 x No acceleration" at bottom left.
- Right: a silver-and-white humanoid robot with a dark face screen with a blue glowing outline standing upright in an office or lab with glass partitions, a grey cabinet and a desk with papers.

Caption under the left image: "Unitree".

## Slide 20 — Why study deep reinforcement learning? (Widely used for performant AI systems): robot manipulation

![Slide 20 — Why study deep reinforcement learning? (Widely used for performant AI systems): robot manipulation](../images/01-class-intro/slide-20.jpg)

Subtitle (grey): Widely used for performant AI systems

Heading: Learning complex physical tasks: robot manipulation

Two side-by-side photographs (each appears to be a still from an embedded video):
- Left: a wide view of a home interior with a laundry area (washer and dryer under a staircase) and a kitchen with a round table in the background; in the foreground a mobile robot with two arms and a keyboard/tablet stands next to a laundry basket. Overlay text bottom left: "autonomous, 2x speed". Caption below: "Physical Intelligence $\pi_0$".
- Right: top-down view of two black robot arms with pincer-like grippers reaching towards a folded orange square piece of paper or cloth lying on a wooden table surrounded by a black metal frame. Overlay text: "Gemini Robotics" at bottom left and "Autonomous 1x" in a black pill at bottom right. Caption below: "Google Gemini Robotics".

## Slide 21 — Why study deep reinforcement learning? (Widely used for performant AI systems): complex games

![Slide 21 — Why study deep reinforcement learning? (Widely used for performant AI systems): complex games](../images/01-class-intro/slide-21.jpg)

Subtitle (grey): Widely used for performant AI systems

Heading: Learning to play complex games

Left: a screenshot of a broadcast of a Go match. The main area shows a Go board seen from above, nearly full of black and white stones, with bowls of stones at the right edge. A right column of three panels: the top panel shows a black stone icon with "柯洁 KE JIE" and a clock "00:13:28"; the middle panel shows a white stone icon with "ALPHAGO" and a clock "01:30:35"; the bottom panel is a wide shot of the match room with two seated players facing each other across a table with a Go board, and a banner whose legible text includes Chinese characters, "Google" and "e Future of Go Summit in Wuzhen" (left end cut off).

Right text: "Ability to **discover** new solutions:" then ""Move 37" in Lee Sedol AlphaGo match surprises everyone".

## Slide 22 — Why study deep reinforcement learning? (Widely used for performant AI systems): language models

Subtitle (grey): Widely used for performant AI systems

**Not just robots and games!**

Nearly all modern language models use some form of RL for post-training.

A logo cluster, two rows: top row "ChatGPT" (with its knot-shaped logo), "Claude" (orange starburst on a cream background), "Gemini" (with a four-pointed sparkle); bottom row "deepseek" (blue whale logo) and "CURSOR" (cube logo).

Bottom right: "*especially* for more advanced reasoning."

## Slide 23 — Why study deep reinforcement learning? (Widely used for performant AI systems): traffic control

![Slide 23 — Why study deep reinforcement learning? (Widely used for performant AI systems): traffic control](../images/01-class-intro/slide-23.png)

Subtitle (grey): Widely used for performant AI systems

**Not just robots and games!**

Heading: Research on traffic control

Two grey-background simulation screenshots of cars (small white and red car icons) on black road shapes, seen from above:
- Top: a wide multi-lane road on the left, densely packed with cars (a few red, one yellow at the far left), narrowing in the middle through a funnel shape to progressively fewer lanes, with cars thinning out, to a single lane at the far right.
- Bottom: a long single-lane horizontal road with widely spaced cars (a few red), and a diagonal on-ramp from the lower middle merging into it from below at about two thirds of the width, with the road thicker and denser after the merge.

Source: https://mit-wu-lab.github.io/automatic_vehicular_control/

## Slide 24 — Why study deep reinforcement learning? (Widely used for performant AI systems): generative image models

![Slide 24 — Why study deep reinforcement learning? (Widely used for performant AI systems): generative image models](../images/01-class-intro/slide-24.jpg)

Subtitle (grey): Widely used for performant AI systems

**Not just robots and games!**

Heading: Training generative image models to follow their prompt

Two strips, each a row of five generated images with a long arrow spanning above it and the prompt printed in italics at the arrow's centre:
- Left strip, prompt "a dolphin riding a bike" with a line at the left end and an arrowhead at the right end: from left to right the images go from a dolphin in the water (no bike), to a person on a bicycle with a small dolphin shape above, to cartoon-style blue dolphins riding bicycles (three images, increasingly clear and stylised).
- Right strip, prompt "an ant playing chess": from left to right the images go from ants on a black-and-white chequered board, to a dark ant near a board, to orange ants at a chess board (two images), to a single large red ant seated at a chess board at the far right.

The arrows indicate progress along the training (left to right, images follow the prompt better); the slide does not print a label for the arrow direction.

Source: https://rl-diffusion.github.io/

## Slide 25 — Why study deep reinforcement learning? (Widely used for performant AI systems): chip design

![Slide 25 — Why study deep reinforcement learning? (Widely used for performant AI systems): chip design](../images/01-class-intro/slide-25.jpg)

(No slide number printed; label inferred.)

Subtitle (grey): Widely used for performant AI systems

**Not just robots and games!**

Heading: Chip design, in Google's production TPU chips

Diagram: a left-to-right sequence of states and an agent. Light-green squares represent chip canvases; black rectangles on them represent placed macros.
- Label "Chip canvas" with a dashed arrow down into the first square $s_0$ (empty).
- Arrows: $s_0$ → "Agent" (blue oval) → $s_1$ (square with one small black block at top left) → "Agent" → $s_2$ (square with one small block at top left and one larger block below it) → "Agent" → "...." 
- Action labels above the arrows: $a_0$, $a_1$, ..., $a_{T-1}$. Reward labels: $r_0 = 0$, $r_1 = 0$, $r_2 = 0$.
- Header text above the early part: "RL agent places macros one at a time".
- After the ellipsis, the arrow $a_{T-1}$ leads into a nearly full square (many black blocks in rows) which, via a dotted arrow, leads to the final state $s_T$ (the same blocks plus a grey cloud-like blob at the top centre). A dashed arrow from the text "Force-directed method places standard cell" points to this blob. $s_T$ → "Agent".
- Reward at the end, to the right: $r_T = -\text{Wirelength} - \lambda \times \text{congestion} - \gamma \times \text{density}$ (printed on three lines).

Source: https://research.google/blog/chip-design-with-deep-reinforcement-learning/

## Slide 26 — Why study deep reinforcement learning? (Fundamental aspect of intelligence): PR2 without vision

![Slide 26 — Why study deep reinforcement learning? (Fundamental aspect of intelligence): PR2 without vision](../images/01-class-intro/slide-26.jpg)

Subtitle (grey): Fundamental aspect of intelligence

A frame from an embedded video (appears to be a still): a two-armed white-and-grey robot labelled "PR2" with a "UC Berkeley" banner on its head, in a lab with monitors, a tripod and a green-covered table, holding a blue-and-yellow toy in its gripper at a green surface. Video overlay text: "linear-Gaussian controller training" (top), "1x real time" (bottom left), and partly hidden text ending in "nomous execution" (bottom right; likely "autonomous execution", partly covered by the slide caption). A sign on the left reads "welcomes ROBOTICS" with smaller text "SCIENCE AND SYSTEMS".

Slide caption overlaid on the video: "The robot has its eyes closed."

Right text: "Enables the ability to get better with practice"

Citation: Levine et al. ICRA '15

In the lecture (≈11:35–12:21) the task in this video is assembling two parts of a toy airplane, learned in the real world by a robot that used no camera.

## Slide 27 — Why study deep reinforcement learning? (Fundamental aspect of intelligence): PR2 with vision, iteration 1

![Slide 27 — Why study deep reinforcement learning? (Fundamental aspect of intelligence): PR2 with vision, iteration 1](../images/01-class-intro/slide-27.jpg)

Subtitle (grey): Fundamental aspect of intelligence

A frame from an embedded video (appears to be a still): the same type of white-and-grey two-armed "PR2" robot with a "UC Berkeley" banner at its head, in a cluttered lab with desks, monitors and a green-covered table, holding a wooden shape-sorting cube near a green draped object. Video overlay text: "10x real time" (bottom left) and "iteration 1" (bottom right).

Slide caption overlaid on the video: "with vision this time"

Right text: "Enables the ability to get better with practice"

Citation: "Levine*, Finn* et al. JMLR '1" followed by a clipped, partly overlapped character (a stray "Z" glyph from another layer); the year is not fully legible here but slide 28 shows the same citation as "JMLR '16".

## Slide 28 — Why study deep reinforcement learning? (Fundamental aspect of intelligence): shape-sorting cube

![Slide 28 — Why study deep reinforcement learning? (Fundamental aspect of intelligence): shape-sorting cube](../images/01-class-intro/slide-28.jpg)

Subtitle (grey): Fundamental aspect of intelligence

A frame from an embedded video (appears to be a still): a top-down view of a human hand (in a light sleeve) holding up a wooden cube with shaped holes (circle, pentagon-like and square shapes) above a grey carpet, with part of the robot labelled "PR2" at the right. Video overlay text: "real time" (bottom left) and "autonomous execution" (bottom right, partly overlapped by the caption).

Slide caption overlaid on the video: "with vision this time"

Right text: "Enables the ability to get better with practice"

Citation: Levine*, Finn* et al. JMLR '16

## Slide 29 — Why study deep reinforcement learning? (Still lots of exciting research problems!)

(No slide number printed; label inferred.)

Subtitle (grey): Still lots of exciting research problems!

Questions, each with an arrow ("—>") to a topic keyword:
- How does robot learn to represent what is good or bad for the task? —> reward learning
- How can an agent generalize its behavior to many different scenarios? (Can we apply such a system at scale?)
  - Leverage large, diverse datasets —> offine RL (printed as "offine", a typo for "offline")
  - Transfer from other tasks, goals —> multitask RL, meta-RL
- Can use RL to learn long-horizon tasks, like cooking a meal? —> hierarchy, reasoning
- Can robots practice fully autonomously? —> reset-free RL

## Slide 30 — Behind the scenes of RL…

![Slide 30 — Behind the scenes of RL…](../images/01-class-intro/slide-30.jpg)

(No slide number is visible: the text layer holds a "30", covered by the image. Label = PDF page number.)

Title text: "Behind the scenes of RL…"

Centre: an image (appears to be a still from an animation or video) of a two-armed PR2-style robot, shown close up from the side in a lab, with its arm extended holding a long blue flexible rod or tube above the floor; a small orange object sits on the floor. Overlay text at the bottom left: "MDGPS network policy" (the left edge is cut off, so it reads "-MDGPS" / "network policy"). In the background are cluttered cables, two black computer towers on the floor and a shelf. A blue arrow from the word "Yevgen" (blue text, right of the image) points to the computer towers and the cables area at the right edge of the image.

Caption below: "Yevgen is doing more work than the robot!" / "It's not practical to collect a lot of data this way."

In the lecture (≈13:53–14:39) this is a video of the robot trying to hit a hockey puck into a goal: after every attempt the task has to be reset by hand, because the robot does not know how to undo it, and the lecturer's friend Yevgen does the resetting. The page holds a single frame. No person is clearly visible in it; the arrow points at the right edge of the frame.

## Slide 31 — The Plan for Today

Same agenda as slides 4 and 16, with item 3 in bold:

1. Course goals & logistics
2. Why study deep reinforcement learning?
3. **Intro to modeling behavior and reinforcement learning**

## Slide 32 — How to represent experience as data?

![Slide 32 — How to represent experience as data?](../images/01-class-intro/slide-32.jpg)

Definitions (term in italics, symbol in bold, as printed):
- *state* $\mathbf{s}_ t$ - the state of the "world" at time $t$
- *observation* $\mathbf{o}_ t$ - what the agent observes at time $t$ — blue annotation: "(only used when missing information)"
- *action* $\mathbf{a}_ t$ - the decision taken at time $t$
- *trajectory* $\tau$ - sequence of states/observations and actions

$$(\mathbf{s}_ 1, \mathbf{a}_ 1, \mathbf{s}_ 2, \mathbf{a}_ 2, \ldots, \mathbf{s}_ T, \mathbf{a}_ T)$$

  blue annotation next to it: "(could be length T=1!)"
- *reward function* $r(\mathbf{s}, \mathbf{a})$ - how good is $\mathbf{s}, \mathbf{a}$?

Top right, three small images: (1) a photo of a pale yellow-green towel lying on a grey table next to a metal hook stand, with part of a robot arm at the top (the same scene as slide 34, left); (2) and (3) two dark-themed screenshots of a chatbot conversation containing code, shown side by side. Their text is far too small to read at this size. The blue annotation partly overlaps the lower part of the screenshots.

## Slide 33 — States vs. observations

![Slide 33 — States vs. observations](../images/01-class-intro/slide-33.jpg)

Subtitle: "Next state is purely a function of the current state and action (and randomness)"

Diagram (a graphical model drawn with circles and arrows, left to right). Top row of circles: $\mathbf{o}_ 1$, $\mathbf{a}_ 1$, $\mathbf{o}_ 2$, $\mathbf{a}_ 2$, $\mathbf{o}_ 3$. Bottom row: $\mathbf{s}_ 1$, $\mathbf{s}_ 2$, $\mathbf{s}_ 3$. Arrows:
- $\mathbf{s}_ 1$ to $\mathbf{o}_ 1$, $\mathbf{s}_ 2$ to $\mathbf{o}_ 2$, $\mathbf{s}_ 3$ to $\mathbf{o}_ 3$ (each state emits an observation, short upward arrows)
- $\mathbf{s}_ 1$ to $\mathbf{s}_ 2$ and $\mathbf{s}_ 2$ to $\mathbf{s}_ 3$ (long horizontal arrows)
- $\mathbf{a}_ 1$ to $\mathbf{s}_ 2$ and $\mathbf{a}_ 2$ to $\mathbf{s}_ 3$ (diagonal arrows)
There are no arrows from observations into later nodes and none from actions to observations directly. No arrows are drawn from the observations to the actions.

Text below the diagram: "unknown dynamics $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$" and "*independent of* $\mathbf{s}_ {t-1}$". Bottom, bold: "Markov property" (in quotation marks).

## Slide 34 — Examples

![Slide 34 — Examples](../images/01-class-intro/slide-34.jpg)

Two columns, each with an image on top.

Left column (robot towel task). Image: a photo from a camera above a grey table showing a pale yellow-green towel on the table and a metal hook on a round base; the top shows a robot arm with gripper; the label "Nest" appears at the top right of the image.
- *state* $\mathbf{s}$ - RGB images, joint positions, joint velocities
- *action* $\mathbf{a}$ - commanded next joint position
- *trajectory* $\tau$ - 10-sec sequence of camera, joint readings, controls at 20 Hz

$$(\mathbf{s}_ 1, \mathbf{a}_ 1, \mathbf{s}_ 2, \mathbf{a}_ 2, \ldots, \mathbf{s}_ T, \mathbf{a}_ T), \thinspace T = 200$$

- *reward* $r(\mathbf{s}, \mathbf{a})$ = 1 if the towel is on the hook in state $\mathbf{s}$; 0 otherwise

Right column (chatbot). Image: two dark-themed chatbot conversation screenshots side by side (the same pair as slide 32), with code blocks and explanatory text too small to read.
- *observation* $\mathbf{o}$ - the user's most recent message
- *action* $\mathbf{a}$ - chatbot's next message
- *trajectory* $\tau$ - variable length conversation trace

$$(\mathbf{o}_ 1, \mathbf{a}_ 1, \mathbf{o}_ 2, \mathbf{a}_ 2, \ldots, \mathbf{o}_ T, \mathbf{a}_ T)$$

- *reward* $r(\mathbf{s}, \mathbf{a})$ = 1 if the user gives upvote; -10 if the user downvotes; 0 if no user feedback (printed with $\mathbf{s}$, although this column uses observations)

## Slide 35 — Think-pair-share: how to represent another example?

![Slide 35 — Think-pair-share: how to represent another example?](../images/01-class-intro/slide-35.jpg)

Left, a 2 by 2 grid of images with captions:
- "autonomous driving": a photograph of a multi-lane highway with cars and a white truck, and a red truck at the lower left
- "web agent": a screenshot of a restaurant booking page with two listings, each with a food photo, a star rating and a row of five red time buttons (6:30 PM, 6:45 PM, 7:00 PM, 7:15 PM, 7:30 PM); the listings read "Wildseed Palo Alto" and "Tacolicious - Palo Alto"
- "poker player": a photograph of a hand holding playing cards over a green felt table with stacks of poker chips
- "choose your own!": a green clip-art drawing of grass blades

Right side: bold "Define", then
- *state* $\mathbf{s}$ or *observation* $\mathbf{o}$
- *action* $\mathbf{a}$
- *trajectory* $\tau$
- *reward* $r(\mathbf{s}, \mathbf{a})$

## Slide 36 — How to represent behavior with a neural network?

![Slide 36 — How to represent behavior with a neural network?](../images/01-class-intro/slide-36.jpg)

Diagram across the top. At the far left, a photo labelled $\mathbf{s}$ (the state): a robot gripper above a towel lying on a grey table beside a metal hook. An arrow leads into a convolutional network drawn as stacks of blue squares followed by narrow grey vertical bars, ending at a label $\mathbf{a}$ and a second photo on the right (the same scene with a red arrow drawn over the towel near the gripper, illustrating the action). Layers from left to right, with the printed labels:
- "7x7 conv, stride 2, ReLU" then a stack labelled "64 filters"
- "5x5 conv, ReLU" then a stack labelled "32 filters"
- "5x5 conv, ReLU" then a stack labelled "32 filters"
- three narrow vertical bars (fully connected layers): "fully connected ReLU", "fully connected ReLU", "fully connected linear"

A magenta $\theta$ at the top centre has six magenta arrows, one pointing at each of the six layers (three convolutional, three fully connected), indicating that $\theta$ denotes the parameters of all layers. Below the network, the label $\pi_\theta(\mathbf{a} \mid \mathbf{s})$.

Text below:
- Observe state $\mathbf{s}_ t$
- Take action $\mathbf{a}_ t$, e.g. by sampling from *policy* $\pi_\theta(\cdot \mid \mathbf{s}_ t)$
- Observe next state $\mathbf{s}_ {t+1}$ — sampled from unknown world dynamics $p(\cdot \mid \mathbf{s}_ t, \mathbf{a}_ t)$
- A blue curved arrow on the left runs from "Observe next state" back up to "Take action", showing the loop repeats.

Result: a *trajectory* $\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T,$ — also called a policy *roll-out* or an *episode*

If you only have observations $\mathbf{o}$, give the policy memory: $\pi_\theta(\mathbf{a}_ t \mid \mathbf{o}_ {t-m}, \ldots, \mathbf{o}_ t)$

## Slide 37 — What is the goal of reinforcement learning?

![Slide 37 — What is the goal of reinforcement learning?](../images/01-class-intro/slide-37.jpg)

Top: "maximize sum of rewards:"

$$\max \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$$

A blue arrow from the formula points to the text "but this is not a deterministic quantity!". Teal text: "Question: what are the sources of variability?"

Bottom left, a graphical model of circles and arrows: $\mathbf{s}_ 1$ to $\mathbf{a}_ 1$, the arrow labelled $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$; $\mathbf{s}_ 1$ to $\mathbf{s}_ 2$, a long arrow labelled $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$; $\mathbf{a}_ 1$ to $\mathbf{s}_ 2$; then $\mathbf{s}_ 2$ to $\mathbf{a}_ 2$, labelled $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$; $\mathbf{s}_ 2$ to $\mathbf{s}_ 3$, labelled $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$; and $\mathbf{a}_ 2$ to $\mathbf{s}_ 3$.

Below it:

$$p(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T) = p(\mathbf{s}_ 1) \prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

A curly brace under the left-hand side $p(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T)$ is labelled $p_\theta(\tau)$. (In this product the subscript "t+1" of $\mathbf{s}_ {t+1}$ is printed in bold, as on slides 38, 40 and 43. The graph's labels print it plain.)

Bottom right, a pink box headed "Example" with the highway photo (cars and trucks on a multi-lane road, as on slide 35) and the text: "1. the world is stochastic" "2. the car may not make the same decision every time."

## Slide 38 — What is the goal of reinforcement learning?

Top line, greyed out and struck through: "maximize sum of rewards:" $\max \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$.

Below, in full black: "maximize *expected* sum of rewards:"

$$\max_{\theta} \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

Then the same trajectory distribution as slide 37:

$$p(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T) = p(\mathbf{s}_ 1) \prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

with a curly brace under the left-hand side labelled $p_\theta(\tau)$. (On this slide the subscript "t+1" of $\mathbf{s}_ {t+1}$ is printed in bold.)

## Slide 39 — Aside: why stochastic policies?

1. **Exploration**: to learn from your own experience, must try different things.
2. **Modeling stochastic behavior**: existing data will exhibit varying behaviors

We can leverage tools from **generative modeling**!

—> generative model over ***actions*** given states/observations

## Slide 40 — What is the goal of reinforcement learning?

"maximize *expected* sum of rewards:"

$$\max_{\theta} \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

$$p(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T) = p(\mathbf{s}_ 1) \prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

with a curly brace under the left-hand side labelled $p_\theta(\tau)$.

Bold: **How good is a particular policy?**
- *value function* $V^\pi(\mathbf{s})$ - future expected reward starting at $\mathbf{s}$ and following $\pi$
- *Q-function* $Q^\pi(\mathbf{s}, \mathbf{a})$ - future expected reward starting at $\mathbf{s}$, taking $\mathbf{a}$, then following $\pi$

## Slide 41 — Types of algorithms

"maximize *expected* sum of rewards:"

$$\max_{\theta} \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

1. **Imitation learning**: mimic a policy that achieves high reward
2. **Policy gradients**: directly differentiate the above objective
3. **Actor-critic**: estimate value of the current policy and use it to make the policy better
4. **Value-based**: estimate value of the optimal policy
5. **Model-based**: learn to model the dynamics, and use it for planning or policy improvement

## Slide 42 — Why so many algorithms?

Algorithms make different trade-offs, thrive under different assumptions.
- How easy / cheap is it to collect data with policy? (e.g. simulator vs. hand-written)
- How easy / cheap are different forms of supervision? (demos, detailed rewards)
- How important is stability and ease-of-use?
- Action space dimensionality, continuous vs. discrete
- Is it easy to learn the dynamics model?

## Slide 43 — Recap of definitions

- *state* $\mathbf{s}_ t$ - the state of the "world" at time $t$
  - (or *observation* $\mathbf{o}_ t$ - what the agent observes at time $t$)
- *action* $\mathbf{a}_ t$ - the decision taken at time $t$
- *reward function* $r(\mathbf{s}, \mathbf{a})$ - how good is $\mathbf{s}, \mathbf{a}$?
- *initial state distr.* $p(\mathbf{s}_ 1)$, *unknown dynamics* $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$

A large curly brace at the right groups these five lines and is labelled "(partially-observed) Markov decision process" and below it "MDP, POMDP".

## Slide 44 — Recap of definitions

- *trajectory* $\tau$ - sequence of states/observations and actions $(\mathbf{s}_ 1, \mathbf{a}_ 1, \mathbf{s}_ 2, \mathbf{a}_ 2, \ldots, \mathbf{s}_ T, \mathbf{a}_ T)$
- *policy* $\pi$ - represents behavior, selecting actions based on states or observations

**Goal**: learn policy $\pi_\theta$ that maximizes *expected* sum of rewards:

$$\max_{\theta} \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

- *value function* $V^\pi(\mathbf{s})$ - future expected reward starting at $\mathbf{s}$ and following $\pi$
- *Q-function* $Q^\pi(\mathbf{s}, \mathbf{a})$ - future expected reward starting at $\mathbf{s}$, taking $\mathbf{a}$, then following $\pi$

## Slide 45 — Course Reminders

Your Initial Steps:
- Homework 1 comes out Friday, due Weds 4/18 at 11:59 pm PT
- Start forming final project groups if you want to work in a group

Coming Up Next:
- Imitation Learning Lecture (Friday 10:30, Hewlett 200)
- PyTorch Tutorial (Friday 1:30, Gates B1)
