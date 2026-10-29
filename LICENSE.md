# Licence and copyright

This repository is a **study aid compiled from Stanford CS224R course material**. It is not a
work with a single owner, so it carries no single repository-wide licence. Three layers of
authorship sit here, with three different copyright positions. Read this before reusing
anything from it.

The short version: **the explanatory writing is yours to reuse with attribution; the course
material and the third-party images in the slides are not mine to give you.**

---

## Layer 1 — the compilation's own editorial work

Copyright (c) 2026 chaimantec.

Licensed under **Creative Commons Attribution 4.0 International** (`CC-BY-4.0`),
<https://creativecommons.org/licenses/by/4.0/>. You may share and adapt it, including
commercially, with attribution.

This covers only the material original to this compilation:

- `INDEX.md`, `AGENTS.md`, `BUILD_LOG.md`, `TODO.md`, `kb.json`, `sources.md`, this file;
- the original explanatory prose in `wiki/`: the explanations, the connective argument, the
  organisation of the material into pages, and the editorial judgements about what is worth
  recording.

It does **not** extend to the lecture content those pages describe, nor to any passage in
`wiki/` that quotes the lecturer or a slide. Those are quotations, marked as such, and belong to
Layer 2 or Layer 3.

### Not offered under CC BY 4.0, even though I wrote the words

`raw/transcripts/` and `raw/slides/` are derivative works. The copy-editing, the `[Ed:]` notes
and the figure descriptions are mine; the lecture being transcribed and the slides being
described are not. I cannot license the underlying expression, so these directories are not
offered under CC BY 4.0. Read them, cite them, and quote them as you would quote the lecture
itself, but do not treat them as freely relicensable text.

The verbatim auto-captions these transcripts were edited from are **deliberately not published
here**. They would be a complete reproduction of each lecture with no editorial contribution.
They are regenerable from the public video with the `cairn-kb` skill's `fetch_transcript.py`,
using the video id in each transcript's front matter, so anyone can re-derive them and audit an
edit.

---

## Layer 2 — Stanford CS224R course material

**No licence exists, and none is granted here.**

The lectures and slide decks are the work of Chelsea Finn, the guest lecturers and the CS224R
course staff (Stanford University, Spring 2025). They are published through the course website,
<https://cs224r.stanford.edu/spring_2025/>, with the lecture videos on YouTube.

As checked on 2026-10-28, no licence, copyright or reuse statement appears on the course
website, and none is printed on any page of the Lecture 1–11 decks. Publicly downloadable is not the
same as licensed. Absent an express grant the default applies: copyright arises automatically,
and no reuse rights pass to anyone.

The images in `raw/images/` and the material described in `raw/slides/` are reproduced here for
study and commentary, relying on fair use (17 U.S.C. § 107) rather than on any permission. Only
slides that carry a figure are rendered. Title slides, text slides, section dividers and
photographs of course staff are not. Every deck is linked at its canonical URL beside the
material drawn from it. **That reliance is mine and does not transfer to you.** Fair use is a
defence to infringement, not a licence. It can justify a particular use by a particular user for
a particular purpose, and gives me nothing to pass downstream. If you want to reuse Layer 2
material, make your own assessment or ask the course staff.

**Takedown.** If the instructor or any rights holder asks for an image or a transcription to be
removed, it will be deleted along with every link to it.

---

## Layer 3 — third-party images reproduced inside the decks

The decks reproduce images that are not Stanford's: video frames, figures from research papers,
screenshots and photographs. They belong to their respective owners, Stanford could not license
them to anyone, and nothing here does either. The attributions below are taken from citations,
logos, captions and URLs visible on the slides. Where nothing on the slide identifies the source,
the entry says so.

### Lecture 1 — Class Intro

Deck: [01_cs224r_intro_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf).
Rendered: 16 slides, in `raw/images/01-class-intro/`.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 19 | Video stills of a quadruped robot (captioned "Unitree") and a humanoid. The humanoid's source is not credited. |
| 20 | Video stills captioned "Physical Intelligence π0" and "Google Gemini Robotics". |
| 21 | Screenshot of a broadcast of the AlphaGo–Ke Jie match, with a "Future of Go Summit in Wuzhen" banner and a Google logo. The broadcaster is not credited. |
| 23 | Traffic-simulation screenshots, "Source: https://mit-wu-lab.github.io/automatic_vehicular_control/". |
| 24 | Generated-image strips, "Source: https://rl-diffusion.github.io/". |
| 25 | Chip-placement diagram, "Source: https://research.google/blog/chip-design-with-deep-reinforcement-learning/". |
| 26 | Video still of a PR2 robot at UC Berkeley, cited on the slide as Levine et al., ICRA '15. |
| 27, 28 | Video stills of a PR2 robot, cited on the slides as Levine, Finn et al., JMLR '16. Slide 28 shows a person's hand holding the cube. |
| 30 | Video still of a PR2 robot with the overlay "MDGPS network policy". Not credited. |
| 32, 34 | A photograph of the towel-and-hook robot setup and two chatbot screenshots. Not credited. |
| 35 | A highway photograph, a restaurant-booking web page, a poker hand and clip-art grass. Not credited. |
| 36 | The policy-network diagram, with photographs of the towel-and-hook robot setup. Not credited. |
| 37 | A highway photograph inside the deck's own graphical model. Not credited. |
| 33 | No third-party content identified: the deck's own graphical model. |

Not rendered from this deck: the photographs of course staff (slide 3), a meme (slide 13), clip-art
icons (slide 8) and product logos (slide 22).

### Lecture 2 — Imitation Learning

Deck: [02_cs224r_imitation_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/02_cs224r_imitation_2025.pdf).
Rendered: 15 slides, in `raw/images/02-imitation-learning/`.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 3 | The policy-network diagram from Lecture 1, with photographs of the towel-and-hook robot setup. Not credited. |
| 5, 7 | A highway photograph (the one used in Lecture 1); slide 7 adds the deck's own arrows and chart. Not credited. |
| 8 | A Super Mario game screenshot and a photograph from a driver's seat. Not credited. |
| 9 | A random-noise image. Not credited. |
| 10 | A photograph of a robot arm pushing a T-shaped block. Not credited. |
| 12 | A simulation render of two robot arms and a photograph of a four-arm robot workspace, beside the deck's bar charts. The slide gives the sources "https://diffusion-policy.cs.columbia.edu/" and "https://aloha-unleashed.github.io/". |
| 13 | Video stills captioned "Physical Intelligence π0", "NVIDIA Gr00t N1" (with a "Fourier GR-1" banner) and "Figure Helix", and robot-view photographs captioned "OpenVLA". |
| 14 | An image captioned "Waymo EMMA" and a video frame captioned "Wayve LINGO-2", carrying a Wayve logo. |
| 20 | A clip-art drawing of a car with a girl and a robot driver. Not credited. |
| 23 | Photographs of kinesthetic teaching with a PR2 robot, of a game controller, a 3D mouse and a VR controller, and of a person puppeteering leader arms. Not credited. |
| 24 | Video frames of a person and of simulated characters, cited on the slide as Peng, Kanazawa, Malik, Abbeel, Levine, "SFV: Reinforcement Learning of Physical Skills from Videos", SIGGRAPH Asia 2018. |
| 18, 19, 21 | No third-party content identified: the deck's own diagrams. |

Not rendered from this deck: slides 1 and 15, whose only images are SpongeBob SquarePants cartoon
stills; the slides print "SpongeBob image rights & credits to Viacom International".

### Lecture 3 — Policy Gradients

Deck: [03_cs224r_policy_gradients_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf).
Rendered: 11 slides, in `raw/images/03-policy-gradients/`. Slides 7–10 print "Slide adapted from
Sergey Levine".

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 5 | A photograph of a humanoid robot, captioned "Source: Unitree", and a screenshot of a chat interface showing code. The screenshot is not credited. |
| 7, 10, 11 | 3D plots of trajectories over a gridded surface (slide 11's heat-mapped). The slides they appear on are "adapted from Sergey Levine" (7, 10); slide 11 credits nothing. |
| 12, 14 | A still of a simulated humanoid. Not credited. |
| 17 | A photograph of a humanoid robot holding a jacket on a table. Not credited. |
| 6, 19, 24, 26 | No third-party content identified: the deck's own loop diagrams and trajectory sketches. |

Not rendered from this deck: slide 27, which repeats slide 5; slides 13, 15 and 16, whose only
picture is a thumbnail of the slide 12 humanoid.

### Lecture 4 — Actor-Critic Methods

Deck: [04_cs224r_actor_critic_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf).
Rendered: 21 slides, in `raw/images/04-actor-critic/`. Slides 21 and 29–33 print "Slide adapted
from Sergey Levine".

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 7 | Stick-figure drawings (a person at a drum kit, at a television, on a lounge chair) and an image of a sheet of drum notation headed "Improvisation test examples and ideas for Rockschool Grade 1 drums exam", credited on the sheet to Theo Lawrence / TL Music Lessons. The stick figures are not credited. |
| 8 | A still of a simulated humanoid and a photograph of a humanoid robot at a table with a jacket, as in Lecture 3. Not credited. |
| 21, 29, 30, 31, 32 | Slides "adapted from Sergey Levine": the discounted-MDP diagram with a skull symbol (21), the replay-buffer timeline (29), and equation slides with hand-drawn annotations (30–32). |
| 35 | Two learning-curve charts, "(e) Humanoid-v1" and "(f) Humanoid (rllab)", comparing SAC, DDPG, PPO, SQL and TD3, pasted from a paper the slide does not name. |
| 3, 6, 9, 12, 14, 16, 17, 18, 19, 20, 22, 23, 27 | No third-party content identified: the deck's own loop diagrams, trajectory sketches, neural-network drawings and annotated equations. |

Not rendered from this deck: slides 11, 15 and 26, build steps of slides 12, 16 and 27; slides 4,
10, 33 and 34, equation slides whose annotations are transcribed; and the title, reminder, plan and
text slides.

### Lecture 5 — Off-Policy Actor Critic

Deck: [05_cs224r_offpolicy_actor_critic_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf).
Rendered: 15 slides, in `raw/images/05-off-policy-actor-critic/`. Slides 15, 16 and 19–21 print
"Slide adapted from Sergey Levine".

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 13 | A "HalfCheetah-v1" learning-curve chart (reward against time steps collected, PPO against vanilla policy gradient and four unlabelled curves) and a small Half-Cheetah simulator render. Not credited. |
| 15, 16, 19, 20 | Slides "adapted from Sergey Levine": the replay-buffer timeline (15) and algorithm slides with hand-drawn annotations (16, 19, 20). |
| 23 | Two learning-curve charts, "(e) Humanoid-v1" and "(f) Humanoid (rllab)", comparing SAC, DDPG, PPO, SQL and TD3, pasted from a paper the slide does not name, and a cropped simulated-humanoid render. |
| 24 | A video frame of a four-legged robot, credited to Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan, Kumar, Zhu, Gupta, Abbeel, Levine, *Soft Actor-Critic Algorithms and Applications* (2018). |
| 25 | Two video frames of robot assembly (motherboard, timing belt) and a results table, credited to Luo, Xu, Wu, Levine, *HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning* (2024). |
| 26 | A simulator frame of a quadruped robot, credited to Tan, Zhang, Coumans, Iscen, Bai, Hafner, Bohez, Vanhoucke, *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots* (2018). |
| 27 | A photograph of a robot hand and a grid of simulated renderings, credited to OpenAI, *Solving Rubik's Cube with a Robot Hand* (2019). |
| 3, 4, 5, 10, 18 | No third-party content identified: the deck's own loop diagrams, trajectory sketches, neural-network drawing and annotated equations. |

Not rendered from this deck: slide 12, whose only picture is a comic-book meme, credited on the
slide only as "from one of the GAE authors"; slides 8 and 9, build steps of slides 7 and 10; slides
7, 11, 17, 21 and 22, equation slides whose annotations are transcribed; and the title, reminder,
plan and text slides.

### Lecture 6 — Q-Learning

Deck: [06_cs224r_qlearning_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf).
Rendered: 12 slides, in `raw/images/06-q-learning/`. Slides 8, 15, 17 and 19–23 print "Slide
adapted from Sergey Levine".

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 8, 15 | Slides "adapted from Sergey Levine": the greedy-policy argument with the three-box loop (8), and the full algorithm with a diagram of the world (an Earth-globe icon), the replay buffer and off-policy Q-learning (15). |
| 19 | Slide "adapted from Sergey Levine": four training-curve charts (average reward and average predicted Q on Breakout and Seaquest) and two figure panels of Atari game frames with value and action-value charts, pasted from sources the slide does not name. |
| 20 | Slide "adapted from Sergey Levine": a four-panel chart of value estimates against true values for DQN and Double DQN (Alien, Space Invaders, Time Pilot, Zaxxon), not credited on the slide. |
| 26 | A bar chart of Atari results, credited to "Mnih et al. 2015"; a video frame of robot arms grasping from bins and a results table comparing QT-Opt with "Levine et al. [27]", credited to "Kalashnikov et al. 2018". |
| 3, 4, 5, 7, 12, 18, 24 | No third-party content identified: the deck's own trajectory sketches, loop diagrams, neural-network drawing, 2D-navigation sketch and annotated algorithms and equations. Slide 18 cites Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013) for the algorithm, not for an image. |

Not rendered from this deck: slides 9–11, 13, 14, 17, 21–23 and 25, equation slides whose
strikethroughs, highlights, loop arrows and annotations are transcribed (17 and 21–23 print "Slide
adapted from Sergey Levine"); and the title, reminder, plan, advice and next-time slides (1, 2, 6,
16, 27, 28, 29).

### Lecture 7 — Offline RL

Deck: [07_cs224r_offline_rl_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/07_cs224r_offline_rl_2025.pdf).
Rendered: 14 slides, in `raw/images/07-offline-rl/`. Slides 25–28 print "Slide adapted from Sergey
Levine".

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 22 | Two charts, the expectile loss for five values of lambda and expectile curves fitted to a 2D scatter of points; the slide cites Kostrikov, Nair, Levine, *Implicit Q-Learning*, ICLR '22, and does not credit the charts separately. |
| 25, 26 | Slides "adapted from Sergey Levine": the randomly initialized Q-function plot with the Q-value penalty (25), and the conservative Q-learning objective with its coloured labels (26). |
| 31 | A system-architecture diagram of LinkedIn's notification pipeline and a results table ("Table 1: Online A/B test results for DDQN with and without CQL"), credited to Prabhakar, Yuan, Yang, Sun, Muralidharan, *Multi-Objective Optimization of Notifications Using Offline RL* ('22). |
| 3, 4, 5, 11, 14, 16, 17, 19, 21, 23 | No third-party content identified: the deck's own loop diagrams, annotated equations, Q-function plot, two-trajectory stitching diagram, histogram and loss sketch, and algorithm slides. Slides 16–17 cite Peters et al. (REPS) and Rawlik et al. ("psi-learning"), slides 18–19 Peng, Kumar, Zhang, Levine, *Advantage-Weighted Regression* ('19), and slides 21 and 23 Kostrikov, Nair, Levine, *Implicit Q-Learning*, for the methods, not for an image. |

Not rendered from this deck: slides 8, 10, 15, 18, 20, 27 and 28, equation slides whose underlines,
labels, strikethrough, loop arrows and circles are transcribed (27 and 28 print "Slide adapted from
Sergey Levine"); and the title, reminder, plan, text, summary and recommendation slides (1, 2, 6, 7,
9, 12, 13, 24, 29, 30, 32, 33).

### Lecture 8 — Reward Learning

Deck: [08_cs224r_reward_learning_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf).
Rendered: 18 slides, in `raw/images/08-reward-learning/`. Slides 5–8 print "Slide adapted from Sergey
Levine". The deck's text layer has no licence notice; slide 13's embedded video frame (not rendered)
prints "© Warneken & Tomasello" in the image.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 5, 6, 8 | Slides "adapted from Sergey Levine": the randomly initialized Q-function plot with the Q-value penalty (5), the conservative Q-learning objective with its coloured labels (6), and the computation of the CQL objective (8). |
| 9 | A system-architecture diagram of LinkedIn's notification pipeline and a results table ("Table 1: Online A/B test results for DDQN with and without CQL"), credited to Prabhakar, Yuan, Yang, Sun, Muralidharan, *Multi-Objective Optimization of Notifications Using Offline RL* ('22). |
| 12 | An Atari Breakout screenshot captioned "Mnih et al. '15"; a photograph of a toddler at a table with a pitcher and glass, speech-bubble clip-art and an aerial photograph of a motorway, not credited. |
| 15 | Photographs of a robot workspace (a notebook, a pencil case and a robot arm), not credited. |
| 19 | Two video frames of a robot on a cloth task, from the work the slide cites: Sharma, Ahmed, Ahmad, Finn, *Self-Improving Robots: End-to-End Autonomous Visuomotor Reinforcement Learning* ('23). |
| 20 | Three generated images of Yorkshire terriers, credited to Yu et al., *ViT-VQGAN* (ICLR '22), and a video frame from Villegas et al., *Phenaki* (ICLR '23). |
| 27 | A simulation screenshot and a chart of learned reward-weight densities with a road diagram, from the two works the slide cites: Christiano et al., *Deep Reinforcement Learning from Human Preferences* (NeurIPS 2017), and Sadigh et al., *Active Preference-Based Learning of Reward Functions* (RSS 2017). |
| 28, 29 | Screenshots of a chat prompt ("How do I cook an omelette?") and two replies, with speech-bubble and trophy icons, not credited. |
| 30 | A helpfulness-against-harmlessness Elo chart, credited to Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022). |
| 32 | A three-panel grid-world self-play diagram, credited to Sukhbaatar et al., *Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play* (ICLR 2018). |
| 4, 17, 23, 25, 26 | No third-party content identified: the deck's own recap with its Q-function plot, the goal-classifier algorithm, the trajectory sketches, and the preference objective and algorithm. |

Not rendered from this deck: slides 2 and 34, whose only pictures are photographs of the two guest
lecturers; slide 13, whose embedded video frame is black; slides 7, 18 and 24, equation and text slides
whose loop arrow, emoji and brace are transcribed (7 prints "Slide adapted from Sergey Levine"); and the
title, plan, summary and recap slides (1, 3, 10, 11, 14, 16, 21, 22, 31, 33).

### Lecture 9 — RL for LLMs: Preference Optimization

Deck: [09_cs224r-2025-rlhf.pdf](https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf),
a guest lecture by Archit Sharma whose title slide reads "Based on slides in CS224N". Rendered: 24
slides, in `raw/images/09-rl-for-llms/`. The deck's text layer has no licence notice.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 2 | The Economist's chart "The blessings of scale" (sources printed on it: Sevilla et al., "Compute trends across three eras of machine learning", arXiv, 2022; Our World in Data), and a screenshot of the AI 2027 website (ai-2027.com). |
| 4 | Screenshots of the ChatGPT, Claude and Gemini start screens, not credited. |
| 7 | A screenshot of a Khan Academy SAT page on circle equations, with its graph (khanacademy.org). |
| 43 | The FLAN-T5 instruction-finetuning diagram, credited "FLAN-T5; Chung et al., 2022". |
| 50 | The Flan-T5 results table, credited to Chung et al., 2022, beside a sketch of the T5 encoder-decoder (Raffel et al., 2018). |
| 61, 80 | The InstructGPT three-step pipeline diagram, credited on slide 80 to Ouyang et al., 2022. |
| 72, 94 | Charts of summary preference against model size (72) and of reward-model over-optimization (94), credited to Stiennon et al., 2020. |
| 74 | A diagram of the PPO-based RLHF pipeline, credited to Zheng et al. 2023, "Secrets of RLHF". |
| 78 | Bar charts of summarization and dialogue win rates; no source printed on the slide. |
| 87 | Screenshots of the Hugging Face Open LLM Leaderboard (with handwritten annotations), the Mistral AI and Mixtral "8X7" logos with an MT-Bench table (mistral.ai), and a Meta blog section with the Llama 3 logo. |
| 92 | A still from a boat-racing game, from OpenAI's post on faulty reward functions (openai.com). |
| 98 | The title block of Huang et al., "Large Language Models Can Self-Improve" (2022), and a diagram of the Self-Taught Reasoner (STaR, Zelikman et al., 2022). |
| 99 | Figures from the PRISM Alignment Project (Kirk et al., 2024) and from FSPO: Few-Shot Preference Optimization (Singh, Hsu et al., 2024). |
| 41, 42, 56, 63, 64, 67, 70, 76, 77 | No third-party content identified: the deck's own diagrams of pretraining and finetuning and of a language model's outputs, and its policy-gradient, reward-model, RLHF and DPO equations. |

Not rendered from this deck: slide 12, identical to slide 4; slides whose pictures are screenshots or
panels of text transcribed in full, or clip-art (6, 8, 9, 10, 39, 40, 51, 52, 60, 65, 81, 82, 83, 84,
85, 88, 93, 95, 96, 97), among them a paper excerpt (Andreas, 2022), GitHub Copilot, a paper abstract
(Larnerd, 2023), Bing chat, OpenAI's ChatGPT and sycophancy posts, AlpacaFarm (Dubois et al., 2023),
NPR, AP and Semafor headlines, and Constitutional AI (Bai et al., 2022); and the title, plan,
section-divider, text and equation slides (1, 5, 13, 14, 57, 62, 66, 69, 73, 75, 79, 89).

### Lecture 10 — RL for LLM Reasoning

Deck: [10_cs224r-rl_for_reasoning_lecture.pdf](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf),
a guest lecture by Aviral Kumar (title slide with the Carnegie Mellon University School of Computer
Science logo). Rendered: 27 slides, in `raw/images/10-rl-for-reasoning/`. The deck's text layer has no
licence notice.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 4 | A chart of the effective stock of internet text against projected dataset sizes, with named models, and two robot-dataset infographics, one titled "DROID, Distributed Robot Interaction Dataset"; no source printed on the slide. |
| 7 | A math problem with its ground-truth solution; no source printed on the slide. |
| 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | Charts, diagrams and a GSM8K example from Setlur, Garg, Geng, Garg, Smith and Kumar, "RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold" (NeurIPS 2024), credited in each footer; the lecturer is an author. |
| 24, 25, 27, 28 | Diagrams and charts from Setlur et al., "Rewarding Progress: Scaling Automated Process Supervision for LLM Reasoning" (arXiv 2024), credited in each footer; the lecturer is an author. |
| 35 | A figure from Gandhi et al., "Cognitive Behaviors that Enable Self-Improving Reasoners" (arXiv 2025), with two model logos that the slide does not name. |
| 36 | A figure from the Kimi team's "Kimi 1.5 Technical Report" (arXiv 2025). |
| 39 | The deck's test-time scaling diagram, citing Snell et al., "Scaling LLM Test-Time Compute Optimally can be more Effective than Scaling Model Parameters" (ICLR 2025). |
| 40 | Bar charts from OpenAI's o3 and o4-mini blog post (2025), credited in the footer, and a bar chart comparing DeepSeek and OpenAI models on AIME 2024, not separately credited. |
| 3, 9, 34, 41 | No third-party content identified: the deck's own diagrams of next-token training, the data-scaling axes, the thinking-model objective and the open questions. |

Not rendered from this deck: panels of text transcribed in full (2, AIME 2025 questions with a
solution signed by its author; 5, an IMO 2021 problem with a model's solution; 31, a thinking trace
from Marjanović et al., "DeepSeek-R1 Thoughtology", arXiv 2025); slide 26, a text pop-up over a faded
copy of earlier slides; the build steps 33, 37 and 38; and the title, outline, section-divider and
text slides (1, 6, 8, 13, 29, 30, 32).

### Lecture 11 — Model-Based RL

Deck: [11_cs224r_mbrl_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf),
given by Chelsea Finn. Rendered: 17 slides, in `raw/images/11-model-based-rl/`. The deck's text layer has
no licence notice.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 6 | Stills from videos generated by the models the slide names, "Veo 2" (a city bus) and "Sora" (an aerial view of a city); no further source printed. |
| 8 | A maze image, no source printed, and a frame of a video generated by Sora from the prompt printed under it. |
| 18, 20 | A photograph of a grassy cliff edge above the sea, no source printed, with the deck's own hand-drawn arrows. |
| 21, 22 | Clip-art of a person and a globe (21) and a drawing of a car on a road (22), no source printed, in the deck's own diagrams. |
| 33, 34, 35, 36 | Simulator renderings, charts, photographs and video frames of the robot hand from the case-study paper named on slide 32, "Deep Dynamics Models for Learning Dexterous Manipulation" by Nagabandi, Konoglie (as printed), Levine and Kumar (Google Brain); the slides print no per-slide credit. |
| 4, 7, 12, 13, 17, 19, 27 | No third-party content identified: the deck's own recap map, algorithm sketches, planning chains, optimization plots and roll-out diagram. |

Not rendered from this deck: slide 5, an agenda whose video frame of the robot hand shows the scene of
slide 36; the build steps 14, 15 and 26; slide 32, the case-study paper's title block, transcribed in
full; text, equation and algorithm slides (10, 16, 23, 25, 28, 29, 30); and the title, divider, agenda
and reminder slides (1, 2, 3, 9, 11, 24, 31, 37, 38).
