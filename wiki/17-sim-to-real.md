# Lecture 17 — Advancing Robot Intelligence with Reinforcement Learning

Stanford CS224R, Spring 2025 (Wednesday, May 28) · guest lecture by Ashish Kumar ·
[video](https://www.youtube.com/watch?v=Hp1WBWghrak) (50 min) ·
[slides (PDF)](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf) ·
[slide-by-slide text](../raw/slides/17-sim-to-real.md) ·
[transcript](../raw/transcripts/17-sim-to-real.md)

**What this lecture establishes.** Kumar contrasts the imitation learning and the reinforcement learning
that have actually worked: imitation relies on off-policy data and gives generalists; RL that has succeeded
is on-policy, learns from good and bad data and gives specialists. RL's big successes (AlphaGo, reasoning in
language models) share two ingredients: **well-specified rewards** that can be computed programmatically, and
**scalable evaluation of policies**. For robots, simulation supplies both, so the problem becomes
**sim-to-real transfer**. The first part presents **Rapid Motor Adaptation (RMA)**: a walking policy trained
in simulation with PPO, given a compressed vector of the simulator's physical parameters (the
**extrinsics**), then an **adaptation module**, trained with DAgger, that estimates the extrinsics online
from the recent history of states and actions, because the real world does not reveal them. One set of
weights, deployed zero-shot, walks blind over rocks, sand, oil, planks and heavy payloads. The second part
adds vision: rather than building a noisy terrain map, a policy first trained with perfect terrain heights is
distilled into one that reads **egocentric depth** directly, and it climbs stairs as tall as itself with a
gait and footstep planning that emerge from RL. The same recipe of proprioceptive history and sim-to-real
transfer handles in-hand object rotation and drone flight. The lecture closes on what is left: humanoids,
manipulation that is reliable and dexterous, and Kumar's speculation that better simulation (betting on
compute, as in the bitter lesson) and a general reward model might get there, and that robots might one
day exceed human capability.

Slide numbers below are the slides' own numbers, which equal the PDF page numbers (the deck prints a number
only on slides 62–67). Timestamps (≈M:SS) point into the video and the
[transcript](../raw/transcripts/17-sim-to-real.md). **The recording covers slides 2–78.** It opens on the
contrast between imitation and reinforcement learning (slide 2, ≈0:05), follows the deck in order and ends
on slide 78's question (≈49:22, ending at about 49:45). Between introducing the vision paper (slide 41,
≈22:24) and its first content slide (slide 42, ≈28:32), Kumar answers seven student questions, mostly
about the first part; they are collected [below](#questions-on-rma). Only Kumar is audible, so the
questions are known from Kumar's restatements. Most of the deck's pictures are frames of videos (slides
5–11, 22–27, 30, 31, 33–36, 45, 58, 62, 67, 69, 70, 73 and 74, and most likely 55 and 56); the PDF holds
only one frame of each, and what they showed comes from the transcript. Where the transcript says "here" or
"this", this page reconstructs what was meant from the surrounding words and the slides. Kumar's pronouns
are not stated in the material, so this page writes "Kumar".

This is the second of the course's two robotics lectures; [Lecture 16](16-autonomous-rl.md) covered
learning without resets, and its slide 44 already showed RMA's quadruped as an example of adapting in
low-level action space. For the concept across lectures see [sim-to-real transfer](sim-to-real.md).

## Contents

1. [Imitation and reinforcement learning that work](#imitation-and-reinforcement-learning-that-work)
2. [What RL has achieved, and what enabled it](#what-rl-has-achieved-and-what-enabled-it)
3. [Zooming in on robotics: simulation to real](#zooming-in-on-robotics-simulation-to-real)
4. [One policy to walk them all](#one-policy-to-walk-them-all)
5. [RMA: learning to walk in simulation](#rma-learning-to-walk-in-simulation)
6. [The reward function](#the-reward-function)
7. [Deploying it: extrinsics from the observation history](#deploying-it-extrinsics-from-the-observation-history)
8. [Training the adaptation module](#training-the-adaptation-module)
9. [Test time: two rates](#test-time-two-rates)
10. [Indoor evaluation](#indoor-evaluation)
11. [Switching adaptation off](#switching-adaptation-off)
12. [What the adaptation module does](#what-the-adaptation-module-does)
13. [Quantitative comparison](#quantitative-comparison)
14. [Questions on RMA](#questions-on-rma)
15. [Why vision?](#why-vision)
16. [Terrain maps, and why not to build them](#terrain-maps-and-why-not-to-build-them)
17. [The vision policy in deployment](#the-vision-policy-in-deployment)
18. [Phase 1: walking with privileged terrain information](#phase-1-walking-with-privileged-terrain-information)
19. [Phase 2: walking with egocentric depth](#phase-2-walking-with-egocentric-depth)
20. [What the policy learned: stairs for a small robot](#what-the-policy-learned-stairs-for-a-small-robot)
21. [Comparison with baselines](#comparison-with-baselines)
22. [Application to dexterous manipulation](#application-to-dexterous-manipulation)
23. [Application to drone flight](#application-to-drone-flight)
24. [What is left to solve: humanoids](#what-is-left-to-solve-humanoids)
25. [Why manipulation lags behind locomotion](#why-manipulation-lags-behind-locomotion)
26. [Simulation, and the bitter lesson](#simulation-and-the-bitter-lesson)
27. [A general reward model?](#a-general-reward-model)
28. [Beyond parity with humans](#beyond-parity-with-humans)

## Imitation and reinforcement learning that work

Before zooming in on robotics, Kumar contrasts reinforcement learning with
[imitation learning](imitation-learning.md), with a caveat: "I'm going to compare imitation learning that
actually works with reinforcement learning that actually works" (≈0:05). Other RL algorithms exist that do
not share these characteristics; these are the characteristics of the ones that have been most successful at
solving extremely hard problems. Slide 2 sets the two side by side.

- **Imitation learning** is used in vision-language models, video generation and image segmentation. It
  relies heavily on **off-policy data**: data collected or generated by humans, or from sources that "have
  nothing to do with the final policy that you're improving" (≈0:51–1:37). You take the dataset, clean it up
  and train on the good data: it "Learns from good examples". It typically gives **generalists**, one single
  model that can do a bunch of things (slide 2, ≈2:22).
- **Reinforcement learning** that has succeeded is used in game playing (Go, and games in simulation),
  reasoning for large language models (beyond what imitation pre-training achieved) and robotics. It is
  typically **on-policy**: the policy you want to improve is the one you roll out to generate the data, and
  "every data point that is generated during the policy rollout is fair game" (≈1:37). "So you learn from not
  just the positive but also the negative data", which gives very precise behaviours. It either gives
  **specialists** or helps specialize pre-trained general models (slide 2, ≈2:22).

"That is not to say that one is strictly better than the other" (≈2:22). The two are combined: for Go,
imitation learning was used "for warm starting the reinforcement learning self-play search", and language
models are warm-started with imitation before reasoning is improved with RL (≈2:22–3:09). See
[offline and online data](offline-and-online.md) for the course's own account of on- and off-policy data.

## What RL has achieved, and what enabled it

Slide 3 lists what RL has qualitatively achieved, behaviours for which it is "the only way to do those
things" (≈3:09):

- **AlphaGo**: "Learned to exceed what humans can do", defeating the best Go player in the world by
  figuring out innovative behaviours humans had not seen before (slide 3, citing Silver et al., *Mastering
  the game of Go with deep neural networks and tree search*, 2016).
- **Reasoning in LLMs**: "Coherent over long horizon": RL goes significantly beyond imitation learning,
  showing "very coherent and precise reasoning over very long horizon tasks" (≈3:54; slide 3 cites Guo et
  al., *DeepSeek-R1*, 2025). See [LLM reasoning](llm-reasoning.md).
- **Dexterity in robotics**: "Reliability", the rest of the talk.

What is common to all three? Kumar points out two things that let on-policy RL shine, printed under "What
enabled these successes ?" (slide 3, ≈3:54–5:25):

1. **Well-specified rewards.** AlphaGo's reward was very sparse and very clear: whether you won the game,
   which you can define from the state. (The transcript is unclear on one word here: "the state of the word",
   probably "world" or "board".) For reasoning, the DeepSeek-R1 paper uses "a rule-based reward function
   which is very clear to evaluate", and because it is rule-based it can be automated and computed cleanly at
   scale (≈4:40).
2. **Running and evaluating policies at scale.** Since the rules of Go are known, the game can be put in
   simulation, where "it scales beautifully with compute. So we can just throw compute at it": evaluating
   the policies converts compute into data, which RL then uses to improve the model. Reasoning models are
   rolled out in a cluster and scored with an automatic reward function, "an extremely scalable way" of
   collecting data, evaluating and improving (≈4:40–5:25; see [the RL objective](rl-objective.md)).

These results are spaced out by seven to eight years, Kumar observes, and all still rest on these two
ingredients (≈5:25).

## Zooming in on robotics: simulation to real

Slide 4 ("Zooming in on robotics") maps the two ingredients onto robots: "Well specified rewards —>
Programmatically calculate them" and "Scalable evaluation of policies —> Simulation", ending on "Simulation
to real?". The closest approximation to the ingredients is simulation: there the full state of the world is
available, so rewards can be computed programmatically, and policies can be evaluated at scale because "we
have converted a physical problem to a digital problem which now just will scale with compute" (≈6:11).
"So this gets us to solving robotics in simulation but that is not the same as solving robotics in the real
world." Getting to the real world means tackling the **sim-to-real** challenge, training policies in
simulation and moving them to the real world, which is the focus of the talk (≈6:11). The results Kumar
presents are from three to four years before the lecture (≈5:25).

## One policy to walk them all

To show first that this works, Kumar plays seven real-world videos, one frame of each on slides 5–11
(≈6:57–8:29). Everything runs in real time on board, and the robot is blind. In Berkeley it tries to get
over rocks, its feet getting stuck until it figures out that something is there and continues; it gets up
staircases it cannot see, "a bit fumbly" but without falling; on a grass patch even vision would not show
the ground underneath, so a certain robustness is needed; on ground sloping to the left beside a big ditch it
does not drift; at a construction site in front of Kumar's house it reliably gets down very unstable ground;
on slightly muddy ground its feet get stuck once in a while and it corrects by applying more force; and it
walks on sand. Slide 12 gathers the frames under its title, "One Policy to Walk them All".

![Slide 12 — One Policy to Walk them All](../raw/images/17-sim-to-real/slide-12.jpg)

*Slide 12 — twelve frames from the real-world videos (rocks, leaf litter, ground cover, a living room, sand, a beach, a gravel pile by a road, mulch, a dirt path), every one walked by the same blind policy. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The important point is that all of these results come from the exact same policy: "So the exact same
weights deployed zero-shot in the real world without any tuning or specific improvement related to any of
these terrains" (≈8:29). The work was published at RSS in 2021 (≈8:29): slide 13's title page, *Rapid Motor
Adaptation for Legged Robots*, by Ashish Kumar (UC Berkeley), Zipeng Fu (CMU), Deepak Pathak (CMU) and
Jitendra Malik (UC Berkeley/FAIR).

## RMA: learning to walk in simulation

The idea starts simple: "Let's just pretend that this simulation is the real world and everything we can
access is fair game" (≈8:29). A neural network, the **base policy**, takes an action in simulation; the
simulator simulates one tick of the real world and returns the updated state, which the policy consumes, and
so it walks (slide 14).

On slide 14 the base policy's inputs are the current state and the previous action, printed $x(t)$ and
$a(t-1)$ (Kumar: "we use the current state and the previous action", ≈26:59; these are the deck's names for
what the course writes $\mathbf{s}_ t$ and $\mathbf{a}_ {t-1}$), together with a vector $z_t$. Its output is
the action $a_t$, sent to the "Physics Simulation". The vector $z_t$ comes from an **environmental factor
encoder**, whose input is the simulator's physical parameters: "Mass", "COM" (centre of mass), "Friction",
"Terrain Height" and "Motor Strength". Dashed lines return the simulator's state to $x(t)$ and its
parameters to the encoder.

![Slide 14 — Learn to Walk in Simulation](../raw/images/17-sim-to-real/slide-14.jpg)

*Slide 14 — the simulator's physical parameters pass through the environmental factor encoder to the extrinsics z_t, which the base policy reads alongside the state and previous action; the two red-outlined modules are trained. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

A policy trained in one setting would be very narrow, walking in one scenario, perhaps deployable by tuning
the physics parameters. That is not good enough: the goal is "a generalist policy that can work in all of
these scenarios" (≈9:14). So every parameter is **randomized** in simulation, and, the important part, the
policy is given access to them. It then sees a huge range of masses and frictions and learns to walk
accordingly: "So it's not oblivious or blind to the environment. It's actually aware of what it's dealing
with" (≈9:14). The compressed vector $z_t$ is called the **extrinsics** (slide 14's green label,
≈9:14–10:00).

The reward is to track a target velocity, plus a set of terms that minimize work and ground impact (slide
15, under an underlined "Important": "Reward Function minimizes work and ground impact (Biomechanics and
Energetics)"). The whole
system is trained end to end with [PPO](ppo.md), and "Takes about a billion samples to train it in this
simulation" (≈10:00). (A short phrase after that sentence is garbled in the captions.)

![Slide 15 — Learn to Walk in Simulation](../raw/images/17-sim-to-real/slide-15.jpg)

*Slide 15 — the same training loop with a simulated robot on jagged terrain, and the note that the reward minimizes work and ground impact. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## The reward function

Slide 16 lists the ten reward terms on three coloured bands. "This is not very elegant, but we tried to be
as principled as we can" (≈10:00). The first terms specify the task; then come terms that minimize
energetics; and the last group is about stability and minimizing hardware damage, "because in simulation
there's no concept of hardware damage", so those biases have to be injected for the behaviour to transfer to
the real world (≈10:00–10:45). (The captions are unclear where Kumar names the first terms: "So the forward
and the latter.") The slide does not define its symbols; the terms' printed names say what each measures, a
superscript $t$ marks the time step, and every term but the first is a penalty.

Forward walking (light blue band):

1. Forward: $\min(v_x^t, 0.35)$
2. Lateral Movement and Rotation: $-\Vert v_y^t \Vert^2 - \Vert \omega_{\texttt{yaw}}^t \Vert^2$

Energetics (light green band):

3. Work: $-\lvert \boldsymbol{\tau}^T \cdot (\mathbf{q}^t - \mathbf{q}^{t-1}) \rvert$
4. Ground Impact: $-\Vert \mathbf{f}^t - \mathbf{f}^{t-1} \Vert^2$

Stability and minimizing hardware damage (light red band):

5. Smoothness: $-\Vert \boldsymbol{\tau}^t - \boldsymbol{\tau}^{t-1} \Vert^2$
6. Action Magnitude: $-\Vert \mathbf{a}^t \Vert^2$
7. Joint Speed: $-\Vert \dot{\mathbf{q}}^t \Vert^2$
8. Orientation: $-\Vert \boldsymbol{\theta}_ {\texttt{roll, pitch}}^t \Vert^2$
9. Z Acceleration: $-\Vert v_z^t \Vert^2$
10. Foot Slip: $-\Vert \text{diag}(\mathbf{g}^t) \cdot \mathbf{v}_ {\mathbf{f}}^t \Vert^2$

How are so many terms balanced? Asked this (≈26:14–26:59), Kumar answers: "There's no good way to do it.
That is the real answer." The practical way is to add one reward at a time: see what the current set lacks,
add a new term, and tune only that term relative to all the others. That reduces the exponential complexity
of tuning them jointly to adding one at a time, though the coefficients found are probably not the most
optimal. See [reward learning](reward-learning.md) for the course's other routes to a reward.

## Deploying it: extrinsics from the observation history

The extrinsics vector is unknown in the real world (slide 17: the parameter box is "Unavailable" and $z_t$
is "Unknown Extrinsics"). Measuring it once would not do either: even estimated in Kumar's living room, "the
moment I get to something which is not my living room, let's say sand or mulch or any of those, it would fail
to work and I have to retune it again" (≈10:45). So the key idea is to estimate it "online in fractions of a
second at runtime" (≈10:45).

![Slide 17 — How can we deploy it?](../raw/images/17-sim-to-real/slide-17.jpg)

*Slide 17 — at deployment the physical parameters are unavailable, so the extrinsics the base policy was trained with are unknown. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The signal for estimating it is the **observation history**: the commands issued and what was actually
executed (≈11:32). It carries the right information because of the discrepancy between them. Say you command
a walk and one leg slips more than expected: the discrepancy between the signal and what you commanded tells
you that you slipped. "This was not intentional. You slipped forward" (≈11:32). Slide 18's key insight is
"Extrinsics from Observation History": an **adaptation module** reads the last 50 steps of states and
actions, $x(t-50), a(t-50)$ through $x(t-1), a(t-1)$, and outputs $z_t$ for the base policy. Its bullets:
"Discrepancy b/w expected movement and actual measured movement" and "Continuously estimate these
extrinsics online" (slide 18, ≈11:32–12:18).

![Slide 18 — Key Insight — Extrinsics from Observation History](../raw/images/17-sim-to-real/slide-18.jpg)

*Slide 18 — the adaptation module replaces the encoder at deployment, estimating z_t online from the last 50 states and actions. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Training the adaptation module

To train the adaptation module, go back to simulation, where both the ground truth and the history are
available (≈12:18). The method is [DAgger](imitation-learning.md#fixing-it-with-corrective-data): "roll out
the student policy to generate the data distribution and then you use the teacher to supervise every single
point of the student policy" (≈12:18). The student, the adaptation module, uses the history of observations
to continuously estimate the extrinsics $z$, which feed the base policy trained with reinforcement learning,
whose weights are copied over (≈12:18–13:03).

Slide 19's "Training Summary" shows the two phases. In **Phase 1** the environmental factor encoder and the
base policy are trained together (both outlined in red, "*Trainable Modules in Red"). In **Phase 2** the
adaptation module is trained (in red), its $z_t$ regressed onto Phase 1's $z_t$ (the green "Regress" arrow),
while the base policy, now outlined in black, is carried down from Phase 1 unchanged.

![Slide 19 — Training Summary](../raw/images/17-sim-to-real/slide-19.jpg)

*Slide 19 — Phase 1 trains the encoder and base policy with privileged parameters; Phase 2 trains the adaptation module to regress the same z_t from history, with the base policy carried over. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

Asked later to explain DAgger here (≈27:45–28:32), Kumar walks through it. Initialize the adaptation module
with random weights and run the Phase 2 policy to collect data; at first it is bad data, since nothing
meaningful has been learned. Use the teacher encoder to supervise, at every time step, what the prediction
should have been, and update the adaptation module with supervised learning. Roll it out in simulation
again; it will hopefully be better, but still not optimal; supervise it again with the teacher, run
supervised learning on the new dataset, and repeat.

## Test time: two rates

At test time (slide 20) the base policy runs at 100 Hz and the adaptation module at only 10 Hz. "This is not
by choice it's because of whatever computers available on board", and this was not accounted for in
training, where everything ran at 100 Hz in lock step; yet "we see that the performance doesn't drop", and
the mismatch is good enough for the results (≈13:03).

![Slide 20 — Test Time](../raw/images/17-sim-to-real/slide-20.jpg)

*Slide 20 — at test time the adaptation module runs at 10Hz and the base policy at 100 Hz. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Indoor evaluation

The indoor evaluation (slide 21) is in Kumar's living room, shown in videos (≈13:03–14:38). On an oily
plastic tarp ("I think that is olive oil"), with the robot's feet covered in plastic, "Partially to make it
challenging, but also to protect the feet of the robot", it slips a bit but continues to march forward;
slowed down, it starts slipping and maintains balance without flipping over (slides 22–24).

![Slide 23 — Oily surface and plastic wrapped feet](../raw/images/17-sim-to-real/slide-23.jpg)

*Slide 23 — a frame of the oily-surface video: the plastic tarp in front of the fireplace, with the robot at the right edge. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

A 5 kg payload thrown onto a 12 kg robot is roughly half its weight; slowed down, the robot goes down, and
"If you focus on the green dots in a few steps, it tries to go back up and recover the regular motion style
that it is working with" (slides 25–26, ≈13:48–14:38). On planks (slide 27) the foothold is not even stable,
since the plank moves with the foot, so there is no clear estimate of the ground, and it still walks.

![Slide 27 — Planks - uneven footholds and moving surfaces](../raw/images/17-sim-to-real/slide-27.jpg)

*Slide 27 — loose wooden boards on the foam floor: uneven footholds that move under the foot. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

All of these results are blind: "There's no vision. All of these just use proprioception history" (≈14:38).
Kumar evaluated it in many more places, including a mattress on foam and steps down and up, and it was
relatively robust in all of them (slide 28, "RMA - Indoors", ≈14:38).

![Slide 28 — RMA - Indoors](../raw/images/17-sim-to-real/slide-28.jpg)

*Slide 28 — twelve indoor tests in the same living room: boards, foam blocks, stacked slabs, a plastic-covered mat and a mattress. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Switching adaptation off

What if the adaptation module is switched off (slide 29)? Then the extrinsics are estimated at the first
time step and frozen for the rest of the run (≈14:38–15:25). With an 8 kg payload on the 12 kg robot, whose
rated payload when bought was around 3 kg, RMA struggles a bit, and heats the motors faster, but walks
(slide 30). Without adaptation the weight tilts the robot forward and it does not correct, by applying more
torque in the front legs or getting back to a higher clearance, so it ends up tilting forward without enough
torque to clear the next step (≈15:25).

![Slide 30 — RMA and RMA w/o Adaptation with an 8kg payload](../raw/images/17-sim-to-real/slide-30.jpg)

*Slide 30 — split screen of the 8 kg payload test, RMA above and RMA without adaptation below. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The same holds on foam (slide 31): with adaptation the robot manages even when it gets stuck; without it, it
does not apply more force for a higher clearance, "because it hasn't detected that event or that change"
(≈15:25–16:12).

![Slide 31 — RMA w/o Adaptation and RMA on foam](../raw/images/17-sim-to-real/slide-31.jpg)

*Slide 31 — the foam-block comparison, RMA above and RMA without adaptation below. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## What the adaptation module does

To see what the module is doing (slide 32, ≈16:12), Kumar replays the oily-surface video, flipped, with
three plots that run with it (slide 34). The top is a **gait plot**: one row per foot, solid wherever that
foot is in contact, measured by foot sensors on the hardware. The second is the torque requested at the
front-right knee ("Knee Tau"). The third plots two values from the adaptation module's output, chosen
because "they show the most variation" in the run (≈17:00); the slide labels them $z_1$ and $z_5$.

![Slide 34 — gait, knee torque and two extrinsics on the oily surface](../raw/images/17-sim-to-real/slide-34.jpg)

*Slide 34 — when the robot slips (about the middle), the gait rows shuffle, the knee torque turns irregular, and the two extrinsics swap levels, crossing; then the gait and torque settle back. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The moment the robot slips, the gait shuffles, the torque profile varies, and the adaptation module changes
its values "to capture that something has happened" (≈17:00). "Since these are learned extrinsics, we
cannot interpret them very easily", but some change has happened, and afterwards the gait and torque return
to the same pattern as before once the event is detected (≈17:00).

The payload test gives the same picture (slide 36, ≈17:46). The 5 kg payload is dropped, so the force at
impact is probably even higher; the gait shuffles and the torque is very high at the moment of the drop. The
gait returns to normal, but the torque settles higher than before, "because it needs more torque to balance
the weight that it is carrying" (≈17:46). Here the slide plots $z_2$ and $z_7$.

![Slide 36 — gait, knee torque and two extrinsics after a payload drop](../raw/images/17-sim-to-real/slide-36.jpg)

*Slide 36 — after the payload lands, the knee torque spikes and then settles on higher plateaus, and the two extrinsics move apart and stay apart. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Quantitative comparison

Slide 38 ("Existing Attempts at Learning General Policies") compares RMA with three existing approaches,
listed beneath the table as "Domain Randomization (Robust)", "System Identification" and "Fine tuning at
test time in the real world". The slide prints no units beyond those in the header, and Kumar does not
explain TTF.

| Method | Success (%) | TTF | Reward | Distance (m) | Samples | Torque | Jerk | Ground Impact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Robust [49, 38] | 62.4 | 0.80 | 4.62 | 1.13 | 0 | 527.59 | 122.50 | 4.20 |
| SysID [54] | 56.5 | 0.74 | 4.82 | 1.17 | 0 | 565.85 | 149.75 | 4.03 |
| AWR [39] | 41.7 | 0.65 | 4.17 | 0.95 | 40k | 599.71 | 162.60 | 4.02 |
| RMA w/o Adapt | 52.1 | 0.75 | 4.72 | 1.15 | 0 | 524.18 | 106.25 | 4.55 |
| RMA | 73.5 | 0.85 | 5.22 | 1.34 | 0 | 500.00 | 92.85 | 4.27 |
| Expert | 76.2 | 0.86 | 5.23 | 1.35 | 0 | 485.07 | 85.56 | 3.90 |

Kumar walks through the baselines (≈17:46–20:51):

- **Robust** is a very simple change: train a policy blind to the changing environment, robust to everything
  it might encounter, which is domain randomization "without necessarily caring about if we are giving this
  input to the policy or not" (≈18:31). Compared with RMA (the row above Expert; RMA "Stands for Rapid Motor
  Adaptation"), it is a bit conservative but has a weaker success rate (62.4 against 73.5), exerts more torque and
  experiences more jerk, "because it has no idea of the environment around it. It is just trying to be as
  aggressive as possible to make sure that it does not fall" (≈18:31).
- **System identification** estimates friction, mass and the other parameters explicitly and uses them
  downstream: instead of the compressed extrinsics, it predicts the parameters themselves (≈18:31–19:18). That
  is a hard problem (you may not detect the full scope of a slip, or know exactly how much the mass is), and
  it is not necessary for the downstream task of walking robustly; explicit estimation opens a gap in
  performance, and interestingly it is lower than the robust policy that is agnostic to the parameters
  (56.5 against 62.4, ≈19:18–20:05).
- **Fine-tuning at test time**: take a policy from simulation and, to bridge the sim-to-real gap, roll it out
  in the real world and improve it (the table's AWR row, with 40k samples). "The RMA style is closer to in
  context learning where you take the history of everything and then the learning happens in the feed
  forward pass itself", which lets it adapt in fractions of a second rather than by rolling out policies
  (≈20:05). See [meta-RL](meta-rl.md) and [policies that learn in context](policies.md#policies-that-learn-in-context).
- **Without adaptation** the performance falls, measured quantitatively, qualitatively and with real-world
  numbers, to below RMA's. There is a small gap between the **Expert** and RMA, "distilling it into an RMA
  method which can actually be deployed" (≈20:51).

## Questions on RMA

Seven student questions come between slide 41 and slide 42 (≈22:24–28:32); the DAgger explanation is given
[above](#training-the-adaptation-module) and the reward balancing [with the reward](#the-reward-function).

**Has there been follow-up work using real-world rollouts to improve the policy?** (≈22:24–23:58) The
question has two parts. One is the sheer power of a general learning algorithm: RL in the limit can learn any
behaviour, but a method that "can continuously improve perpetually online" is still an open problem. The
other is whether a policy that makes a mistake and figures out a fix in a few tries will remember it. That is
simpler, and might be answered by increasing the context given to the policy: compress a day or two of
context into it and in-context learning could let it remember how it fixed the problem last time. The first
is more powerful but very slow to learn; the second learns much faster in a simpler way. Kumar can see a
system with infinite context, like humans, maintaining a memory that lets it keep improving in the forward
pass without taking reinforcement learning gradients, but then the burden shifts to how to train such a model,
which needs something done in simulation.

**Can Phase 2 be trained directly?** (≈23:58–25:29) Yes, with more compute: Kumar used a CPU-based
simulator, and with GPU-based simulation and more GPUs it is possible. But most successful sim-to-real
policies don't use a history; they deploy on the current state alone. Kumar's guess at why: with only the
current state the policy is "state indexed" rather than state-and-time indexed. With a history, the
distribution of histories must be randomized enough that nothing exploited during training stops the
policy transferring. "The more information you give it in sim", as with a very long and dense history, the
more it may exploit the environment, the simulator's physics or the reward structure. For example, given a
periodic push, a robot with a long enough context or a notion of time commonly learns to be robust only when
it expects the push; randomizing the push timing gets around it.

**How large are the extrinsics?** (≈25:29–26:14) The start of the answer is garbled in the captions, but
Kumar says there is maybe a 2x compression: the extrinsics are around eight numbers, and the full physical
parameters maybe 17 or 18. In the vision version the compression is much more aggressive, from the full image
to a much smaller encoding.

**What if the previous action is not an input?** (≈26:59–27:45) The previous action carries an important
signal, the command given at the last time step. It lets the policy be smooth and consistent, and it allows
some correction behaviour: if that action did not work, the next one needs to be higher. "So the state is
lossy." Without the previous action, Kumar guesses, performance would drop.

**What about two, three or more time steps of state history?** (≈27:45) That is related to training the
whole system end to end in Phase 2 with RL: it can be done, but with care over sim-to-real transfer, since
more context gives more scope to exploit the simulation, the environment or the reward structure in
undesirable ways.

## Why vision?

Everything so far was blind, but a general robot model needs perception (≈20:51). Is walking even a good
testbed for vision, given the blind results (slide 39, ≈21:37)? Two arguments say yes. In some scenarios
vision is critical: crossing gaps or stepping stones, or making a jump, needs vision to estimate how to make
the jump. The second is efficiency, elegance and longevity: "So if you have to collide with a staircase
every single time that you climb it, it's a very inelegant policy", and the hardware will probably not
survive for long (≈21:37). Slide 40's banner puts it as "Perception enables precise and agile locomotion".

![Slide 40 — Perception enables precise and agile locomotion](../raw/images/17-sim-to-real/slide-40.jpg)

*Slide 40 — stepping stones across water, and a quadruped on concrete steps and on stone ledges: terrain where footholds have to be seen to be placed. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The work was published at CoRL 2022 (≈22:24): slide 41, *Legged Locomotion in Challenging Terrains using
Egocentric Vision*, by Ananye Agarwal* (CMU), Ashish Kumar* (UC Berkeley), Jitendra Malik† (UC Berkeley) and
Deepak Pathak† (CMU).

## Terrain maps, and why not to build them

The most common approach decouples vision from control entirely: build an explicit map, then feed it to the
controller (slide 42, ≈28:32–29:17). Kumar sees a fundamental flaw: building an explicit map is itself a very
hard problem, and "So your map will always be noisy". Non-rigid objects, or drift in the estimate of your
pose, lead to very noisy maps (slide 42's examples, "Non rigid obstacles" and "Pose estimation drift", from
Miki et al., *Science Robotics* 2022, and Kim et al., ICRA 2020: "But terrain maps are very noisy!").
Information is lost at that stage and cannot be recovered, so by solving a harder problem you lose
information you would have kept by putting the perception input directly into the policy (≈29:17).

![Slide 42 — Typical approach: build terrain maps from vision](../raw/images/17-sim-to-real/slide-42.jpg)

*Slide 42 — terrain maps built from vision, and two ways they go wrong: plants (non-rigid obstacles) and drift in the robot's pose estimate. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

Slide 43 shows how people add noise to the true terrain to train a map-based policy ("Noisy maps wipe out
signal => degraded performance"): perhaps the edges alone were enough to tell you how to take the step, but
once the terrain is a metric map the policy must be trained to be robust to all of these kinds of noise. That
is fine for simple scenarios and very hard once you must estimate precisely where a gap is and avoid stepping
in it (≈30:05).

![Slide 43 — Noisy maps wipe out signal => degraded performance](../raw/images/17-sim-to-real/slide-43.jpg)

*Slide 43 — sampled terrain heights under zero, nominal, large-offset and large noise, beside the stools and stone steps where precise footholds matter. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

So the work asks "Do we really need terrain maps?", hypothesizes that it does not, and instead "We tightly
couple vision and control" in the final system (slide 44, ≈30:05).

## The vision policy in deployment

The deployed system comes first, before how it is trained (slide 45, ≈30:52–32:26). The depth image the
robot sees is at the top right; cameras and compute are taped on, and everything runs on board in real time.
In Kumar's living room it crosses stepping stones made of 20 stools, over repeated runs with different gaps,
so that it must be using perception. On a hiking path with uneven stairs, a step is almost as high as the
robot, which struggles but climbs it; it handles even and uneven stairs, up and down; it keeps trying on
extremely slippery ground with barely a stable foothold and makes it; it takes a few tries at a very high
step. On debris around CMU, "it doesn't always get everything in one shot": the retries "make more sense than
the previous try", an example of learning within the context fed into the model (≈32:26). (Slide 46 is blank
in the PDF; the transcript does not say what was on screen there.)

![Slide 45 — the vision policy crossing stepping stones](../raw/images/17-sim-to-real/slide-45.jpg)

*Slide 45 — a frame of the deployment video: the robot crossing stools about 15 cm apart, with the egocentric depth image it sees at the top right. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Phase 1: walking with privileged terrain information

Training again has two phases. The first is as before, with added terrain-height information: "perfect
terrain height", with no noise, encoded and given to an MLP, and the whole system trained with RL (≈32:26).
"It is much faster to train because you don't need to render any perception inputs" (≈32:26). The recipe is
the same: PPO in simulation, this time in Isaac Gym, with rewards to track a target velocity and minimize
energy (slide 47: "PPO in IsaacGym", "Reward = (Track Velocity) + (Minimize Energy)", ≈33:14).

In slide 47's diagram the Phase 1 policy $\pi_1$, an MLP, takes three inputs. One is $o_t$ (the slide does
not name it). Another is $\gamma_t$, produced by an RNN from the **scandots** $m_t$, terrain heights sampled
in a grid around the robot (the yellow dots in the simulation picture). The third is $z_t$, produced by an
RNN from $e_t$. Its output is the action $a_t$.

![Slide 47 — Phase 1: Learning to Walk with Privileged Terrain Information](../raw/images/17-sim-to-real/slide-47.jpg)

*Slide 47 — Phase 1: the policy reads perfect terrain heights (scandots m_t, the yellow dots) through one RNN and environment parameters through another, trained with PPO in Isaac Gym. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

Training uses different terrains (slide 48: stairs, slopes, stepping stones, rough flat ground, gaps and
discrete obstacles), and the result is a single policy that walks on all of them in simulation (slide 49,
≈33:14).

![Slide 48 — Training Terrains](../raw/images/17-sim-to-real/slide-48.jpg)

*Slide 48 — the six simulated training terrains: stairs, slopes, stepping stones, rough flat, gaps and discrete obstacles. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

![Slide 49 — Phase 1 Policy](../raw/images/17-sim-to-real/slide-49.jpg)

*Slide 49 — the Phase 1 policy walking across the simulated terrains. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Phase 2: walking with egocentric depth

The problem is the same as before: how do we get these dots in the real world (≈33:14)? Slide 50 boxes the
scandots and "[friction, payload.....]" in red: "Cannot directly measure in real world". Most approaches
construct a map and feed it in; instead this work integrates depth directly (≈33:14).

![Slide 50 — How do we deploy it?](../raw/images/17-sim-to-real/slide-50.jpg)

*Slide 50 — the Phase 1 policy's privileged inputs, the scandots and the environment parameters, cannot be measured in the real world. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

Depth is rendered in simulation, and a depth encoder is supervised by the one trained in the first phase,
"using the same approach as DAgger where we roll out the student policy and then we supervise it with the
teacher", and likewise for the extrinsic vector (≈33:14–33:59). On slide 51 the Phase 1 policy's MLP is
copied and frozen ("Copy and Freeze") into the Phase 2 policy, whose new RNNs, one reading the depth image
$d_t$ and one reading $o_t$, produce estimates $\hat{\gamma}_ t$ and $\hat{z}_ t$, each regressed onto its
Phase 1 counterpart ("Regress"). Training Phase 2 directly would be an order of magnitude slower, "because
rendering depth takes a lot of time", so the two phases give a significant speed-up and still deploy very
robustly (≈33:59).

![Slide 51 — Phase 2: Learning to Walk with Egocentric Depth](../raw/images/17-sim-to-real/slide-51.jpg)

*Slide 51 — Phase 2: the frozen Phase 1 MLP is copied, and new RNNs over the depth image and o_t learn to regress the privileged latents (gamma and z with hats onto gamma and z). [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The final deployment policy (slide 52) takes the egocentric depth from the robot's camera and the full
proprioceptive history (≈33:59). One change is the use of **RNNs**, recurrent neural networks, which give a
somewhat longer context by integrating information over time, though they are not known to preserve very
long-horizon information, which is why the community moved to transformers for very long sequence modelling.
The RNN gave good enough context; the authors did not ablate a transformer to see whether it would improve
performance (≈34:45).

![Slide 52 — Deployment Policy](../raw/images/17-sim-to-real/slide-52.jpg)

*Slide 52 — the deployed policy: egocentric depth and o_t through two RNNs into the frozen MLP. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## What the policy learned: stairs for a small robot

"Stairs are designed for humans" (slide 53), and this robot is extremely short, so relative to its height the
stairs are tall: the slide marks steps of 17cm and 26cm (≈34:45). There is not enough clearance under the
robot to just raise its leg and take the step.

![Slide 53 — Stairs are designed for humans](../raw/images/17-sim-to-real/slide-53.jpg)

*Slide 53 — a 17 cm stair and a 26 cm step beside the small robot. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

Going down is also harder: a Spot robot tilts much less when it goes down a step, because it is a much taller
robot, while a small robot must use most of its leg and the discrepancy is much larger (≈35:32). Slide 54
compares the robots' sizes (ANYmalC, Spot and the A1, against a 26cm line) and draws the challenges as
"Obstructed" and "Topple".

![Slide 54 — Stairs are challenging for small robots](../raw/images/17-sim-to-real/slide-54.jpg)

*Slide 54 — ANYmalC, Spot and the much smaller A1 against a 26 cm line, and the two ways a step defeats a small robot: its leg is obstructed, or it topples. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

What the policy learns is to "move the leg out sideways to climb all of these stairs" (slide 55, "No
predefined gait => emergent hip abduction", ≈35:32). That is possible because no predefined gait or foot
motions are used: the behaviour is learned with reinforcement learning through search. Footstep planning is
emergent too: "So it only takes a step when it needs to", rather than moving its feet constantly as a
predefined gait would, and it is smooth and seems to work "with intention and precision" (slide 56,
≈35:32–36:18).

![Slide 56 — Emergent footstep planning](../raw/images/17-sim-to-real/slide-56.jpg)

*Slide 56 — a frame of the stepping-stones video on a tabletop, stones about 20 cm apart: the robot steps only when it needs to. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

The same policy handles all of these variations: "We don't use any explicit maps, no explicit gaits and that
is critical to get the results we do in this scenario" (slide 57, "Map Free, Gait Free", ≈36:18).

![Slide 57 — Map Free, Gait Free](../raw/images/17-sim-to-real/slide-57.jpg)

*Slide 57 — twelve real-world scenes for the one vision policy: walkways, rocks, stone and concrete stairs, log steps, debris and a dirt slope at night. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

Live demos at CoRL 2022 in New Zealand and at CVPR 2022 in New Orleans (slide 58) were quite robust; in one,
"I think at the bottom one", the right foot's padding came off and the robot started limping, but it
completed the full demo (≈36:18–37:04).

![Slide 58 — live demos at CoRL 2022 and CVPR 2022](../raw/images/17-sim-to-real/slide-58.jpg)

*Slide 58 — the live demos: stairs at CoRL 2022 in New Zealand, and a crowd around a bench and platforms at CVPR 2022 in New Orleans. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Comparison with baselines

Slide 60 ("Performance is better without maps") compares three policies by average distance (≈37:04–37:50):
a **blind** policy, as before; a **map-based** policy, fed a slightly noisy version of the map and trained with
that noise; and **ours**, going directly from depth.

| Terrain | Blind | Map Based | Ours |
| --- | --- | --- | --- |
| Slopes | 34.72 | 36.14 | 43.98 |
| Stepping Stones | 1.02 | 1.09 | 18.83 |
| Stairs | 16.64 | 6.74 | 31.24 |
| Discrete Obstacles | 32.41 | 29.08 | 40.13 |

The gap is small on simple terrain like slopes. On more challenging terrain "a map-based method is almost as
bad as blind": sampling the noise from the same distribution other works used, simulated performance drops
significantly (≈37:04). The slide boxes the stepping-stones row, 1.02 and 1.09 against 18.83, and notes
"Performance gap is larger on challenging terrains". The final action head needs whatever it needs from
perception directly, with no added noise or bottleneck (≈37:50). On stairs and discrete obstacles depth also
wins, and "One interesting thing is for stairs it's better to be blind and just learn to put your steps
blindly than to use a map which is noisy" (16.64 against 6.74, ≈37:50).

## Application to dexterous manipulation

The same idea applies to dexterous manipulation: rotating objects in hand (slide 61, ≈37:50–38:36). One
policy handles many objects, including a paper towel and a badminton shuttle (slide 62, a video; Qi et al.,
*In-Hand Object Rotation via Rapid Motor Adaptation*, CoRL 2022).

![Slide 62 — in-hand object rotation](../raw/images/17-sim-to-real/slide-62.jpg)

*Slide 62 — a frame of the in-hand rotation video: a multi-fingered robot hand turning a 48 g reaction ball. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

It generalizes across objects with very different physical properties (slide 63, ≈38:36–39:22): weight, from
5 g to almost 200 g (the slide's shuttlecock and 198 g avocado); the coefficient of friction (an ice ball, and
a kiwi with a slightly rough surface); the centre of mass, above the finger for one object and below it for
another, where the weight hangs; and shape, a cube or a cup. It was also trained in simulation and deployed
in the real world zero-shot, and the key idea is the same: the model is given its proprioceptive history and
uses it to work out how to act (≈39:22).

![Slide 63 — Generalization across Objects with Different Physical Properties](../raw/images/17-sim-to-real/slide-63.jpg)

*Slide 63 — pairs of objects that differ in weight (5 g against 198 g), friction, centre of mass and shape, all rotated by one policy. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

What is the model doing with the history? One hypothesis is **contact detection** (slides 64–66,
≈39:22–40:07). Slide 65 plots, for the bottom joint of one finger, the commanded actions against the joint
position actually achieved.

![Slide 65 — Proprioceptive history for Contact Detection](../raw/images/17-sim-to-real/slide-65.jpg)

*Slide 65 — for the bottom joint of one finger, the commanded action (blue) and the joint position achieved (red), over eight cycles. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

When the finger is not in contact, the two overlap at the bottom of the graph; when it makes contact, at the
top, it cannot reach the joint position it commands (slide 66: "Joint Position ≈ Action", "No Contact on
Pinky Finger"; "Pinky Finger in Contact", "Joint Position != Action"). That is enough signal for the network
to estimate contact, and from contacts you can work out the object's shape and size, which helps in adapting
to all these variations (≈40:07).

![Slide 66 — Contact Detection](../raw/images/17-sim-to-real/slide-66.jpg)

*Slide 66 — where action and joint position agree (pink ovals) the pinky is free; where the joint falls short of the action (green ovals) the pinky is touching the ball. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

There were live demos of this too (slide 67, ≈40:07–40:53): in the centre an orange peel, which deforms
completely under slightly more force than necessary, and on the right an adapter. "This has not been seen in
training data and simulation"; objects are not simulated at that level of granularity.

![Slide 67 — Real-world Demos](../raw/images/17-sim-to-real/slide-67.jpg)

*Slide 67 — live demos of the in-hand policy, at the University of Bristol, CoRL 2022 and ICRA 2024. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Application to drone flight

The same thing applies to drone flight (slide 68, ≈40:53). The same policy, the same weights, flies the drone
at the top and the one at the bottom of slide 69, estimating the mass and arm length. Kumar says the top drone
is four times heavier and its arm length three times the bottom one's; the slide prints "Large quad: 792g,
16.6cm Armlength" and "Small quad: 177g, 5.8cm Armlength". The weights work out which morphology they are
operating in and adapt, directly controlling each propeller's motor velocity: "So, it's not abstracting away
to a level where this becomes an easy problem" (≈40:53).

![Slide 69 — one policy flying a large and a small quadrotor](../raw/images/17-sim-to-real/slide-69.jpg)

*Slide 69 — the same weights flying a 792 g quadrotor (top) and a 177 g one (bottom). [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

It is also robust (slide 70): another drone with a higher mass wobbles at first, still estimating what drone
it is flying, and stays robust when a tennis ball is shot at one of its propellers (≈40:53–41:40).

![Slide 70 — a heavier drone on its launch platform](../raw/images/17-sim-to-real/slide-70.jpg)

*Slide 70 — the heavier drone before take-off; in the video it wobbles while it works out which drone it is flying, then stays steady when a tennis ball hits a propeller. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## What is left to solve: humanoids

"So what's left to solve?" (slide 71). "Well, I think almost everything we still don't have a
general-purpose robot model that can control a general-purpose hardware to achieve the task that humans can"
(≈41:40). Kumar focuses on **humanoids** (slide 72), the general form factor known to be able to do
everything humans can, and shows the progress made at Tesla (≈41:40–42:26). Slide 73's videos include a biped
walking on a hill: Kumar showed similar results with a quadruped, but "a biped is much less forgiving because
it's a much more unstable system and you need to be more reactive". These too were trained in simulation and
deployed zero-shot, and a more recent video shows the robot dancing, all real-time, on board and autonomous.
"Getting the first thing to work and the 10th thing to work does not take very long because the pipeline is
very simple" (≈42:26).

![Slide 73 — humanoid robots walking and dancing](../raw/images/17-sim-to-real/slide-73.jpg)

*Slide 73 — frames of three humanoid videos: walking on a mulch slope, and standing in a hall and a warehouse, one marked 1x Speed. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

On manipulation (slide 74, "Optimus is learning many new tasks", ≈42:26–43:56), one neural network takes
language as input and executes the command. The key ingredient is video: it was learned from egocentric
videos of humans doing things and a little robot data, translated somehow into robot behaviour. The tasks get
decent performance, but they are "not very reliable".

![Slide 74 — Optimus is learning many new tasks](../raw/images/17-sim-to-real/slide-74.jpg)

*Slide 74 — a frame of the manipulation video: a humanoid in a lab, the one language-conditioned network behind every task. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

## Why manipulation lags behind locomotion

The eventual goal is reliability and dexterity at the level humans have: consistency, dexterity, agility and
reliability (slide 75, "Getting to reliability and dexterity", ≈43:56). The locomotion results were very
agile, dexterous and very reliable in the real world, three to four years before the lecture. Why are there
not already similar results for manipulation? Go back to the two ingredients that made locomotion work
(slide 75's bullets: "Simulation technology is still very primitive" and "Rewards are much harder to
specify"):

- **Simulation was ready for locomotion.** Rigid-body contact was good enough for very robust locomotion. For
  manipulation, chopping vegetables has to be simulated, and deformable objects such as dough, to get
  sim-to-real transfer to work (≈44:42).
- **Rewards are easy to specify for locomotion.** A programmatic reward to track a velocity is the complete
  task specification; randomize the terrain and a general policy comes out. Not so for manipulation: even
  with full simulated state, chopping a vegetable needs a reward that measures each slice, which depends on
  what you are cooking. That gets you one task, not the generality of manipulation; stretching the dough or
  putting flour on it needs another reward. "So it's a very manual process and even if you get one of this
  right, it will not scale to anything else" (≈44:42–45:29).

![Slide 75 — Getting to reliability and dexterity](../raw/images/17-sim-to-real/slide-75.jpg)

*Slide 75 — first-person kitchen clips from Epic Kitchens (stirring chicken, soaking a pan, stretching dough), under the two obstacles: primitive simulation and hard-to-specify rewards. [Deck](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf)*

"So everything I'll present from now on is speculative" (≈45:29).

## Simulation, and the bitter lesson

Some scenarios, such as deformable objects, can be simulated, but not very fast, and some more complex
interactions cannot be simulated at all (slide 76, "We can, in fact model complex scenarios like deformable
objects, etc. However, they are slow.", ≈45:29–46:14). Yet there is no obvious fundamental reason why more
complex scenarios cannot be simulated as compute grows. If we bet on compute, simulation can progress with
it and give progressively better real-world results as its scope expands. That is what the **bitter lesson**
suggests: Rich Sutton's very short piece, which Kumar recommends everyone read, opens by saying, in Kumar's
paraphrase, that "general methods that leverage compute are ultimately the most effective and by a large
margin" (≈46:14–47:01). The slide quotes it: "The biggest lesson that can be read from 70 years of AI research
is that general methods that leverage computation are ultimately the most effective, and by a large margin".

## A general reward model?

The other ingredient is a very general reward model (slide 77, ≈47:01–47:47). The most obvious answer is a
large model trained for general understanding of the world, used to define the rewards that improve the
policies. "This could work but RL as we know has exploited weaknesses in the model if they exist." Continued
tuning with [RLHF](rlhf.md) takes the model completely out of the distribution you want, and the DeepSeek
paper finds that without verifiable or rule-based rewards, RL at some point starts exploiting the rewards
defined for it (≈47:01–47:47).

Simulation is a hard bet too, so why be optimistic here? Because "the absolute worst case is we get humans to
label all the rewards for us and humans are very scalable because we have billions of them" (≈47:47; slide 77:
"Worst case: humans label everything for us!"). See [reward learning](reward-learning.md) and
[learning rewards from preferences](preference-learning.md).

## Beyond parity with humans

Most of the talk, and most work in robot learning, treats robotics as achieving parity with humans (slide
78, ≈47:47–48:35). "The day we achieve that, it will be a civilization changing event", but only a first
step: actuators driven by electrical torque and computers based on silicon rather than carbon might be
exploited for capabilities humans have not thought of or cannot work out, doing things faster, more
efficiently or better, and being more agile and dexterous than humans (≈48:35). One system has already done
it: AlphaGo, where the key was "a sparse reward and then lots and lots of search or reinforcement learning" to
discover behaviours beyond what humans could think of (slide 78: "Learnings from AlphaGo:" followed by "Sparse
reward + lots of search!", ≈48:35–49:22). The closing question: can we get robots "to not just match human capability
but exceed what humans can do?" (≈49:22). The recording ends there; slide 79's "Questions?" is not shown
being reached.

## Summary

Successful imitation learning uses off-policy data and gives generalists; successful RL is on-policy, uses
good and bad data, and gives specialists. RL's successes rest on well-specified rewards and scalable policy
evaluation, and for robots simulation supplies both, so the problem is getting from simulation to the real
world. RMA trains a base policy in simulation with PPO and about a billion samples, giving it the
randomized physics parameters through a compressed extrinsics vector and a ten-term reward that encodes
energetics, stability and hardware care. Because the real world hides those parameters, an adaptation module
trained with DAgger estimates the extrinsics online from the last 50 states and actions, adapting in a
fraction of a second in the forward pass, closer to in-context learning than to fine-tuning. One set of
weights then walks blind over rocks, sand, oil, planks and heavy payloads, beats domain randomization, system
identification and real-world fine-tuning, and comes close to the expert that saw the true parameters. Adding
vision without a terrain map, a teacher trained with perfect terrain heights is distilled into a student that
reads egocentric depth; the small robot learns to climb human stairs with emergent hip abduction and footstep
planning, and beats both blind and map-based policies, which barely move on stepping stones. Proprioceptive history and
sim-to-real transfer also give in-hand rotation, apparently through contact detection, and a single drone
controller across morphologies. What remains is almost everything: humanoids, and reliable, dexterous
manipulation, held back by simulation and by rewards that are hard to specify. Kumar speculates that compute
will improve simulation, that a general reward model could work with humans labelling rewards in the worst
case, and that robots, like AlphaGo, might exceed human capability.

## Related pages

- [Sim-to-real transfer](sim-to-real.md): the topic page: simulation as the source of rewards and evaluation,
  privileged teacher and adapted student, the alternatives, and the limits for manipulation.
- [Imitation learning](imitation-learning.md): DAgger, which trains the adaptation module and the depth
  student, and the off-policy data this lecture contrasts with RL.
- [PPO](ppo.md): the RL algorithm both locomotion policies are trained with in simulation.
- [Reward learning](reward-learning.md) and [RLHF](rlhf.md): hand-designed locomotion rewards, the hope for a
  general reward model, and how RL exploits a learned one.
- [Policies](policies.md) and [meta-RL](meta-rl.md): policies that adapt in context from their history, and
  what a long history lets a policy exploit in simulation.
- [Offline and online data](offline-and-online.md): on- and off-policy data in the course's own terms.
- [LLM reasoning](llm-reasoning.md): DeepSeek-R1's rule-based rewards, one of RL's successes here.
- [Lecture 16 — RL for Robots: Autonomous Learning](16-autonomous-rl.md): the first robotics lecture, whose
  slide 44 contrasts RMA's low-level adaptation with adapting in a space of skills.
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md) and
  [Lecture 1 — Class Intro](01-class-intro.md): earlier examples of legged robots trained in simulation and
  transferred to the real world.
- [Course logistics](course-logistics.md): the guest lecturers.
