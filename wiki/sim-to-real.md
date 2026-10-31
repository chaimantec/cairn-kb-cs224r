# Sim-to-real transfer

Training a robot's policy in a physics simulator and deploying it on the real robot. Simulation supplies
what reinforcement learning needs most and a real robot supplies worst: rewards that can be computed from
the full state, and evaluation of policies at a scale that turns compute into data. What it does not supply
is the real world, whose physical parameters differ from the simulator's and are never observed. The course
shows sim-to-real policies in [Lecture 1](01-class-intro.md) and [Lecture 5](05-off-policy-actor-critic.md),
and [Lecture 17](17-sim-to-real.md), a guest lecture by Ashish Kumar, teaches how to make one work: train with
privileged information in simulation, then distil it into a policy that infers what it cannot see from its
own recent history. Kumar's pronouns are not stated in the material, so this page writes "Kumar".

## Why simulation

Kumar traces RL's largest successes (AlphaGo, reasoning in language models) to two ingredients:
well-specified rewards that can be computed programmatically, and scalable evaluation of policies (Lecture 17,
slide 3, ≈3:54–5:25). A game whose rules are known can be put in simulation, where "it scales beautifully
with compute" (≈4:40). For robots, simulation is the closest approximation to both ingredients: the full
state of the world is available, so rewards can be computed, and "we have converted a physical problem to a
digital problem which now just will scale with compute" (slide 4, ≈6:11). But "this gets us to solving
robotics in simulation but that is not the same as solving robotics in the real world" (≈6:11); the
remaining problem is sim-to-real.

Simulation also changes which algorithm is attractive. Where data is not a problem, the stable but less
data-efficient [PPO](ppo.md) is a common choice, and with a carefully tuned simulator a policy learned there
may transfer to the real robot, as with a quadruped that learns to walk in simulation (Lecture 5, slide 26,
≈1:08:40). Both of Lecture 17's locomotion policies are trained with PPO in simulation, one of them with
"about a billion samples" (≈10:00, ≈33:14). [Lecture 1](01-class-intro.md) shows legged robots that walk and
dance, trained with deep RL in simulation and transferred to real robots (Lecture 1, slide 19, ≈8:32).

## Randomize the simulator, and tell the policy

A policy trained under one set of physics would be very narrow. The standard remedy is **domain
randomization**: randomize the physical parameters in simulation so that one policy is trained across a huge
range of masses, frictions and terrains (Lecture 17, ≈9:14). A policy that is not told the parameters must be
robust to all of them at once; Kumar's comparison calls this the **robust** baseline, and it ends up
conservative yet less successful, using more torque and suffering more jerk, "because it has no idea of the
environment around it" (slide 38, ≈18:31).

Rapid Motor Adaptation (RMA) does the opposite in simulation: everything the simulator knows is "fair game"
(≈8:29). An environmental factor encoder compresses the parameters (mass, centre of mass, friction, terrain
height, motor strength) into a short vector $z_t$, the **extrinsics**, and the base policy takes $z_t$
together with the state $x(t)$ and the previous action $a(t-1)$ (slide 14). So "it's not oblivious or blind
to the environment" (≈9:14). The extrinsics are a compression: around eight numbers against maybe 17 or 18
physical parameters, Kumar says in answer to a question (≈25:29–26:14). The rewards also carry what the
simulator lacks: alongside tracking a velocity, terms minimize work, ground impact and anything that
damages hardware, "because in simulation there's no concept of hardware damage" (slide 16, ≈10:00–10:45).
See [reward learning](reward-learning.md#hand-designed-rewards-for-locomotion-and-a-general-reward-model).

## Infer the hidden parameters from history

The extrinsics are unknown in the real world, and measuring them once does not help: estimated in one room,
they fail the moment the robot reaches sand or mulch (≈10:45). RMA instead estimates them **online**, in
fractions of a second, from the recent history of states and actions (slides 17–18). The signal is the
discrepancy between what was commanded and what happened: if a leg slips more than expected, the difference
tells the robot "This was not intentional. You slipped forward" (≈11:32). An **adaptation module** reads the
last 50 states and actions and outputs an estimate of $z_t$ for the unchanged base policy.

It is trained in simulation, where both the true extrinsics and the history exist, with
[DAgger](imitation-learning.md#distilling-a-privileged-teacher): roll out the student (the adaptation module)
to generate the data, and supervise each of its predictions with the teacher's $z_t$ (slide 19,
≈12:18–13:03, ≈27:45–28:32). The result is a two-phase recipe that recurs through the lecture: a privileged
teacher trained with RL, and a student, trained to imitate it, that sees only what the real robot can sense.
At deployment the module ran at 10 Hz against the base policy's 100 Hz, which training had not accounted for,
without a drop in performance (slide 20, ≈13:03).

Kumar calls this "closer to in context learning", where "the learning happens in the feed forward pass
itself" (≈20:05): the adaptation takes no gradient step. See [policies](policies.md#adapting-from-a-history-of-states-and-actions)
and [meta-RL](meta-rl.md). Lecture 16 cites RMA as the example of **adaptation in low-level action space**,
suited to small out-of-distribution changes such as new terrain, against adapting in a space of skills for
drastic ones ([Lecture 16](16-autonomous-rl.md), slide 44; see [autonomous and single-life RL](autonomous-rl.md)).

## The alternatives compared

Lecture 17's slide 38 compares RMA with three existing ways of getting a general policy into the real world
(≈17:46–20:51): domain randomization (the robust policy), at 62.4% success; **system identification**, which
predicts friction, mass and the rest explicitly instead of a compressed vector, at 56.5%, lower than the
robust policy because exact estimation is hard and unnecessary for walking; and **fine-tuning at test time**,
rolling the simulated policy out in the real world and improving it, at 41.7% after 40k samples. RMA reaches
73.5%, against 52.1% with its adaptation switched off and 76.2% for the expert it is distilled from. The
lecture page has [the whole table](17-sim-to-real.md#quantitative-comparison).

## Perception without a map

Adding vision usually means building an explicit terrain map and feeding it to the controller. Kumar argues
this solves a harder problem than needed and loses information that cannot be recovered: "So your map will
always be noisy", from non-rigid obstacles or drift in the pose estimate, and a policy trained on noisy maps
struggles exactly where a foothold must be precise (Lecture 17, slides 42–43, ≈28:32–30:05). The alternative
couples vision and control directly, with the same teacher and student (slides 47–52, ≈32:26–34:45).
The teacher, trained with PPO, gets perfect terrain heights sampled around the robot, which is fast because
nothing has to be rendered. The student reads egocentric depth and is trained with DAgger to reproduce the
teacher's internal representations. Training on depth from the start would be "an order of magnitude slower
because rendering depth takes a lot of time" (≈33:59). On average distance the depth policy beats a blind one
and a map-based one on every terrain; on stepping stones the other two barely move (18.83 against 1.02 and
1.09), and on stairs a
blind policy does better than a noisy map (16.64 against 6.74; slide 60, ≈37:04–37:50).

## What a policy can exploit in simulation

A simulator is not the world, and RL finds the differences. Most successful sim-to-real policies use only the
current state, Kumar notes, and a longer history gives the policy more scope to exploit "either your
environment or physics in simulation or your reward structure" (≈24:44–25:29). A robot pushed periodically,
with enough context or a notion of time, learns to be robust only when it expects the push; randomizing that
too gets around it (≈25:29). The previous action, by contrast, is a useful input, because "the state is lossy"
and the last command lets the policy be smooth and correct itself (≈26:59–27:45). See
[policies](policies.md#adapting-from-a-history-of-states-and-actions).

## Beyond legs: hands and drones

The same recipe of proprioceptive history and zero-shot transfer extends past locomotion (Lecture 17,
slides 61–70, ≈37:50–41:40). A multi-fingered hand rotates objects from 5 g to almost 200 g, with different
friction, centre of mass and shape, apparently by detecting contact: when a finger touches the object, the
joint position it reaches stops matching the action it commanded (slides 63–66). One set of weights flies
drones of different mass and arm length by estimating their morphology, directly controlling each
propeller's motor velocity (slides 69–70). Earlier, Lecture 5 showed a robot hand solving a Rubik's cube with a
policy trained in simulation with PPO, while noting that sim-to-real transfer is "a lot harder" for
manipulation in general (Lecture 5, slide 27).

## Where it falls short

Kumar's locomotion results were agile, dexterous and reliable three to four years before the lecture; why is
manipulation not there yet (Lecture 17, slide 75, ≈43:56–45:29)? Both ingredients fail. Simulation was ready
for locomotion, since rigid-body contact was good enough, but manipulation needs vegetables that can be
chopped and dough that deforms. And a velocity-tracking reward is a complete specification of walking, while
each manipulation task needs its own reward (the size of each slice depends on what you are cooking), "So
it's a very manual process and even if you get one of this right, it will not scale to anything else"
(≈45:29). Kumar's speculation (slides 76–77): bet on compute, as Rich Sutton's bitter lesson advises, to make
simulation faster and broader; and a general reward model might work, though "RL as we know has exploited
weaknesses in the model if they exist", with humans labelling rewards as the worst case (≈45:29–47:47).

## Where this appears

- [Lecture 17 — Advancing Robot Intelligence](17-sim-to-real.md): the whole topic: RMA, its adaptation
  module and comparison, vision without maps, hands and drones, and what is left for humanoids and
  manipulation.
- [Lecture 1 — Class Intro](01-class-intro.md): legged robots trained in simulation and transferred to the
  real world, as a reason to study deep RL.
- [Lecture 5 — Off-Policy Actor Critic](05-off-policy-actor-critic.md): PPO for simulation, where data is
  cheap, with a quadruped and a Rubik's-cube hand transferred to real robots.
- [Lecture 16 — RL for Robots: Autonomous Learning](16-autonomous-rl.md#single-life-rl): RMA as adaptation in
  low-level action space, against adapting in a space of skills.
- [PPO](ppo.md): the algorithm the simulated policies are trained with.
- [Imitation learning](imitation-learning.md#distilling-a-privileged-teacher): DAgger as the way a student
  learns from a privileged teacher.
- [Policies](policies.md#adapting-from-a-history-of-states-and-actions) and [meta-RL](meta-rl.md): adapting
  from history, in context.
- [Model-based RL](model-based-rl.md): models of the dynamics learned from data, where this page's
  simulators are given.
