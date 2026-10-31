# Lecture 15 — Hierarchical RL and IL

Stanford CS224R, Spring 2025 (Wednesday, May 21) ·
[video](https://www.youtube.com/watch?v=iKWYLSVAtfM) (69 min) ·
[slides (PDF)](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf) ·
[slide-by-slide text](../raw/slides/15-hierarchical-rl.md) ·
[transcript](../raw/transcripts/15-hierarchical-rl.md)

**What this lecture establishes.** Lectures 12–14 trained one policy to do many tasks. This lecture
asks how to string behaviours for several subtasks together into one **long-horizon task**, such as
cooking a meal or driving to Yosemite. Such tasks are hard because the agent visits a very large number
of states, has many chances to make mistakes and many chances to get stuck. The main idea of
**hierarchy** is to split the policy in two: a **high-level policy** $\pi_{\text{HL}}$ that predicts
intermediate goals, or subgoals, and a **low-level policy** $\pi_{\text{LL}}$, a multi-task policy that
tries to accomplish the current subgoal by predicting actions, at a higher frequency. Hierarchy can
help through extra supervision, knowledge shared across similar subtasks, structured exploration in RL,
and cheaper computation. Finn then walks through three design choices: how to **represent** the goals
(expressive, structured, at the right level of abstraction, and domain specific), how to **supervise**
each level (separately at first, but at least one level must be adapted to the deficiencies of the
other, since training both end to end with a latent goal is just a flat policy), and **when to move on**
to the next goal (when the low level is done, or after a fixed number of steps, which most systems
choose). The second half looks at example systems: hierarchical imitation learning with language
subgoals, fine-tuned with DAgger in language space; hierarchical imitation with image subgoals produced
by an image-editing model; a chain-of-thought version; hierarchical RL with state and language goals
and hindsight relabeling; and the open problem of discovering skills without supervision.

Slide numbers below are the numbers printed on the slides, which equal the PDF page numbers.
Timestamps (≈M:SS) point into the video and the [transcript](../raw/transcripts/15-hierarchical-rl.md).
**The recording covers slides 3–44 and slide 46's "next time".** It opens with the recap of slide 3
(≈0:05) without discussing slide 2's reminders, and ends at about 1:09:28 on slide 46. Slide 45's
summary outline and the reminders on slides 2 and 46 are not discussed; the
[section on them](#not-in-the-recording-the-summary-and-reminders) gives them from the
[slide file](../raw/slides/15-hierarchical-rl.md).

Finn draws on a board or on the slides in places the recording does not show: the hierarchical policy
diagram (≈3:58–6:16), the roll-out procedure before slide 8 appears (≈10:09–11:41), the pasta-making
example's sets of states (≈25:31–28:36 and ≈30:57–37:04), which no slide holds, the variable-length
segments of a demonstration (≈47:01) and an image-subgoal policy in answer to a question (≈1:01:38).
Where the transcript says "here" or "this", this page reconstructs what was meant from the surrounding
words and the slides, and says so. Videos play on slides 31–33, 35, 36 and 39; the PDF holds only a
frame of each, and what they showed comes from the transcript.

## Contents

1. [From many tasks to long-horizon tasks](#from-many-tasks-to-long-horizon-tasks)
2. [Why long-horizon tasks are hard](#why-long-horizon-tasks-are-hard)
3. [The main idea: high-level and low-level policies](#the-main-idea-high-level-and-low-level-policies)
4. [Rolling out a hierarchical policy](#rolling-out-a-hierarchical-policy)
5. [Why may hierarchy help?](#why-may-hierarchy-help)
6. [Do we really need multiple policies?](#do-we-really-need-multiple-policies)
7. [Design choice 1: how to represent goals](#design-choice-1-how-to-represent-goals)
8. [Design choice 2: how to supervise each level](#design-choice-2-how-to-supervise-each-level)
9. [Design choice 3: when to move on to the next goal](#design-choice-3-when-to-move-on-to-the-next-goal)
10. [Hierarchy in current robot systems](#hierarchy-in-current-robot-systems)
11. [Hierarchical imitation with language subgoals](#hierarchical-imitation-with-language-subgoals)
12. [Fine-tuning the high level with DAgger](#fine-tuning-the-high-level-with-dagger)
13. [Results: hierarchy against a flat policy](#results-hierarchy-against-a-flat-policy)
14. [Hierarchical imitation with image subgoals](#hierarchical-imitation-with-image-subgoals)
15. [Questions on image subgoals and representations](#questions-on-image-subgoals-and-representations)
16. [A chain-of-thought version, and RL as an open problem](#a-chain-of-thought-version-and-rl-as-an-open-problem)
17. [Hierarchical reinforcement learning](#hierarchical-reinforcement-learning)
18. [Discovering skills without supervision](#discovering-skills-without-supervision)
19. [Not in the recording: the summary and reminders](#not-in-the-recording-the-summary-and-reminders)

## From many tasks to long-horizon tasks

The recent lectures trained RL and imitation learning systems that do many different tasks (slide 3,
≈0:05–0:51). [Multi-task imitation and RL](12-multi-task-rl.md) conditions a policy or a Q-function not
only on the state but also on a descriptor of the task, $\mathbf{z}_ i$, and in the goal-conditioned
setting that descriptor can be a state itself, a goal state $\mathbf{s}_ g$.
[Meta-RL](13-meta-rl.md) instead has the policy figure out the task from a few examples and a little
experience with rewards: a few-shot policy that adapts from a small amount of experience. Today's
question, in the slide's words: "can we string together behaviors from multiple (sub)tasks?" The aim is
not only a policy that does several tasks, but one that strings behaviours for several tasks together
into a longer-horizon task (≈0:51).

![Slide 3 — Brief recap](../raw/images/15-hierarchical-rl/slide-3.png)

*Slide 3 — the policies of the last three lectures: conditioned on a task descriptor or a goal state (multi-task and goal-conditioned) or on few-shot experience (meta-RL). Today asks how to chain such behaviours. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

The plan (slide 4, ≈0:51–1:39) is the problem (why long-horizon tasks are hard, why hierarchy may
help), the key design choices behind hierarchical imitation and RL methods (how to represent skills and
goals, when to move on from one goal to the next, and the supervision for each level of the hierarchy),
and then example systems, because there is a variety of ways to design them. Slide 4 lists the design
choices in that order; the lecture takes supervision before when to move on, the order slide 12 prints.

## Why long-horizon tasks are hard

A long-horizon task takes a while to complete and may consist of a number of subtasks or steps to be
taken in some sequence (slide 5, ≈1:39–2:26). Slide 5's examples are "Cook focaccia." for a robot (Finn
says cooking a meal), "Fix a bug that is causing neural network training loss to explode." for a
language-model agent, "Drive to Yosemite." for an autonomous vehicle and "Give feedback on an 8-page
report." for a language model. Some steps depend on each other, so one must come before another; others
are less interdependent and can be done in parallel (≈2:26).

These are really hard problems for AI systems, and slide 5 gives three reasons (≈2:26–4:44):

1. **A very large number of states is visited.** A task that takes a long time passes through many
   states, and the policy must make good decisions in every one of them, so it must succeed on a much
   broader distribution.
2. **Many opportunities to make mistakes**, each of which the agent must recover from, if it can be
   recovered from at all.
3. **Many opportunities to get "stuck".** A mistake that makes no progress can return the agent to the
   same state, where it makes the same mistake over and over again. Finn calls these mistakes
   particularly gnarly: back in the same state, the policy keeps deterministically staying there and
   never makes progress towards the goal.

## The main idea: high-level and low-level policies

Finn builds the idea on a board, starting from the multi-task policies of earlier lectures (≈3:58–6:16).
A policy takes the state at some time step and a task or goal, written $g$ because it can take many
forms, and produces actions that address that goal. If one policy can do many things, a bigger task can
be broken into subtasks, with **another policy** telling this one which task to do at each point in
time. That second policy outputs goals, and it also sees the state, so it knows where the agent is and
which goal makes sense next. The goal-following policy is the **low-level policy**, written
$\pi_{\text{LL}}$, which does shorter tasks; the goal-setting one is the **high-level policy**, which
outputs goals for the low level to accomplish. (Finn corrects a label on the board: "I wrote this as
low-level this should be high-level", ≈5:30.) To do several long-horizon tasks rather than one, the
high-level policy can also take a task descriptor or a high-level goal as input: not only driving to
Yosemite but also to the Grand Canyon, LA or Seattle (≈6:16).

The crux is that a long task broken into smaller parts may be easier to address (slide 6, ≈6:16–7:49).
Training a robot policy to output motor commands for "Bake a cheesecake" is really hard ("doing this is
really hard", slide 6). But the task can be broken down: the first step is to buy ingredients, whose
first step is to go to the store, which starts by walking to the door, which starts with taking a step,
which means contracting a muscle (the slide prints "Contract quad muscle"). Smaller tasks like "Take a
step" are much easier to train systems to do ("doing this is easy"), and once they can be done they can
be composed with a higher-level policy to accomplish the harder task.

Slide 7 draws the result. The task ("Bake a cheesecake") and the observation $\mathbf{o}_ t$ go into
the high-level policy $\pi_{\text{HL}}$, which "predicts intermediate goals $\mathbf{g}_ t$", such as go
to the store or go to the door; the goal and the observation go into the low-level policy
$\pi_{\text{LL}}$, which "tries to accomplish $\mathbf{g}_ t$ by predicting actions $\mathbf{a}_ t$"
(≈7:49–8:37).

![Slide 7 — The main idea](../raw/images/15-hierarchical-rl/slide-7.png)

*Slide 7 — the two-level policy: the high-level policy (blue) turns the task and observation into a goal, and the low-level policy (green) turns the goal and observation into actions, at a higher frequency. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

The two levels run at different rates (≈8:37–9:23). The low-level policy must act at the original
frequency: perhaps 20 hertz when controlling a robot, word by word for a language model. The high-level
policy can predict goals on a coarser time scale, for instance after a paragraph is complete, or after
the agent has finished going to the store. As slide 7 puts it, the low level "operates at a *higher
frequency* than the high-level policy."

The goals go by many names, as slide 7 lists: "subgoals, subtasks, skills, options, high-level
actions" (≈9:23). Finn explains the last: they are actions, but for a different policy that operates at
a higher level of abstraction than the original one. And the hierarchy "Could have more than 2 levels"
(slide 7): for the cheesecake, one level could handle going to the store, buying things and bringing
them back, and another could control the robot's motors to accomplish those (≈9:23–10:09).

## Rolling out a hierarchical policy

How is such a policy run (≈10:09–12:27)? Finn writes the procedure on the board, then shows it on
slide 8. Predicting a goal is a kind of planning, thinking forward to the next step, so step 1 samples
a goal from the high-level policy given the current observation and, optionally, a high-level goal
(the slide's "prompt"). Then the low-level policy is sampled repeatedly with that goal: sample an
action, execute it, get the next state, and with the same goal and the new state sample a new action.
When the agent is done with the goal, or enough time has passed, it returns to step 1 to plan a new goal
from the latest state:

0. Observe initial observation $\mathbf{o}_ 1$ at $t = 1$.
1. Plan goal $\mathbf{g}_ t$ according to $\pi_{\text{HL}}(\cdot \mid \mathbf{o}_ t, \text{prompt})$.
2. Until a new goal is selected: (a) execute predicted action $\mathbf{a}_ t$ according to
   $\pi_{\text{LL}}(\cdot \mid \mathbf{o}_ t, \mathbf{g}_ t)$; (b) set $t \leftarrow t + 1$ and observe
   the new observation $\mathbf{o}_ t$.

Here $\mathbf{o}_ t$ is the observation at time $t$, $\mathbf{g}_ t$ the current goal and
$\mathbf{a}_ t$ the action. When to select a new goal is a design choice that comes later in the
lecture ([design choice 3](#design-choice-3-when-to-move-on-to-the-next-goal)).

## Why may hierarchy help?

Slide 9 lists four reasons (≈12:27–14:46).

1. **"Supervision signal on how to complete the task".** The low-level policy is typically trained to
   reach goals and the high-level policy to maximize rewards or reach high-level goals. The low level's
   objective therefore differs from the overall objective of completing the long-horizon task, and this
   intermediate supervision on how to complete intermediate goals can guide the policy towards
   completing the whole task. Depending on the setup, hierarchy is a way to leverage intermediate
   supervision.
2. **"More direct knowledge sharing across similar subtasks".** A big flat policy could also get
   intermediate supervision, through shaped rewards for completing subtasks. The difference is that the
   low-level multi-task policy, with the goal as input, targets a particular subtask, so when two
   subtasks are similar it can share data across them and learn more efficiently than one policy
   trained across the whole long-horizon task.
3. **"In RL: structured exploration in higher-level space".** When the high-level policy proposes goals
   to reach, it structures exploration, encouraging the agent to explore not only in the original action
   space but also in the higher-level goal space (see [exploration](exploration.md)).
4. **"Practical challenge: Address speed requirements by running polices at different frequencies"**
   (the slide prints "polices"). Making a cheesecake might require reasoning through the steps; a single
   flat policy must do that reasoning at every time step, 20 times a second for a robot at 20 hertz,
   whereas a hierarchy can run the reasoning at a lower frequency than the low-level policy, a practical
   benefit in compute.

Two questions follow (≈14:46–16:20). How does the supervision of the two levels differ? The high-level
policy's goal is to complete the long-horizon task: it gets rewards when the cheesecake is baked. The
low-level policy gets rewards for completing intermediate subtasks, such as finding the ingredients or
preheating the oven, and that intermediate supervision guides it towards completing the long task. What
if several tasks must be done at once? Unless the low-level agent has something like 10 arms or can
output things in parallel, it must pick some order. Working out whether the order matters is the
high-level policy's job: if it matters, the high level gets rewards for predicting the correct order;
if it does not, the high level should learn that and pick one arbitrarily.

A later question asks why intermediate supervision helps at all (≈17:53–19:24). In RL, if the only
reward comes when the cheesecake is baked, rewards are really sparse and do not let the agent make
progress. In imitation learning, a policy that is not explicitly thinking about which part of the task
it is doing can oscillate between subtasks, especially if it has no history of past observations; so
there are benefits in imitation settings too.

## Do we really need multiple policies?

Slides 10 and 11 set three designs side by side (≈16:20–17:53). **Hierarchy** uses two policies. A
**flat policy**, with no hierarchy, takes the long-horizon task and the state and predicts the action.
And there is an intermediate approach that keeps many of hierarchy's benefits and much of its design,
similar to **chain of thought** in language models: a single policy first reasons through which goal or
subtask it should be completing and then predicts the low-level action, outputting $\mathbf{g}_ t$ and
then $\mathbf{a}_ t$. This combines the two policies into one that predicts them in sequence.

![Slide 11 — Do we really need multiple policies?](../raw/images/15-hierarchical-rl/slide-11.png)

*Slide 11 — hierarchy (two policies), a flat policy, and chain of thought (one policy that predicts the goal and then the action), with the slide's verdict: same supervision, possibly too expensive at 50 Hz, no conclusive comparison yet. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

The chain-of-thought approach "Can benefit from same supervision!" (slide 11). Its practical downside is
that it loses the computational benefit of separating the levels, since the reasoning still runs at the
full frequency: it "May be too computationally expensive (e.g. 50 Hz control)". Empirically there is
"No conclusive empirical comparison (yet)." with a hierarchical approach (≈17:53). There might be
benefits to sharing knowledge through separate low-level policies rather than a combined model, so
whether one can just do this is an open area of research; the only conclusive downside is
computational, the upside of reasoning at a lower frequency than acting.

## Design choice 1: how to represent goals

The design choices start with how to represent the goals, posed as a think-pair-share (slide 13,
≈19:24–20:56): for one task, discuss the intermediate goals and the best representation for them. The
four tasks are a robot that learns to cook Italian dishes such as focaccia, controlling its motors; an
autonomous bike that navigates to locations on campus (slide 13's example is "Bike to the dish
entrance."); a language model that drafts legal briefs, itself a complex and involved process; and a web
agent that plans and books one-week vacations for users.

![Slide 13 — What goal representation would you use?](../raw/images/15-hierarchical-rl/slide-13.jpg)

*Slide 13 — the think-pair-share: pick one of four long-horizon tasks (cooking, biking across campus, drafting legal briefs, booking vacations) and choose its intermediate goals and their representation. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

The students' answers (≈20:56–23:13). For the bike, the intermediate goals could be waypoints towards
the goal location, represented as GPS coordinates, with the low-level policy rewarded for how close it
gets to them. For vacations, the goals could be booking day one, day two and day three, or something
coarser such as booking travel, booking hotels and booking specific activities; Finn adds that the agent
first needs to work out what the user wants from the vacation, doing high-level research, which is
itself an intermediate goal. For cooking, the goals are steps like opening the cabinet and beating the
eggs. Their representation could simply be natural language, literally the string "beat the eggs"; or,
when the food should look a certain way, an image ("I want the bread to look like this"); so goals can
be multimodal.

The lessons are on slide 14 (≈22:27–24:46). First, "Best goal representations will likely be domain
specific!": GPS coordinates for biking, language for cooking and for booking a vacation, possibly with
images involved. Second, good representations share three properties:

- **Expressive**, able to "communicate many different low-level behaviors". To cook many different
  meals, natural language is expressive enough to describe many low-level behaviours.
- **Structured**: "similar behaviors should have similar $\mathbf{g}$". If the goals were only labels like "subtask one, do
  subtask two", the low-level policy would find it hard to see how they relate; with
  GPS coordinates, two nearby coordinates call for similar actions, and that structure makes the low
  level's learning problem much easier.
- **An appropriate level of abstraction**, "not too hard for low-level or for high-level". For the bike,
  the exact velocity to pedal for the next second is too low-level, while "go to the quad" is too
  high-level to help break the problem into smaller steps. Choosing the right level is often tricky, and
  domain dependent.

Should each level also get an abstracted state space (≈24:46–25:31)? That depends on whether such an
abstraction is available. Often even the high-level policy benefits from all the information about the
current state to make its goal predictions; one could, for example, ask a language model to summarize
the high-level information in a state and give that to the high-level policy. But in practice both
levels often use the full state, and little is lost by doing so.

## Design choice 2: how to supervise each level

Finn takes supervision through an example drawn on the board, making pasta, a slightly simpler
long-horizon task (≈25:31–28:36). No slide holds the drawing, so what follows is reconstructed from the
spoken description. In an abstract picture of the state space, the task starts from an initial state
distribution. The steps are to get out a pot and the pasta, boil the water, and pour the pasta into the
boiling water (a straining step is skipped). Each step corresponds to a set of states: states where the
pot has been taken out, before any water is in it; states where the water has boiled; states where the
pasta has been found; and states where the pasta is cooked. The low-level policy works out how to take
actions to reach each of these sets, rewarded according to the low-level task; the high-level policy
predicts, from a given state, which command to give the low-level policy: find a pot, boil the water,
find the pasta, cook the pasta.

### End to end is a flat policy

Why not train everything end to end, with a latent goal representation (slide 16, ≈28:36–30:10)? The
goal would just be the network's activations, and both levels would be trained only on the high-level
task and its rewards. The "**Result**: a flat policy" (slide 16): a flat policy with a particular
architecture, which is basically a single neural network. Finn stresses why this matters: when
designing these systems, think about where the benefits come from, whether from additional supervision
or from structured exploration. Without that, one can end up intuitively expecting the high level to do
one thing and the low level another, and see no benefit over naively training a single policy. Slide 16
makes the point general: "Think carefully about where the benefits are coming from!", "Generally good to
do in any kind of research!!". For anyone doing research, being honest about what might go wrong, and
about why a new idea might not show benefits, makes it easier to think critically about what will
actually improve the final results (≈29:23–30:10).

### What each level is trained for

Slide 17 states the two objectives (≈30:10–33:15). The low-level policy is "Trained to accomplish
$\mathbf{g}$, not the original task!": given the string "find pot", it is rewarded for reaching the
states where the pot has been found. The high-level policy is "Trained to accomplish original,
long-horizon task": it predicts subtasks so that the agent ends in the state where the pasta is cooked.

Training them separately is where it gets tricky (≈30:57–33:15). Suppose the find-the-pot policy has
become very reliable, and a find-the-pasta policy is then trained, carelessly, from a start state near
the pot. If the high-level policy actually wants the pasta found after the water has boiled, when the
agent is at the stove, the two low-level skills will not string together: the robot will struggle to
transition from the stove to finding the pasta. Training modules completely independently can give a
system that does not work, unless at some point they are connected and trained so that each skill starts
from the states in which the high-level policy will actually command it. The low-level policy should work
for the states where the high level will command each goal, and for the goals the high level will
predict. Hence slide 17's key questions: the low level's goals should be drawn from "Ideally whatever the
high-level policy will output", "... but we haven't learned a high-level policy yet"; and the high level
should be trained "Ideally with the learned low-level policy", "... but we haven't learned a low-level
policy yet." A high-level policy that assumes a perfect low level will not learn its deficiencies, for
example that it must keep commanding a skill until the agent reaches a state where a different skill can
operate.

A student points out that the high level can walk the low level through particular sequences of states
(≈33:15–34:00). Finn agrees: the low-level policy does not need to work from every state, only for the
state-goal combinations the high-level policy will command. But there must be no **gaps** (≈34:00–35:32).
If the low level knows how to go from one set of states to the next, but finding the pasta requires going
back to states it has no command for, the high-level policy will struggle to reach the next set. Successive
skills need at least some overlap in their states, ideally with some buffer: if finding the pasta only
works when standing exactly in front of the boiled water, and boiling the water ends a little to the
right, the result is brittle.

Must the sequence be fixed (≈35:32–36:17)? No: the high-level policy decides the order, which can depend
on what is most efficient and on where the low-level policy is most successful. Finn chose the pasta
example partly because finding the pot and finding the pasta can be done in either order; as long as the
low level can find the pasta from the relevant states, a different order might be more efficient, for
instance finding the pasta while waiting for the water to boil.

### Adapt the levels to each other

The way to address all this is on slide 18 (≈36:17–37:04): "**Key:** Can be trained separately first.
But, at least one policy should be adapted to the deficiencies of the other." "(if not both fine-tuned
jointly)": ideally both are adapted together to optimize reward, or some complete objective.

Slide 18 also notes "LLMs are often good high-level policies!" (≈37:04–38:35). Even for problems that are
not traditionally language problems, LLMs are often good at the higher-level reasoning: asked the steps
for cooking pasta, they can often give them. So the high-level policy can be trained completely separately
from the low-level policy and then fine-tuned, because the language model does not know, for example,
which order the low-level policy is best suited for, or other things about its capabilities. And a robot
standing in front of a coffee machine, asked to get some coffee, might be told by a language model that
ignores the image to go out and buy coffee at Starbucks, an affordance the robot does not have. Adapting
the high-level and low-level policies together is therefore important even when language models are good
at the high-level task.

## Design choice 3: when to move on to the next goal

The roll-out procedure left open how long to keep sampling from the low-level policy before re-querying
the high-level policy for a new goal: step 2, "Until new goal is selected", highlighted on slide 20
(≈38:35–39:21). Slide 21 gives the two natural options (≈39:21–44:41).

**Option 1: when the low-level policy has completed $\mathbf{g}_ t$**, "e.g. by estimating progress
towards $\mathbf{g}$". When the water has boiled, tell the low level to go and find the pasta. The
low-level policy must then predict not only the action but also whether it has completed the goal, with a
binary indicator or a progress indicator; this done-or-not-done output is a success classifier for the
subtask. Done well, this is "ideal in principle" and the most compute-efficient choice, because there is
no need to rethink the task while the low level is executing. But it is hard: estimating when a task is
done is "definitely not trivial", and the system must "account for mistakes that require redoing past
goals". If a robot organizing a table drops an object on the ground, its subgoal should no longer be to
put that object on the table but to pick it up off the ground, so the system must estimate not only when
it is done but whether it has left the states in which the subtask makes sense.

**Option 2: after a fixed number of $n$ timesteps**, "e.g. by choosing fairly frequent interval to
replan". If the low-level policy runs at 50 hertz and the high-level policy is queried once a second, it is
queried every 50 time steps. This is "simple", but it probably wastes compute, and it trades compute
against delay: with $n$ too long, the agent that has found and taken out the pot just sits there with it,
waiting for a new goal, before it moves on to finding the pasta or pouring water into the pot. Because
long delays are unwanted, $n$ is usually fairly small; but a small $n$ puts "more burden on HL policy",
which must predict accurate goals more frequently, on more states, and spends more compute.

The options also fail differently (≈43:09–44:41). With option 1, an error in estimating when the task is
done can leave the agent perpetually stuck: if it thinks it is not done when it is, the low-level policy
keeps going and makes no further progress, a rough outcome even if such errors are infrequent. With
option 2, an error in predicting the subtask means the low-level policy "may take wrong actions for $n$
steps" (slide 21) until, hopefully, a more accurate goal arrives. Option 1's errors are the more fatal
("These errors are more fatal!", slide 21). As a result, many hierarchical systems choose option 2, for its
simplicity and because the mistakes a frequently queried high-level policy makes are less fatal than
mistakes in estimating when a subtask is done or whether to return to a past goal. The two can be combined:
estimate progress, but re-query anyway if a long time has passed. The takeaway is that option 2, querying
at a fixed, lower frequency, is what people typically do.

## Hierarchy in current robot systems

"Hierarchy is hot" (slides 23–24, ≈44:41–46:16): four systems announced in the four or five months before
the lecture all use hierarchy. Slide 23 shows stills from their videos: Physical Intelligence's
$\pi_{0.5}$, NVIDIA's Gr00t N1, Figure's Helix and Gemini Robotics. Their tasks are fairly long-horizon:
cleaning a bedroom, handing over objects and pouring them or putting them into bins, putting objects away
into a fridge, putting fruit into bowls.

![Slide 23 — Hierarchy is hot](../raw/images/15-hierarchical-rl/slide-23.jpg)

*Slide 23 — stills from the videos of four recent robot systems (Physical Intelligence's π0.5, NVIDIA's Gr00t N1, Figure's Helix, Gemini Robotics), all built hierarchically. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

Their architectures (slide 24) all have a higher-level model that predicts intermediate goals and a
lower-level model that predicts low-level actions. In $\pi_{0.5}$'s diagram, a pre-trained VLA takes camera
images and the high-level prompt "clean the bedroom", predicts the subtask "pick up the pillow", which is fed
back as the low-level command, and an action expert produces continuous actions. Gr00t N1 pairs a
vision-language model ("System 2") with a diffusion transformer ("System 1") that outputs motor actions.
Helix pairs a 7B pretrained vision-language model for "Infrequent Vision-Language Semantic Reasoning"
(System 2, at 7-9 Hz) with an 80M transformer for "Fast, Reactive Control" (System 1), joined by a latent
vector and outputting control at 200 Hz. Gemini Robotics splits a cloud backbone from a local action decoder.
In language-model domains, chain of thought, reasoning through intermediate steps before outputting tokens
for them, is likewise popular for harder language tasks (≈45:27).

![Slide 24 — Hierarchy is hot](../raw/images/15-hierarchical-rl/slide-24.jpg)

*Slide 24 — the four architectures: in each, a slower high-level model (a pretrained VLA predicting a subtask, or a vision-language "System 2") feeds a faster low-level model that outputs actions. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

Finn picks examples that illustrate different design choices and that come with comparisons showing the
benefit of one choice over another (≈46:16). The outline (slide 25) is hierarchical imitation learning with
language subgoals and with image subgoals, then hierarchical reinforcement learning with language and with
state subgoals.

## Hierarchical imitation with language subgoals

The data for a hierarchical imitation learning system with language subgoals, here on a real robot, is a
long-horizon demonstration of the complete task, segmented, with a language label for each segment
(slide 27, ≈46:16–48:33). In the first example the task is to make trail mix and put it into a bag; the
segments are labelled "pick up the bag", "pick up the metal scoop", "I want some M&Ms", "I want less", "open
the bag with scoop" and "pour it into the bag". Segments have variable length: Finn draws a robot trajectory
whose first subtask is one length, the second shorter and the third longer. The deployed system, however,
re-queries the high level at a fixed frequency, more often than the typical segment length in the data.

![Slide 27 — Hierarchical imitation with language subgoals](../raw/images/15-hierarchical-rl/slide-27.jpg)

*Slide 27 — the training data: long demonstrations (making trail mix; building a sandwich) cut into variable-length segments, each with a language label that becomes the high-level policy's target. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

The labels can be recorded during data collection, by saying "I'm going to pick up the bag" and then
demonstrating it, then "I'm going to pick up the metal scoop", and so on; or post hoc, by collecting the full
demonstration and labelling its segments in an interface (≈47:47). The second example on slide 27, building a
sandwich, shows that one segment (picking up a slice of cheese) can be longer and another (putting the cheese
on the bread) shorter (≈48:33). (Slide 27 prints the last two labels of its third row both as "put bread on
lettuce".)

Both policies are then trained with imitation learning (slide 28, ≈48:33–50:07). The high-level language
policy takes the observation, an RGB image, and outputs the language label; the low-level policy, a
language-conditioned behavior cloning ("LCBC") policy, takes the language and observations and predicts the
actions. In this work the low level sees RGB images and joint angles, so the two levels see different
observations, which is fine. The low level outputs joint targets at 50 hertz; Finn cannot remember the high
level's exact frequency but thinks it is around 1 hertz.

![Slide 28 — Hierarchical imitation with language subgoals](../raw/images/15-hierarchical-rl/slide-28.jpg)

*Slide 28 — the ALOHA workcell and the architecture: a high-level policy maps RGB images to language, and a language-conditioned low-level policy maps language, images and joints to joint targets. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

## Fine-tuning the high level with DAgger

Imitation on the offline data alone trains the two levels somewhat independently, although the data is at
least of the full long-horizon task (≈50:07). So, "Could we also use DAgger on either policy? Yes!"
(slide 28; DAgger is from [Lecture 2](02-imitation-learning.md#corrective-data-dagger)). It can be used on
both levels, and slide 28's note is the striking part: "Can update the high-level policy with only language
supervision!" DAgger on the high-level policy happens in language space, not joint space.

In this approach (slides 29–30, ≈50:53–51:39), the low-level policy is frozen and the full system runs. A
person types or speaks a language command that overrides the high-level policy's prediction, keeps
collecting data, and at some point gives control back to the high level. The high-level policy is then
fine-tuned with gradient steps on the data containing these language interventions. Because the two
policies act as one system during this, the high level adapts to the deficiencies of the low level: if the
low-level policy makes a mistake on one command, corrections can compensate. In slide 30's words: "Freeze
low-level policy. Update high-level policy by supervising on language corrections."

![Slide 30 — Hierarchical imitation with language subgoals](../raw/images/15-hierarchical-rl/slide-30.jpg)

*Slide 30 — DAgger in language space: the low-level policy is frozen (snowflake), a person's spoken correction overrides the high-level policy's language output, and the correction becomes a gradient update for the high-level policy (flame). [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

A video on slides 31–32 shows it (≈51:39–53:10). The yellow commands are the high-level policy's output and
the robot's motions the low-level policy's. The high-level policy says pick up spoon, and the robot picks up
the spoon; with "I want some peanuts" the low level scoops peanuts; then the high level says move the scoop
into the bag. The robot is about to make a mistake (slide 31's subtitle is "e.g. avoid pouring outside the
bag"), so the person intervenes: instead of move the scoop into the bag, "move the left arm to the left", "go
higher", and then move the scoop into the bag. These user commands override the high-level policy; when
control returns, it predicts pour into the bag, then get some banana chips.

![Slide 31 — Language correction on-the-fly](../raw/images/15-hierarchical-rl/slide-31.jpg)

*Slide 31 — a frame of the correction video: the yellow text is the high-level policy's current prediction; in the video a person overrides it with commands such as "move the left arm to the left" and "go higher". [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

![Slide 32 — Language correction on-the-fly](../raw/images/15-hierarchical-rl/slide-32.jpg)

*Slide 32 — a later frame of the same video, with the high-level policy predicting "move the scoop into the bag". [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

After fine-tuning on such interventions, "the policy can self-correct" (slide 33, ≈53:10–53:56): the
high-level policy recognizes that the low-level policy is about to make the same mistake again, says "Go
higher", and then moves the scoop into the bag. This is the high-level and low-level policies adapting
together to account for deficiencies, as slide 18 asked.

![Slide 33 — After fine-tuning high-level policy with DAgger](../raw/images/15-hierarchical-rl/slide-33.jpg)

*Slide 33 — after fine-tuning on the corrections, the high-level policy issues the correction itself ("Go higher") before the low-level policy repeats its mistake. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

## Results: hierarchy against a flat policy

Does hierarchy beat a single flat policy here (slide 34, ≈53:56–54:41)? The comparison comes from a
different paper with basically the same approach (slide 34 cites "Hi Robot. ICML 2025" alongside "Yell At
Your Robot", RSS 2024). In task progress there is a 34% improvement from using hierarchy over training a
single policy on the full long-horizon imitation task, and there is also a gap in an instruction-accuracy
metric; the chart labels the two "TP" ("34% gap") and "IA" ("19% gap"), comparing a "Flat VLA policy" with a
"Hierarchical VLA policy". Comparing DAgger with plain imitation for the high-level policy also shows an
improvement, the gain from adapting the policies together: on "Final task performance", the "Base
hierarchical policy" scores about 27, "With high-level DAgger" about 51, and "With human corrections
(oracle)" about 55, with the first gap labelled "20% gap".

![Slide 34 — Hierarchical imitation with language abstraction](../raw/images/15-hierarchical-rl/slide-34.jpg)

*Slide 34 — left, the hierarchical policy beats the flat one on both metrics (19% and 34% gaps; no axis values are printed); right, high-level DAgger lifts final task performance from about 27 to about 51, near the human-correction oracle at about 55. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

With hierarchy, "the robot is better at long-horizon tasks" (slides 35–36, ≈54:41–55:27). The full system
can put three objects into a Ziploc bag, traditionally a difficult task, especially in one go. With a similar
system ($\pi_{0.5}$, slide 36), the high-level command "clean the bedroom" becomes low-level commands such as
pick up shirt, put shirt in basket, pick up shirt, put shirt in basket, and later the pillow, adjust the
blanket, put the pillow at the top of the bed, adjust the pillow, pick up tissue, put the tissue in the trash
can. Tasks like these would be very hard with a single flat policy.

![Slide 35 — With hierarchy, the robot is better at long-horizon tasks](../raw/images/15-hierarchical-rl/slide-35.jpg)

*Slide 35 — a frame of the video in which the hierarchical system packs three objects into a zip bag, starting from the instruction "pick up the bag". [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

![Slide 36 — With hierarchy, the robot is better at long-horizon tasks](../raw/images/15-hierarchical-rl/slide-36.jpg)

*Slide 36 — a frame of π0.5 cleaning a bedroom from the single human command "clean the bedroom", which its high level breaks into commands like "pick up shirt" and "put shirt in basket". [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

## Hierarchical imitation with image subgoals

Instead of language, the intermediate representation can be an image (slide 38, ≈55:27–57:00). The
high-level policy outputs a **goal image**, an image of what the low level should accomplish in the future.
"In practice: use an image editing model" (slide 38): a generative model takes the current image and edits it
to look like the image at a future time step the robot should reach. The low-level policy is then a
goal-image-conditioned policy rather than a language-conditioned one, which can be seen as a multi-step
inverse model: it takes the current image and a future image and outputs the actions that get from one to
the other. Both are again trained with imitation learning.

![Slide 38 — Hierarchical imitation with image subgoals](../raw/images/15-hierarchical-rl/slide-38.jpg)

*Slide 38 — the same architecture with the language output struck through and replaced by a goal image; the low-level policy becomes goal-image conditioned. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

The benefits (slide 38): "No need for segmented videos with language annotations." Only videos of the
long-horizon task are needed. The high-level policy can be trained on robot data to predict future images in
the trajectory, 1, 5 or 10 seconds ahead, and also on something like YouTube videos, predicting future images
that would accomplish a long-term goal: "High-level policy can incorporate unlabeled video data!", which
helps especially when robot data is sparse (≈57:00).

Slide 39 shows examples, from SuSIE ("Subgoal Synthesis via Image Editing", ICLR 2024), as frames of videos
(≈57:00–58:34). Told "move the wooden bowl to the top of the table", the high-level policy generates subgoal
images, shown in the videos with a pink border, and each generated image is passed to the low-level policy,
which predicts actions to reach it. For "put the toothpaste into the wooden bowl", it generates a subgoal of
holding the toothpaste above the bowl, which the robot reaches, then a subgoal of lowering the arm. For "open
the drawer", the first subgoal puts the gripper on the handle and the next moves the gripper to the right so
the drawer opens. Images also help with objects the policy has never seen and that are hard to describe in
language: it can generalize across objects and generate their pixels without knowing their names.

![Slide 39 — Hierarchical imitation with image subgoals](../raw/images/15-hierarchical-rl/slide-39.jpg)

*Slide 39 — frames of three SuSIE videos (move the bowl, put the toothpaste in the bowl, open the drawer); in the videos, generated subgoal images appear with a pink border and the robot then reaches each one. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

Is it helpful to include unlabeled videos when training the high-level policy (slide 40, ≈58:34–59:20)?
Adding videos of humans increases performance over robot data alone. Slide 40's table compares "BridgeData
Only" (robot data only) with "BridgeData + Something-Something" (robot data plus videos of humans):

| Scene | Task | Robot data only | Robot data + videos of humans |
| --- | --- | --- | --- |
| Scene B | Bell pepper in pot | 0.2 | 0.5 |
| Scene B | Bell pepper in bowl | 0.4 | 0.5 |
| Scene B | Average | 0.30 | 0.50 |
| Scene C | Toothpaste in bowl | 0.7 | 0.6 |
| Scene C | Crayon in bowl | 0.9 | 1.0 |
| Scene C | Spoon in bowl | 0.9 | 0.9 |
| Scene C | Bowl to top | 0.7 | 1.0 |
| Scene C | Average | 0.80 | 0.88 |

## Questions on image subgoals and representations

Several questions follow (≈1:00:50–1:04:41).

- **What exactly is the high-level policy here?** An image generative model, which Finn sketches on the
  board: trained to take an image observation and a high-level goal and to output $\mathbf{o}_ {t+k}$, an
  image of the future. With videos, no language annotations are needed to get such images. Architecturally
  it is a diffusion model, which is why the videos showed noise moving into the image (≈1:01:38–1:02:23).
- **Is the subgoal in the direction of the goal, and how far ahead?** Yes, in the direction of the goal. The
  goal might be on a horizon of about a minute and the subgoal perhaps 10 seconds in the future (≈1:02:23).
- **What do image-editing models and human videos contribute?** General pre-training teaches the model to
  generate images in general. The human videos used show people doing tasks, each with a single language
  annotation for the whole video, so they are especially relevant; random YouTube videos would help less but
  might still teach which images can follow in the future, useful pre-training for this task (≈1:02:23–1:03:09).
- **What if the goal is a sketch rather than a real image?** Ideally the low-level policy is trained to use
  sketches; there are approaches that convert images into sketches and train a low-level policy on them. The
  goal need not be a full image: language is a particularly popular intermediate representation, and there
  are many others, including sketches or a trace of where the robot should go (≈1:03:09–1:03:55).
- **What if the language is inaccurate?** The robot will often follow the different task until it is told to
  do something else, so if the high-level policy predicts bad language, the system suffers. Sometimes the
  low-level policy ignores the high level: given gibberish, it may fall back on a default behaviour, because it
  was trained to output actions. If it is holding the pasta over the pot, the next action is very likely to
  pour it whatever the language says, so with no instruction or a noisy one it may still pour (≈1:03:55–1:04:41).

## A chain-of-thought version, and RL as an open problem

Just as chain of thought can replace hierarchy, there are chain-of-thought versions of these systems
(slide 41, ≈59:20–1:00:05): from the high-level task and the current observation, one model outputs subgoal
images or subtasks, or other representations such as bounding boxes or waypoints, and then the action. Slide
41's figure, from "Training Strategies for Efficient Embodied Reasoning" (2025), shows one such chain for "Put
the corn in the dish rack": a plan, a subgoal image, the subtask, bounding boxes, movements, waypoints and a
description, then the action. These approaches are from roughly the last year.

![Slide 41 — Let's cover a few examples](../raw/images/15-hierarchical-rl/slide-41.jpg)

*Slide 41 — a chain-of-thought robot policy that writes out its intermediate reasoning (plan, subgoal image, subtask, bounding boxes, movements, waypoints, description) before the action, with the note that fine-tuning such systems with RL is open. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

All of this is imitation learning (≈1:00:05–1:00:50). To date, large-scale systems mostly use imitation rather
than reinforcement learning, and the best recipe for fine-tuning them with RL at scale is an open research
question: "Fine-tuning large-scale hierarchical robot learning systems with RL is an open (and important!)
research direction" (slide 41). It matters because these systems are still not very reliable, as the results
on slide 40 show, though Finn notes those are challenging settings with held-out objects the robot has not
seen, out of distribution relative to training. Training with RL should be able to push that performance much
higher than imitation alone.

Could hierarchy help personalized recommendation systems (≈1:00:50)? Finn has not seen it used there; in their
experience hierarchy is usually used for longer-horizon, more complicated tasks, and in language modelling for
things like math problems that take multiple steps. Finn says they will think about it.

## Hierarchical reinforcement learning

Finn finds the hierarchical imitation systems quite recent and has fewer exciting examples of hierarchical RL
(slide 42, ≈1:04:41–1:05:28). Getting hierarchical RL to work really well on top of initialized hierarchical
imitation systems would be exciting and important to explore. Slide 43 gives two approaches.

**State-reaching goals**, "(e.g. relative target position of agent & objects)" (≈1:05:28–1:06:59). One work
uses the four-legged ant agent from Homework 1. In one task it must reach a target, and to get there it has to
push an object to the right and then move through; in another, going straight to the target would drop it into
a crevasse, so it must push a block in to make a little bridge and walk over it. Taking actions greedily will
not reach the goal in these long-horizon problems. The work trains a goal-conditioned low-level policy, with
a reward for reaching goal states, and a high-level policy that outputs goal states, which are its high-level
actions. It also uses hindsight relabeling on the high-level policy, relabeling the goals the high level should
have output in certain circumstances, and an off-policy algorithm to make this work (slide 43: "Use hindsight
relabeling *on HL actions*, using off-policy algorithm"; see [hindsight relabeling](hindsight-relabeling.md)).

![Slide 43 — Hierarchical reinforcement learning](../raw/images/15-hierarchical-rl/slide-43.jpg)

*Slide 43 — hierarchical RL with state goals: in each pair the ant (magenta box) must first move the red block, aside or into a gap as a bridge, before it can reach the target opening (green arrow). The black square is a video the PDF does not hold. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

**Language goals**, "(for simple semantic tasks)" (≈1:06:59–1:07:44). Another work trains a
language-conditioned low-level policy with hindsight relabeling in language space rather than goal space, and
a high-level policy that outputs language, which is a pretty difficult task. The work is from 2019 (slide 43
cites "Language as an Abstraction for Hierarchical Deep Reinforcement Learning", NeurIPS '19), and its tasks
are relatively simple: arranging objects in a particular order, pushing the green object, then the red, then
the blue. (The video of these tasks is not in the PDF; slide 43 shows a black square in its place.)

## Discovering skills without supervision

So far the low level has been supervised to reach chosen goals and the high level to output them. "Another
active research area": "Can you *discover* a set of diverse skills without supervision?" (slide 44,
≈1:07:44–1:09:15). One work, "Diversity is all you need" (ICLR 2019), shows that a simple half cheetah agent,
which students may have seen in a homework, learns skills such as backing up, walking forward and, Finn thinks,
a front flip. In principle, skills discovered without supervision could then be sequenced to do harder tasks.
This remains an active research area, and challenging to scale up, especially in very large state spaces.

![Slide 44 — Another active research area](../raw/images/15-hierarchical-rl/slide-44.jpg)

*Slide 44 — six poses of the half cheetah from skills discovered without any reward function; the slide does not say which skill each frame shows. [Deck](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf)*

Next time is RL for robotics, specifically learning with autonomy ([Lecture 16](16-autonomous-rl.md)), and a guest lecture from Ashish Kumar of the
Tesla Optimus team (slide 46, ≈1:09:15); slide 46 names its topic as "sim2real transfer".

## Not in the recording: the summary and reminders

Slide 45 repeats the outline as a summary, "Hierarchy for Imitation and Reinforcement Learning": what's the
problem (why long-horizon tasks are hard, why hierarchy may help), the key design choices (how to represent
skills and goals, supervision for each level, when to move on from one goal to the next) and example systems.
Slides 2 and 46 carry the course reminders: "Project milestone due Friday." and "Homework 4 due next Friday".
Slide 2 adds, "Based on course feedback", "Posting tips on Ed for default project!". See
[course logistics](course-logistics.md).

## Summary

Long-horizon tasks are hard because the agent visits many states, has many chances to err and many chances to
get stuck. Hierarchy splits the policy into a high-level policy that predicts intermediate goals at a low
frequency and a low-level, multi-task policy that pursues the current goal by predicting actions at the original
frequency. It can help through intermediate supervision, knowledge sharing across similar subtasks, structured
exploration in RL and lower compute; a chain-of-thought policy that predicts the goal and then the action keeps
the supervision but not the compute saving, and no conclusive comparison exists yet. Goal representations should
be expressive, structured and at the right level of abstraction, and are usually domain specific (GPS
coordinates, language, images). Each level is trained for its own objective, but trained end to end with a latent
goal the hierarchy is just a flat policy, and trained fully independently its skills may not chain; at least one
level must be adapted to the other's deficiencies, as when a language model serves as the high level. Re-querying
the high level after a fixed number of steps is simpler and fails less fatally than waiting for the low level to
report it is done, so most systems do that. Current robot systems are hierarchical; with language subgoals, the
high level can be fine-tuned with DAgger in language space on human corrections, and hierarchy beat a flat policy
in task progress by 34%. Image subgoals from an image-editing model need no language labels and can use human
videos. Fine-tuning these systems with RL is open; hierarchical RL has used state and language goals with
hindsight relabeling, and discovering skills without supervision is an active research area.

## Related pages

- [Hierarchical RL and imitation](hierarchical-rl.md): hierarchy as a topic: the two-level policy, why it helps,
  the design choices and the example systems.
- [Multi-task RL](multi-task-rl.md): the task- and goal-conditioned policies that serve as the low level.
- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md): where the question on hierarchy first came up, and goal-conditioned
  RL with hindsight relabeling.
- [Hindsight relabeling](hindsight-relabeling.md): relabeling goals, used for the high level's actions and in
  language space in hierarchical RL.
- [Imitation learning](imitation-learning.md) and [Lecture 2](02-imitation-learning.md): behavior cloning and DAgger,
  here applied to the high level in language space.
- [Exploration](exploration.md): structured exploration in a goal space.
- [RL for LLM reasoning](llm-reasoning.md): chain-of-thought reasoning, the single-policy alternative to hierarchy.
- [Course logistics](course-logistics.md): the project milestone and Homework 4 deadlines.
