# Course logistics — CS224R, Spring 2025

How the Spring 2025 offering ran, from Lecture 1's logistics slides and the course website.
**The recorded lecture does not include these slides** (slides 5–6 and 10–15), so everything
below is from the [slide text](../raw/slides/01-class-intro.md) or the site, not from anything
said in the video. Enrolled students should treat the course website and Canvas as
authoritative. This page records a past offering.

## Staff

- **Instructor:** Chelsea Finn, an assistant professor at Stanford whose research covers
  reinforcement learning, robotics and language models (Lecture 1, ≈0:05).
- **Head teaching assistant:** Jubayer Ibn Hamid. **Course managers:** Amelie Byun and John Cho.
  Slide 3 also shows eighteen further staff members by name, without roles.
- **Guest lecturers** (slide 6): Ashish Kumar (Tesla), Archit Sharma (Google DeepMind) and one
  still to be announced. The site's schedule has Archit Sharma on RL for LLMs: preference
  optimization ([Lecture 9](09-rl-for-llms.md)), Aviral Kumar on RL for LLMs: reasoning
  ([Lecture 10](10-rl-for-reasoning.md)) and Ashish Kumar on
  RL for robots: sim-to-real transfer ([Lecture 17](17-sim-to-real.md), whose title slide reads
  "Advancing Robot Intelligence with Reinforcement Learning").
- **Contact** (slide 5): the course website, <https://cs224r.stanford.edu/>; Ed and Gradescope,
  through Canvas; and the staff mailing list, cs224r-staff-spr2425@cs.stanford.edu, which reaches
  the student liaison, course manager, head CA and instructor. OAE letters go to the mailing list
  or in a private Ed post.

## Lectures and office hours

Lectures are in person, livestreamed and recorded, and are meant to be interactive: the
instructor asks questions and wants students to ask them too. Office hours are a mix of in-person
and remote, listed on the website and Canvas (slides 5–6).

## Prerequisites

From slide 10 and the site:

- **Machine learning:** CS229 or equivalent. Knowledge of stochastic gradient descent,
  cross-validation, calculus and probability theory is assumed.
- **Some deep learning:** the course builds on backpropagation, neural networks and sequence
  models (the site adds convolutional networks and transformers), and the assignments require
  training networks in PyTorch. A PyTorch review session was held on the first Friday, 1:30 pm in
  Gates B1, led by Marcel (slide 10).
- **Some reinforcement learning:** the basics are covered quickly. For introductory material on
  RL and MDPs the site points to CS221's modules and to chapters 3 and 4 of Sutton & Barto (both
  linked in [`sources.md`](../sources.md#background-reading-the-site-recommends)).

The site describes the course as complementary to CS234, with neither a prerequisite for the
other: CS224R has a more applied, deep-learning focus with use-cases in robotics and language
modeling. Lecture 1 points to CS 234 for more theory and other applications (slide 9, ≈4:43).

## Coursework and grading

From slides 11–12:

- **Four two-week homework assignments, 50% in total.** The lowest-scoring homework counts for
  5% and the other three for 15% each. Homeworks implement methods in PyTorch and run experiments
  in physics simulators and navigation environments:
  1. imitation learning;
  2. online reinforcement learning;
  3. offline reinforcement learning;
  4. goal-conditioned and meta reinforcement learning.
- **A final project, 50%**, default or custom, in teams of 1–3: proposal 10%, milestone 10%,
  poster 10%, report 20%. The custom project is a topic of the team's choosing, and students are
  encouraged to use their own research. The default project, new in 2025, is to fine-tune an LLM
  with RL plus an open-ended extension.
- **Late days:** 6 free late days, at most 2 per assignment unless arranged in advance; after
  that, 2% of the course grade per day late. The site adds that no late days are allowed for the
  final project report.
- **Collaboration and AI tools:** read the course website, the honor code and the AI tools
  policy; document collaborators, write solutions on your own and submit independently.
  Substantial use of AI tools (e.g. ChatGPT, Cursor) is not allowed for homework or for parts of
  the default project.

Lecture 11's reminders put Homework 3 due Friday, May 16, and the project milestone due Friday,
May 23 (Lecture 11, slides 2 and 38); Lecture 12 repeats them (Lecture 12, slides 2 and 29). Lecture 12's
hindsight relabeling is "Part of homework 4!" (Lecture 12, slide 3): Homework 4's handout, due Friday,
May 30, has students make a Deep Q-Network goal-conditioned and implement Hindsight Experience Replay on
top of it, before a second part on meta-RL. See [hindsight relabeling](hindsight-relabeling.md). Homework due dates and links are in
[`sources.md`](../sources.md#assignments-and-project). Lecture 13's reminders repeat "Homework 3 due
**Friday**" and "Project milestone due **next Friday**", and announce "More evening office hours!" "(Andy
and Marcel moving to evening times)" (Lecture 13, slides 2 and 39). Its plan brackets the meta-RL problem
statement, black-box meta-RL methods and the next lecture's exploration as "part of HW4" (slides 5, 15
and 38). See [meta-RL](meta-rl.md). Lecture 14, on Friday, May 16, reminds students "Homework 3 due
**tonight**", "(and HW4 out today)" and "Project milestone due **next Friday**" (Lecture 14, slides 2 and
34), and marks "Efficiently Meta-Learning Optimal Exploration" as "Part of HW 4" (slide 3): students
implement its DREAM algorithm in Homework 4 (≈1:04:26). See [exploration](exploration.md). Lecture 15, on
Wednesday, May 21, reminds students "Project milestone due Friday." and "Homework 4 due next Friday", and,
"Based on course feedback", announces "Posting tips on Ed for default project!" (Lecture 15, slides 2 and
46). Lecture 16, on Friday, May 23, reminds students "Project milestone due tonight" and "Homework 4 due next
Friday", and names the next lecture as a guest lecture from Ashish Kumar (Tesla Optimus) (Lecture 16, slides 2
and 46; the recording does not discuss them). The
poster session was Wednesday, June 4, and the final report was due Monday, June 9 (site).

## Advice from the course

Deep RL methods take time to learn behavior. The homeworks are designed to train quickly, for
example in simple environments, but they still take time, and an ambitious project may take
more. The staff recommend not starting homework or project deliverables the night before the
deadline (slide 13). The staff also expect to make mistakes and want feedback; a detailed
feedback form went weekly to a subset of students (slide 14).

## First steps (slides 15 and 45)

Homework 1 came out on Friday of week 1 and was due Friday, April 18, at 11:59 pm PT. Slide 45
prints "due Weds 4/18", but slide 15 and the site both give Friday, and April 18, 2025 was a
Friday. Students were also asked to start forming final-project groups. The second lecture,
on imitation learning, was that Friday at 10:30 in Hewlett 200, followed by the PyTorch tutorial
at 1:30 in Gates B1.

## Where this appears

- [Lecture 1 — Class Intro](01-class-intro.md): slides 3–15 and 45.
- [Lecture 11 — Model-Based RL](11-model-based-rl.md#not-in-the-recording-reminders-and-next-time):
  the Homework 3 and milestone dates.
- [Lecture 12 — Multi-Task RL](12-multi-task-rl.md#not-in-the-recording-the-rest-of-goal-conditioned-relabeling-and-next-time):
  the same dates, and hindsight relabeling in Homework 4.
- [Lecture 13 — Meta RL](13-meta-rl.md#not-in-the-recording-the-plan-next-time-and-reminders): the
  reminders again, the evening office hours, and meta-RL in Homework 4.
- [Lecture 14 — Exploration](14-exploration.md#not-in-the-recording-the-plan-and-reminders): Homework 3
  due, Homework 4 out, the milestone, and DREAM in Homework 4.
- [Lecture 15 — Hierarchical RL and IL](15-hierarchical-rl.md#not-in-the-recording-the-summary-and-reminders):
  the milestone and Homework 4 dates, and tips on Ed for the default project.
- [Lecture 16 — RL for Robots: Autonomous Learning](16-autonomous-rl.md#not-in-the-recording-the-reminders):
  the milestone and Homework 4 dates, and the next guest lecture.
- [Lecture 17 — Advancing Robot Intelligence](17-sim-to-real.md): Ashish Kumar's guest lecture, which
  carries no course reminders.
