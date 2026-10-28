---
title: Lecture 4 — Actor Critic Methods (slide deck)
lecture: 4
slides: 37
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–37). Page 6 prints no number; its label is inferred from position. Pages 21 and 29–34 print the number in the bottom-right corner.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (6, 14, 19, 21, 30, 35) were checked by Claude Opus 5.5 against 1400px renders, plus slides 22, 26 and 27 to check the transcriber's listed doubts. Three slides were corrected. Slide 19's trajectory layout was clarified. Slide 27 had been given slide 26's loop arrow. On slide 35's chart, the widest shaded band is PPO's, not SAC's, and two curve values were off (PPO at 6 million steps, SQL at 3 million).
images: 21 slides are rendered and shown under their headings (3, 6, 7, 8, 9, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 27, 29, 30, 31, 32, 35). Not rendered — slides 11, 15 and 26, build steps whose content slides 12, 16 and 27 show (slide 26's one extra mark, a small arrow from step 5 back to step 4, is described under it); slides 4, 10, 33 and 34, equation slides whose highlights, strike-throughs and pointer arrows are transcribed here; and the title, reminder, plan and text slides (1, 2, 5, 13, 24, 25, 28, 36, 37).
---

# Lecture 4 — Actor Critic Methods: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 4 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 37 pages. Page 6 prints no number; its label 6 is inferred from its position between slides 5 and 7. Pages 21 and 29–34 print their number in the bottom-right corner rather than at the bottom centre; those numbers are real.

The recording covers slides 3–23. It opens on the recap (slide 3) without discussing slide 2's reminders, and it ends at about 1:03:07 on the review slide (23). Slides 24–37 — the off-policy actor-critic section (importance weights and the KL and PPO ideas, replay buffers, fixing the value function and the policy update, soft actor-critic), the comparison of more and less off-policy methods, the plan recap and the next-week slide — are not discussed in this recording, so for them this file is the only source.

Companion pages: [wiki page for this lecture](../../wiki/04-actor-critic.md) · [transcript](../transcripts/04-actor-critic.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–5 | Title, course reminders, recap of policy gradients and on-policy versus off-policy, plan for today |
| 6–10 | Value, Q and advantage functions, a worked example, what is dissatisfying about policy gradients, improving them with Q and baselines |
| 11–14 | Online RL outline with a fitted model, plan marker, estimating expected return (why fit $V^\pi$) |
| 15–20 | Estimating $V^\pi$: Monte Carlo, bootstrapping, comparison, N-step returns |
| 21–24 | Discount factors, a full actor-critic walkthrough, review, plan marker |
| 25–28 | Off-policy actor-critic, version 1: multiple gradient steps, importance weights, KL constraint and PPO idea, towards replay buffers |
| 29–34 | Off-policy actor-critic, version 2: replay buffers, fixing the value function and the policy update, remaining problems, implementation details |
| 35–37 | More versus less off-policy comparison, plan recap, next week and course reminders |

## Slide 1 — Actor Critic Methods

Title slide. Title: "Actor Critic Methods". Below: "CS 224R". No images.

## Slide 2 — Course reminders

- Start forming **final project groups** (survey due next Wednesday)
- **Homework 1** due next Friday

## Slide 3 — Recap of Last Time

![Slide 3 — Recap of Last Time](../images/04-actor-critic/slide-3.png)

Left: a two-box cycle. A light blue rounded box "Run policy to collect batch of data" and a light pink rounded box "Improve policy using batch of data", joined by two grey curved arrows forming a loop (blue box to pink box down the right side, pink box back to blue box up the left side).

Right, heading: "Online reinforcement learning with policy gradients"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

The parts of the equation are colour-highlighted and labelled underneath: the two sums $\sum_{i=1}^{N}\sum_{t=1}^{T}$ in a blue highlight labelled "samples from policy"; $\log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$ in a pink highlight labelled "policy log likelihood"; the inner sum $\sum_{t'=t}^{T} r(\ldots)$ in a green highlight labelled "reward to go"; and $-b$ in a teal highlight labelled "baseline".

Bottom: "Do more of the above average stuff, less of the below average stuff."

## Slide 4 — Recap of Last Time

Left: "Vanilla policy gradients is fully **on-policy**." (the words "on-policy" in orange-red). Below, indented: "—> only one gradient step per batch of data".

Upper right, an orange-red label "attribute of online RL algorithms" above a pink box with a red outline, headed "Definitions.":
- *on-policy*: update uses only data from current policy
- *off-policy*: update can reuse data from other, past policies

Below left, "Importance weights" (in orange) "for *off-policy* policy gradient:"

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

The ratio $\pi_{\theta'}/\pi_\theta$ is in a pale yellow highlight.

Bottom: "**Note**: Not a silver bullet. Only suitable when policies are very similar."

## Slide 5 — The plan for today

"Actor critic methods"

1. Improving policy gradients
2. How to estimate the value of a policy
   a. Sample & directly supervise
   b. Use your own estimate
   c. Something in-between?
3. Off-policy actor-critic
   a. Importance weights & constraining step size
   b. Full off-policy version with replay buffers

"**Key learning goals**:"
- How to estimate how good a state and action is for a policy
- How to use those estimates to form a more efficient RL algorithm

## Slide 6 — Revisiting Some Useful Objects

![Slide 6 — Revisiting Some Useful Objects](../images/04-actor-critic/slide-6.jpg)

(No printed slide number; the label 6 is inferred from position.)

- *value function* $V^\pi(\mathbf{s})$ - future expected rewards starting at $\mathbf{s}$ and following $\pi$
- *Q-function* $Q^\pi(\mathbf{s}, \mathbf{a})$ - future expected rewards starting at $\mathbf{s}$, taking $\mathbf{a}$, then following $\pi$

Two diagrams side by side. Left: a blue dot labelled "s" at the lower left with five black wavy trajectory arrows fanning out from it to the upper right, labelled "run $\pi$". A thin red double-headed arrow running alongside, parallel to the trajectories, is labelled "expected sum of these rewards". To the right of it the label $V^\pi(\mathbf{s})$ with the s in blue.

Right: the same blue dot "s" with a similar fan of trajectories, but the trajectories are grey except for two (and a short arrow at the start) that are drawn in green; the label "take action **a**, then run $\pi$" (the a in green). A red double-headed arrow labelled "expected sum of these rewards" (green text on a light-grey band) runs alongside. To the right the label $Q^\pi(\mathbf{s}, \mathbf{a})$ with s in blue and a in green.

"Useful relation:"

$$V^\pi(\mathbf{s}) = \mathbb{E}_ {\mathbf{a} \sim \pi(\cdot \mid \mathbf{s})} \left[ Q^\pi(\mathbf{s}, \mathbf{a}) \right]$$

- *advantage* $A^\pi(\mathbf{s}, \mathbf{a})$ - how much better it is to take $\mathbf{a}$ than to follow policy $\pi$ at state $\mathbf{s}$

$$A^\pi(\mathbf{s}, \mathbf{a}) = Q^\pi(\mathbf{s}, \mathbf{a}) - V^\pi(\mathbf{s})$$

## Slide 7 — Let's look at an example

![Slide 7 — Let's look at an example](../images/04-actor-critic/slide-7.jpg)

Left: a black stick-figure person labelled $\mathbf{s}_ t$. Three black arrows fan out from the person, labelled $\mathbf{a}_ 1$ (lower arrow, pointing down-right), $\mathbf{a}_ 2$ (middle, pointing right) and $\mathbf{a}_ 3$ (upper arrow, pointing up-right). At the end of $\mathbf{a}_ 3$ is a black stick figure playing a drum kit with sticks; at the end of $\mathbf{a}_ 2$ is a black stick figure slumped on a seat facing a television or monitor; at the end of $\mathbf{a}_ 1$ is a black stick figure reclining on a lounge chair.

Centre: "Reward = 1 if I can play it in a month, 0 otherwise", above a picture of a sheet of drum music notation. Its heading reads "IMPROVISATION TEST EXAMPLES AND IDEAS FOR ROCKSCHOOL GRADE 1 DRUMS EXAM" with a small credit line beneath it ("Written by Theo Lawrence / TL Music Lessons"); eight numbered exercises (rock, funk rock, blues) are shown, the rest of the notation is too small to read [unclear].

Right, in teal: "Question:" then "For each action, what are these?"
- Value function: $V^\pi(\mathbf{s}_ t) = ?$
- Q function: $Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = ?$
- Advantage function: $A^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = ?$

Bottom left: "Current policy: $\pi(\mathbf{a}_ 1 \mid \mathbf{s}) = 1 \thinspace \forall \mathbf{s}$"

(A small stray mark sits just left of the printed "7" in the footer.)

## Slide 8 — What is dissatisfying about policy gradients?

![Slide 8 — What is dissatisfying about policy gradients?](../images/04-actor-critic/slide-8.jpg)

Top left: the same two-box cycle as slide 3 (blue "Run policy to collect batch of data", pink "Improve policy using batch of data", grey arrows in a loop). Top right: the same colour-labelled policy gradient equation as slide 3, with labels "samples from policy", "policy log likelihood", "reward to go", "baseline":

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

Bottom left: a small still of a black stick-figure humanoid in a simulator on a light grey tiled floor, mid-stride and leaning back. Text: $\tau^4$: "one small step forward then falls backwards". Below in red: "-> pushes down likelihood of step forward".

Bottom right: a small photograph of a grey-and-black humanoid robot with a blue-lit face standing at a table, with a jacket spread in front of it, in front of white shelving. Text: $\tau^2$: "folds only the sleeves", $\tau^3$: "flattens the jacket but does not fold it", $\tau^4$: "folds the jacket". Below in red: "-> with sparse rewards, don't utilize $\tau^2$ or $\tau^3$".

Bottom line: "Policy gradients doesn't make efficient use of data!" followed by a light-bulb emoji and "Can we learn what is good & bad?"

## Slide 9 — Improving policy gradients

![Slide 9 — Improving policy gradients](../images/04-actor-critic/slide-9.jpg)

Subtitle: "Estimating reward to go".

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right)$$

The parenthesised sum is in a pale teal highlight with a black brace below it labelled "reward to go".

Lower left: a green dot at the base of a long black wavy line, from which six black wavy trajectory arrows fan out to the upper right; a thin red double-headed arrow alongside is labelled "sum together these rewards". Right of the picture, in teal: "estimate of future rewards if we take $\mathbf{a}_ {i,t}$ in state $\mathbf{s}_ {i,t}$".

Then a greyed-out, struck-through copy of $\sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'})$, with "Can we get a better estimate?" to its right. Below it:

$$\sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right] = Q(\mathbf{s}_ t, \mathbf{a}_ t)$$

Under it, in italics: "true expected rewards to go"; to the right in teal: "This would be way better!"

Bottom equation:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \thinspace Q(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

## Slide 10 — Improving policy gradients

Subtitle: "What about baselines?"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( Q(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) - b \right)$$

The $b$ is struck through with a blue slash, and the blue text $V^\pi(\mathbf{s}_ t)$ is written just below it, replacing $b$.

Right:

$$b = \text{average reward} = \frac{1}{N} \sum_i Q(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

$$V(\mathbf{s}_ t) = E_{\mathbf{a}_ t \sim \pi_\theta(\cdot \mid \mathbf{s}_ t)} \left[ Q(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

"Recall:" $A^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) - V^\pi(\mathbf{s}_ t)$

"With baseline:" (in blue)

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \thinspace A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

with $A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$ in a pale teal highlight. Bottom, in teal: "Better estimates of $A$ lead to less noisy gradients!"

## Slide 11 — Online RL Outline

Top: the equation $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \thinspace A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$ with the $A^\pi$ term in a teal highlight.

"First: Initialize the policy" followed by "(randomly, with imitation learning, with heuristics)".

Diagram: a blue box "Run policy to collect batch of data" (upper left), a pink box "Improve policy" (lower left), joined by grey curved arrows in a loop (blue to pink down the right side, pink to blue up the left side). To the right sits a teal box "Fit model to estimate expected return" with no arrows attached yet.

## Slide 12 — Online RL Outline

![Slide 12 — Online RL Outline](../images/04-actor-critic/slide-12.png)

Same equation and "First: Initialize the policy (randomly, with imitation learning, with heuristics)" as slide 11. The diagram now has the teal box "Fit model to estimate expected return" inserted into the loop: a grey arrow from the blue "Run policy to collect batch of data" box curves down to the teal box; a grey arrow from the teal box curves down and back to the pink "Improve policy" box; the arrow from the pink box up to the blue box remains. Beside the teal-to-pink arrow, in teal: "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$". Under the pink box, in magenta: $\theta \leftarrow \theta + \nabla_\theta J(\theta)$.

## Slide 13 — The plan for today

Same outline as slide 5, with item 2 and its sub-items highlighted (item 2 in bold, 2a–2c in black) and everything else greyed out:

Actor critic methods
1. Improving policy gradients (grey)
2. **How to estimate the value of a policy**
   a. Sample & directly supervise
   b. Use your own estimate
   c. Something in-between?
3. Off-policy actor-critic (grey)
   a. Importance weights & constraining step size (grey)
   b. Full off-policy version with replay buffers (grey)

"Key learning goals:" (grey)
- How to estimate how good a state and action is for a policy
- How to use those estimates to form a **better** RL algorithm (the word "better" here; slide 5 prints "more efficient")

## Slide 14 — Estimating expected return

![Slide 14 — Estimating expected return](../images/04-actor-critic/slide-14.jpg)

Top: $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \thinspace A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$ with $A^\pi$ in a teal highlight.

In teal: "Should we fit $V^\pi$, $Q^\pi$, or $A^\pi$?"

Colour-coded derivation ($A^\pi$ in teal, $Q^\pi$ in orange, $V^\pi$ in pink highlights):

$$A^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) - V^\pi(\mathbf{s}_ t)$$

$$Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right]$$

$$= r(\mathbf{s}_ t, \mathbf{a}_ t) + \sum_{t'=t+1}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right]$$

$$= r(\mathbf{s}_ t, \mathbf{a}_ t) + E_{\mathbf{s}_ {t+1} \sim p(\cdot \mid \mathbf{s}_ t, \mathbf{a}_ t)} \left[ V^\pi(\mathbf{s}_ {t+1}) \right]$$

$$\approx r(\mathbf{s}_ t, \mathbf{a}_ t) + V^\pi(\mathbf{s}_ {t+1})$$

with "(use the sampled $\mathbf{s}_ {t+1}$ )" in grey beside the last line, and finally

$$A^\pi(\mathbf{s}_ t, \mathbf{a}_ t) \approx r(\mathbf{s}_ t, \mathbf{a}_ t) + V^\pi(\mathbf{s}_ {t+1}) - V^\pi(\mathbf{s}_ t)$$

with, in red-pink, "Let's just fit $V^\pi$!"

Right: a fully connected neural network diagram with an input column of nodes labelled "s" on the left, three further columns of nodes with dense connecting lines, and an output column labelled $\hat{V}^\pi(\mathbf{s})$ on the right; below it "parameters $\phi$". The output column is drawn with many nodes. In the lecture Finn says the network outputs "a single scalar value, not as pictured here" (≈34:00).

## Slide 15 — Estimating $V^\pi$

Subtitle: "Version 1: Monte Carlo estimation".

$$V^\pi(\mathbf{s}_ t) \approx \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$$

followed by "<- original single sample estimate".

Right: a single black wavy trajectory with a blue dot labelled "s" part way along it; the portion above the dot continues to an arrowhead at the upper right and is labelled "run $\pi$"; a thin red double-headed arrow alongside is labelled "expected sum of these rewards"; to the right $V^\pi(\mathbf{s})$ with s in blue.

## Slide 16 — Estimating $V^\pi$

![Slide 16 — Estimating V-pi](../images/04-actor-critic/slide-16.jpg)

Subtitle: "Version 1: Monte Carlo estimation".

$$V^\pi(\mathbf{s}_ t) \approx \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$$

"<- original single sample estimate"

$$V^\pi(\mathbf{s}_ t) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$$

"<- multi-sample estimate", and beneath it in grey "(but can't reset the world)".

Right: as on slide 15 but now four black wavy trajectory arrows fan out from the blue dot "s" (all beginning at the same state) towards the upper right, labelled "run $\pi$", with the red double-headed arrow "expected sum of these rewards" alongside and $V^\pi(\mathbf{s})$ (s in blue). The single trajectory below the dot comes in from the lower left.

## Slide 17 — Estimating $V^\pi$

![Slide 17 — Estimating V-pi](../images/04-actor-critic/slide-17.jpg)

Subtitle: "Version 1: Monte Carlo estimation". The two estimate equations and notes from slide 16 repeat: "<- original single sample estimate", "<- multi-sample estimate", "(but can't reset the world)".

"**Step 1:** Aggregate dataset of single sample estimates:"

$$\left\lbrace  \left( \mathbf{s}_ {i,t}, \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) \right\rbrace$$

with a black brace under the second element (the sum) labelled $y_{i,t}$.

"**Step 2:** Supervised learning to fit estimated value function"

$$\mathcal{L}(\phi) = \frac{1}{2} \sum_i \left\Vert \hat{V}^\pi_\phi(\mathbf{s}_ i) - y_i \right\Vert^2$$

Right: two black wavy trajectories, from different starting points, crossing each other; at the crossing, a blue circle outline encloses two blue dots (two visits to nearly the same state). Labelled "run $\pi$" and $V^\pi(\mathbf{s})$ (s in blue). Below, a fully connected neural network diagram with input "s" at its left and output $\hat{V}^\pi(\mathbf{s})$ and "parameters $\phi$" at its right.

## Slide 18 — Estimating $V^\pi$

![Slide 18 — Estimating V-pi](../images/04-actor-critic/slide-18.jpg)

Subtitle: "Version 2: Bootstrapping". Upper right: the neural network diagram, input "s", output $\hat{V}^\pi(\mathbf{s})$, "parameters $\phi$".

$$\text{Monte Carlo target: } y_{i,t} = \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'})$$

$$\text{ideal target: } y_{i,t} = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ {i,t} \right] \approx r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \sum_{t'=t+1}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ {i,t+1} \right]$$

$$\approx r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + V^\pi(\mathbf{s}_ {i,t+1}) \approx r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1})$$

A black brace under the final term $\hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1})$ is labelled "directly use previous fitted value function!".

"**Step 1:** Aggregate dataset of "bootstrapped" estimates:"

$$\text{training data: } \left\lbrace  \left( \mathbf{s}_ {i,t}, \thinspace r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1}) \right) \right\rbrace$$

with a brace under the second element labelled $y_{i,t}$, and to the right in italics "<- update labels every gradient update!"

"**Step 2:** Supervised learning to fit estimated value function"

$$\mathcal{L}(\phi) = \frac{1}{2} \sum_i \left\Vert \hat{V}^\pi_\phi(\mathbf{s}_ i) - y_i \right\Vert^2$$

Bottom right: "Also referred to as a form of **temporal difference learning**".

## Slide 19 — Estimating $V^\pi$: Monte Carlo vs. Bootstrap

![Slide 19 — Estimating V-pi: Monte Carlo vs. Bootstrap](../images/04-actor-critic/slide-19.jpg)

Two rows. "Monte Carlo": $y_{i,t} = \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'})$, italic note "supervise with roll-out's summed rewards". "Bootstrapped": $y_{i,t} = r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1})$, italic note "supervise using reward and current value estimate".

"Let's look at an example:" Diagram: two trajectories, one black and one grey, start from the same point at the bottom left. The black one runs up through a pink dot and on to a blue dot in the middle. The grey one swings out in a wide loop to the left, bypassing the pink dot, and joins the black one at the blue dot. Above the blue dot the paths split again: the black one curves up and left to a red dot labelled "r = -1" (in red), the grey one goes up and right to a green dot labelled "r = 1" (in green). So the blue state is visited by both trajectories and the pink state only by the black one, which ends at $-1$. Near the bottom, grey text "r = 0 along trajectory". Left, "run $\pi$".

A teal box headed "Think-pair-share:" holds a 2x2 grid:
- $\hat{V}^\pi_{MC}(\mathbf{s}) \approx ?$ (s in blue) and $\hat{V}^\pi_{MC}(\mathbf{s}) \approx ?$ (s in pink)
- $\hat{V}^\pi_{TD}(\mathbf{s}) \approx ?$ (s in blue) and $\hat{V}^\pi_{TD}(\mathbf{s}) \approx ?$ (s in pink)

The blue s refers to the blue dot and the pink s to the pink dot in the diagram.

Bottom right: "Is there middle ground? Can we balance bias and variance?"

## Slide 20 — Estimating $V^\pi$: Monte Carlo vs. Bootstrap

![Slide 20 — Estimating V-pi: Monte Carlo vs. Bootstrap](../images/04-actor-critic/slide-20.jpg)

Three rows. "Monte Carlo": $y_{i,t} = \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'})$, "supervise with roll-out's summed rewards". "Bootstrapped": $y_{i,t} = r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1})$, "supervise using reward and current value estimate". "N-step returns":

$$y_{i,t} = \sum_{t'=t}^{t+n-1} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) + \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+n})$$

(no discount factor is printed on this slide).

Diagram, lower left: a blue dot at the base of a fan of five black wavy trajectory arrows heading to the upper right. Three coloured marks: a short orange bar crossing the trajectories just above the dot, a blue bar crossing them higher up, and, running parallel to the trajectories, a long red double-ended bar labelled "sum rewards" (red text) a shorter purple bar beside it, and a very short magenta bar nearest the dot (the one reward of the bootstrapped target). Annotations: "Monte Carlo: using $\sum_t^T r$" (the sum in red) near the top of the red bar; "Bootstrapped: using $r + V$" (r in pink, V in orange) at the bottom left; "How about $\sum_t^{t+n-1} r + V_{t+n}$" (sum and r in purple, $V_{t+n}$ in blue) in the middle, followed by two italic notes: "-> less variance than MC" and "-> lower bias than 1-step bootstrap". Bottom right: $n \gt 1$, $n \lt T$, followed by "often works the best in practice!"

## Slide 21 — Aside: discount factors

![Slide 21 — Aside: discount factors](../images/04-actor-critic/slide-21.jpg)

(Number printed in the bottom-right corner.) Bottom-left credit: "Slide adapted from Sergey Levine".

$$y_{i,t} \approx r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1})$$

$$\mathcal{L}(\phi) = \frac{1}{2} \sum_i \left\Vert \hat{V}^\pi_\phi(\mathbf{s}_ i) - y_i \right\Vert^2$$

"what if $T$ (episode length) is $\infty$?" and $\hat{V}^\pi_\phi$ "can get infinitely large in many cases".

"simple trick: better to get rewards sooner than later"

$$y_{i,t} \approx r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) + \gamma \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+1})$$

An arrow points up from the text "discount factor $\gamma \in [0, 1]$ (0.99 works well)" to the $\gamma$.

Right: $\gamma$ "changes the MDP:" and a diagram of four circled states $\mathbf{s}_ 1$ (top left), $\mathbf{s}_ 2$ (top right), $\mathbf{s}_ 3$ (bottom left), $\mathbf{s}_ 4$ (bottom right), with double-headed arrows between $\mathbf{s}_ 1$ and $\mathbf{s}_ 2$, $\mathbf{s}_ 1$ and $\mathbf{s}_ 3$, $\mathbf{s}_ 2$ and $\mathbf{s}_ 4$, $\mathbf{s}_ 3$ and $\mathbf{s}_ 4$, and the two diagonals ($\mathbf{s}_ 1$ and $\mathbf{s}_ 4$, $\mathbf{s}_ 2$ and $\mathbf{s}_ 3$). A single-headed arrow leaves the vertical link between $\mathbf{s}_ 2$ and $\mathbf{s}_ 4$ and points right to a circle holding a black skull symbol (a terminal "death" state) to the right. Labels: beside the skull $\tilde{p}(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}) = (1 - \gamma)$, and under the four states $\tilde{p}(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}) = \gamma p(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$.

## Slide 22 — A Full Algorithm Walkthrough

![Slide 22 — A Full Algorithm Walkthrough](../images/04-actor-critic/slide-22.jpg)

Top right: the online RL cycle diagram: blue box "Run policy to collect batch of data" with a grey arrow to a teal box "Fit model to estimate expected return"; an arrow from the teal box to a pink box "Improve policy", and an arrow from the pink box back up to the blue box. Teal text "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$" beside the teal-to-pink arrow and magenta text $\theta \leftarrow \theta + \nabla_\theta J(\theta)$ under the pink box.

Numbered steps (step 1 blue, step 2 teal, steps 3 and 4 dark grey, step 5 magenta):

1. Sample batch of data $\lbrace (\mathbf{s}_ {1,i}, \mathbf{a}_ {1,i}, \ldots, \mathbf{s}_ {T,i}, \mathbf{a}_ {T,i})\rbrace$ from $\pi_\theta$
2. Fit $\hat{V}^{\pi_\theta}_ \phi$ to summed rewards in data
3. Evaluate $\hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) = r(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) + \gamma \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t+1,i}) - \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t,i}) \quad \forall t, i$
4. Evaluate $\nabla_\theta J(\theta) \approx \sum_{t,i} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {t,i} \mid \mathbf{s}_ {t,i}) \thinspace \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$
5. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

Bottom: two neural network diagrams side by side. The left has input "s" and output $\pi_\theta(\mathbf{a} \mid \mathbf{s})$ (underlined in magenta); the right has input "s" and output $\hat{V}^\pi(\mathbf{s})$ (underlined in teal). Two black arrows run from the bold text "actor-critic algorithm" (in quotation marks; "actor" underlined in magenta and "critic" underlined in teal) up to the two networks. To the right of the networks:

$$y_{i,t} = \sum_{t'=t}^{t+n-1} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) + \gamma^n \hat{V}^\pi_\phi(\mathbf{s}_ {i,t+n})$$

$$\mathcal{L}(\phi) = \frac{1}{2} \sum_i \left\Vert \hat{V}^\pi_\phi(\mathbf{s}_ i) - y_i \right\Vert^2$$

## Slide 23 — Review so far

![Slide 23 — Review so far](../images/04-actor-critic/slide-23.jpg)

"**Algorithms**"
- *Policy gradients*: observe what is good vs. bad, then do more of good stuff
- *Actor-critic*: learn to estimate what is good vs. bad, then do more of the good stuff

"i.e. get a better policy gradient by" (then in teal) "using neural network to estimate value function".

"**How to estimate value**:" "policy evaluation" (in quotation marks), then three lines:
- Supervised learning directly on observed sum of future rewards
- Supervised learning on current reward + value estimate of next state
- Hybrid: Supervised learning on sum of next $n$ rewards + value of state after that

Bottom: "What about off-policy versions?"

Right: the same small online RL cycle diagram as slide 22 (blue "Run policy to collect batch of data", teal "Fit model to estimate expected return", pink "Improve policy", arrows between them, "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$", and the update $\theta \leftarrow \theta + \nabla_\theta J(\theta)$ under the pink box).

## Slide 24 — The plan for today

Same outline as slide 5, with item 3 and its sub-items highlighted in black (item 3 in bold) and items 1 and 2, and the key learning goals, greyed out. The grey items now carry parenthetical names:

Actor critic methods
1. Improving policy gradients (grey)
2. How to estimate the value of a policy ("policy evaluation") (grey)
   a. Sample & directly supervise ("Monte Carlo estimation")
   b. Use your own estimate ("bootstrapping", "temporal difference learning")
   c. Something in-between? ("N-step returns")
3. **Off-policy actor-critic**
   a. Importance weights & constraining step size
   b. Full off-policy version with replay buffers

"Key learning goals:" (grey)
- How to estimate how good a state and action is for a policy
- How to use those estimates to form a better RL algorithm

## Slide 25 — Off-Policy Actor-Critic Methods

Subtitle: "**Version 1**: Multiple Gradient Steps".

1. Sample batch of data $\lbrace (\mathbf{s}_ {1,i}, \mathbf{a}_ {1,i}, \ldots, \mathbf{s}_ {T,i}, \mathbf{a}_ {T,i})\rbrace$ from $\pi_\theta$ (blue)
2. Fit $\hat{V}^{\pi_\theta}_ \phi$ to summed rewards in data (teal)
3. Evaluate $\hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) = r(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i}) + \gamma \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t+1,i}) - \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ {t,i}) \quad \forall t, i$ (grey)
4. Evaluate $\nabla_\theta J(\theta) \approx \sum_{t,i} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {t,i} \mid \mathbf{s}_ {t,i}) \thinspace \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$ (grey), with, to the right, "<- use importance weights here"
5. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ (magenta)

## Slide 26 — Off-Policy Actor-Critic Methods

Subtitle: "**Version 1**: Multiple Gradient Steps". Steps 1, 2, 3 and 5 as on slide 25. Step 4 now carries the importance weights:

4. Evaluate $\nabla_{\theta'} J(\theta') \approx \sum_{t,i} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {t,i} \mid \mathbf{s}_ {t,i}) \thinspace \hat{A}^{\pi_\theta}(\mathbf{s}_ {t,i}, \mathbf{a}_ {t,i})$, followed by "<- use importance weights here".

A hand-drawn black curved arrow at the left runs from step 5 up to step 4 (the update feeds back to the evaluation, i.e. step 5 then repeats step 4). A black arrow from the text "Policy will increase probability on actions with high advantages" points up to the importance-weight fraction. A black curved arrow from the text "Advantages based on old policy." points to the $\hat{A}^{\pi_\theta}$ term.

Bottom, bold: "What can go wrong if you take too many gradient steps?"

## Slide 27 — Off-Policy Actor-Critic Methods

![Slide 27 — Off-Policy Actor-Critic Methods](../images/04-actor-critic/slide-27.png)

Subtitle: "**Version 1**: Multiple Gradient Steps". The same five steps, the same two annotation arrows and the same question as slide 26, but without slide 26's small curved arrow from step 5 back to step 4; the annotation next to $\hat{A}$ now reads "Advantages are based on old policy, will get out-dated" (slide 26 prints "Advantages based on old policy.").

Added at the bottom, two ideas, each preceded by a light-bulb emoji.

"Idea 1: Use KL constraint on policy." Under it: "We will see this in LLM preference optimization" and

$$\mathbb{E}_ {\mathbf{s} \sim \pi_\theta} \left[ D_{KL}(\pi_{\theta'}(\cdot \mid \mathbf{s}) \thinspace\Vert\thinspace \pi_\theta(\cdot \mid \mathbf{s})) \right] \leq \delta$$

"Idea 2: Can we bound the importance weights?" Under it: "Doesn't directly constrain policy, but removes incentives" and "-> Key idea behind proximal policy optimization (PPO)".

## Slide 28 — Off-Policy Actor-Critic Methods

"**So far**:"
- use one batch of policy data for one gradient step (fully on-policy)
- use one batch of policy data for multiple gradient steps (starting to be off-policy)

Bold: "Can we be even more off-policy?" Then: "Can we reuse data from previous batches, i.e. all of the past trial-and-error data?"

"Key ideas:"
- maintain a buffer of all past data "*replay buffer*"
- adjust equations to remove on-policy assumptions

## Slide 29 — Off-Policy Actor-Critic Methods

![Slide 29 — Off-Policy Actor-Critic Methods](../images/04-actor-critic/slide-29.jpg)

(Number printed in the bottom-right corner.) Bottom-left credit: "Slide adapted from Sergey Levine".

Subtitle: "**Version 2**: Replay buffers".

"actor-critic algorithm:"
1. collect experience $\lbrace \mathbf{s}_ i, \mathbf{a}_ i\rbrace$ from $\pi_\theta(\mathbf{a} \mid \mathbf{s})$  — followed by a right brace and "Add this to replay buffer"
2. fit $\hat{V}^\pi_\phi(\mathbf{s})$ to sampled reward sums
3. evaluate $\hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i) = r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i) - \hat{V}^\pi_\phi(\mathbf{s}_ i)$
4. $\nabla_\theta J(\theta) \approx \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}_ i \mid \mathbf{s}_ i) \hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A right brace spanning steps 2–5 is labelled "Do this on minibatch sampled from **all previous data**". A thick light-green arrow runs down the left edge from step 5 back up to step 1 (the loop).

Bottom-left diagram headed "off-policy actor-critic": a vertical timeline (arrow pointing down, labelled $t$) with a column of boxes: blue tall box with "get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$", small green box with "update $\theta$", blue box "get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$", small green box "update $\theta$", then another blue box. To the right a grey cylinder labelled "replay buffer" with the caption "transitions that we saw in prior time steps"; two black arrows run from the cylinder to the two green "update $\theta$" boxes.

## Slide 30 — Off-Policy Actor-Critic Methods

![Slide 30 — Off-Policy Actor-Critic Methods](../images/04-actor-critic/slide-30.jpg)

(Number printed in the bottom-right corner.) Bottom-left credit: "Slide adapted from Sergey Levine".

Subtitle: "**Version 2**: Replay buffers".

"online actor-critic algorithm:"
1. collect experience $\lbrace \mathbf{s}_ i, \mathbf{a}_ i\rbrace$ from $\pi_\theta(\mathbf{a} \mid \mathbf{s})$ (& add to replay buffer)
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i\rbrace$ from buffer $\mathcal{R}$
3. update $\hat{V}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$ for each $\mathbf{s}_ i$
4. evaluate $\hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i) = r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i) - \hat{V}^\pi_\phi(\mathbf{s}_ i)$
5. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}_ i \mid \mathbf{s}_ i) \hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i)$
6. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A thick light-green arrow runs down the left edge from step 6 back up to step 1.

Hand-drawn red annotations: a red oval around the target $= r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$ and the word "for" in step 3, and a second red oval around $\nabla_\theta \log \pi_\theta(\mathbf{a}_ i \mid \mathbf{s}_ i)$ in step 5. A black arrow from the text "not the right target value" points up-left towards the $\hat{V}^\pi_\phi(\mathbf{s}'_ i)$ term in step 4, just below the first oval (it is unclear whether it points at step 3's oval or the step 4 term [unclear]). A black arrow from the text "not the action $\pi_\theta$ would have taken!" points up to the second red oval.

Upper right: a grey cylinder "replay buffer", and below it

$$\mathcal{L}(\phi) = \frac{1}{N} \sum_i \left\Vert \hat{V}^\pi_\phi(\mathbf{s}_ i) - y_i \right\Vert^2$$

with an arrow from the text "mini batch size" pointing up to the $N$.

Bottom: on a pink background, "The algorithm is currently broken" followed by a crying-face emoji.

## Slide 31 — Off-Policy Actor-Critic Methods

![Slide 31 — Off-Policy Actor-Critic Methods](../images/04-actor-critic/slide-31.jpg)

(Number printed in the bottom-right corner.) Bottom-right credit: "Slide adapted from Sergey Levine".

Subtitle: "**Version 2**: Fixing the value function".

"online actor-critic algorithm:"
1. take action $\mathbf{a} \sim \pi_\theta(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i\rbrace$ from buffer $\mathcal{R}$
3. update $\hat{V}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$ for each $\mathbf{s}_ i$
4. evaluate $\hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i) = r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i) - \hat{V}^\pi_\phi(\mathbf{s}_ i)$
5. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}_ i \mid \mathbf{s}_ i) \hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i)$
6. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A thick light-green arrow runs down the left edge from step 6 back up to step 1. A thick red hand-drawn oval circles the target $= r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$ and the word "for" in step 3. A black arrow from the underlined red text "not the right target value" points up to the end of step 4, near the $\hat{V}^\pi_\phi(\mathbf{s}'_ i)$ term below the oval.

Upper right, two equations. The first is struck through by a red line:

$$V^\pi(\mathbf{s}_ t) = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t \right]$$

The second is not struck through:

$$Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right]$$

and under it the quote: "total reward we get if we take $\mathbf{a}_ t$ in $\mathbf{s}_ t$... ... and then follow the policy $\pi$".

Lower half: "where does this come from?" with a black arrow pointing to the $\hat{V}^\pi_\phi(\mathbf{s}'_ i)$ in the new step 3 below:

3. update $\hat{Q}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{V}^\pi_\phi(\mathbf{s}'_ i)$ for each $\mathbf{s}_ i, \mathbf{a}_ i$

$$= r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$$

A small black arrow points up at the $\mathbf{a}'_ i$ from the bold text "**not** from replay buffer $\mathcal{R}$!", and under that $\mathbf{a}'_ i \sim \pi_\theta(\mathbf{a}'_ i \mid \mathbf{s}'_ i)$. To the right:

$$\mathcal{L}(\phi) = \frac{1}{N} \sum_i \left\Vert \hat{Q}^\pi_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - y_i \right\Vert^2$$

Bottom line:

$$V^\pi(\mathbf{s}_ t) = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t \right] = E_{\mathbf{a} \sim \pi(\mathbf{a}_ t \mid \mathbf{s}_ t)} \left[ Q(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

## Slide 32 — Off-Policy Actor-Critic Methods

![Slide 32 — Off-Policy Actor-Critic Methods](../images/04-actor-critic/slide-32.jpg)

(Number printed in the bottom-right corner.) Bottom-left credit: "Slide adapted from Sergey Levine".

Subtitle: "**Version 2**: Fixing the policy update".

"online actor-critic algorithm:"
1. take action $\mathbf{a} \sim \pi_\theta(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i\rbrace$ from buffer $\mathcal{R}$
3. update $\hat{Q}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$ for each $\mathbf{s}_ i, \mathbf{a}_ i$
4. evaluate $\hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i) = Q(\mathbf{s}_ i, \mathbf{a}_ i) - \hat{V}^\pi_\phi(\mathbf{s}_ i)$
5. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}_ i \mid \mathbf{s}_ i) \hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}_ i)$
6. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A thick light-green arrow runs down the left edge from step 6 back up to step 1. A thick red hand-drawn oval circles $\nabla_\theta \log \pi_\theta(\mathbf{a}_ i \mid \mathbf{s}_ i)$ in step 5, with a black arrow from the text "not the action $\pi_\theta$ would have taken!" pointing to it.

Below, in order:

"use the same trick, but this time for $\mathbf{a}_ i$ rather than $\mathbf{a}'_ i$!"

"sample $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \hat{A}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i)$$

A small black arrow points up at $\mathbf{a}^\pi_ i$ from the bold text "**not** from replay buffer $\mathcal{R}$!". Then:

"in practice:" $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i)$

A black arrow from the text "higher variance, but convenient / why is higher variance OK here?" points to the $\hat{Q}^\pi$ in this last equation.

## Slide 33 — Off-Policy Actor-Critic Methods

(Number printed in the bottom-right corner.) Bottom-left credit: "Slide adapted from Sergey Levine".

Subtitle: "**Version 2**: Anything else?".

"online actor-critic algorithm:"
1. take action $\mathbf{a} \sim \pi_\theta(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i\rbrace$ from buffer $\mathcal{R}$
3. update $\hat{Q}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$ for each $\mathbf{s}_ i, \mathbf{a}_ i$
4. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i)$ where $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A thick light-green arrow runs down the left edge from step 5 back up to step 1.

On a pink background: "Any remaining problems?". Then:
- $\mathbf{s}_ i$ "didn't come from $p_\theta(\mathbf{s})$" and "nothing we can do here, just accept it"
- "**intuition:** we want optimal policy on $p_\theta(\mathbf{s})$" and "but we get optimal policy on a *broader* distribution"

## Slide 34 — Off-Policy Actor-Critic Methods

(Number printed in the bottom-right corner.)

Subtitle: "**Version 2**: Some implementation details".

"online actor-critic algorithm:" with the same five steps as slide 33 (step 4 here prints $\hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i)$ with the hat on $Q$ unchanged, and the where-clause "where $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$"), and the green loop arrow from step 5 to step 1.

Annotations: a black arrow from the text "also fancier ways to fit Q-functions (more on this in next two lectures)" points to step 3. A black arrow from the text "can also use **reparameterization trick** to better estimate the gradient (for Gaussian policy)" points up to the $\nabla_\theta \log \pi_\theta$ term in step 4.

"Example practical algorithm:" then the citation "Haarnoja, Zhou, Abbeel, Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. 2018."

## Slide 35 — More off-policy vs. less off-policy actor critic?

![Slide 35 — More off-policy vs. less off-policy actor critic?](../images/04-actor-critic/slide-35.jpg)

Two line charts pasted from a paper, captioned "(e) Humanoid-v1" (left) and "(f) Humanoid (rllab)" (right). Each has x-axis "million steps" from 0 to 10 (ticks at 0, 2, 4, 6, 8, 10) and y-axis "average return"; both plot noisy learning curves with a lighter shaded band around each. The legend, printed inside the right chart, lists five series: SAC (orange), DDPG (green), PPO (brown), SQL (pink), TD3 (concurrent) (purple).

Left chart, "(e) Humanoid-v1": y-axis 0 to 8000 (ticks 0, 2000, 4000, 6000, 8000). Approximate values read from the curves:
- SAC (orange): jumps to about 4800 by roughly 0.5 million steps, about 5500 at 2 million, about 6000 at 5 million, about 6300 near 10 million.
- PPO (brown): rises slowly from about 400 at 0, about 1000 at 2 million, about 3000 at 4 million, about 6000 at 6 million (where it meets SAC), ending near 6500 at 10 million with a jagged curve; its brownish shaded band is the widest on the chart, reaching roughly 8000 at later steps.
- SQL (pink): rises to about 3000 by 1 million steps, about 5000 by 3 million, about 5500 from 6 to 9 million, ending a little lower at 10 million.
- DDPG (green): stays low, roughly 500 throughout.
- TD3 (concurrent) (purple): a flat line at about 0 that stops at about 4 million steps.

Right chart, "(f) Humanoid (rllab)": y-axis 0 to about 6000 (ticks 0, 2000, 4000, 6000). Approximate values:
- SAC (orange): near 0 at 0, about 2000 at 3 million, about 4000 at 5 million, about 5200 at 8 million, ending near 5500 at 10 million with a dip around 9.5 million.
- PPO (brown): rises roughly linearly, about 500 at 2 million, about 1200 at 5 million, ending about 2300 at 10 million.
- SQL (pink): close to the PPO curve, ending about 1700 at 10 million.
- TD3 (concurrent) (purple): flat at about 200, stopping at about 8.7 million steps.
- DDPG (green): flat near 0 throughout.

Right of the charts, in orange: "<- more off-policy" with "(i.e. with replay buffer)" beneath, aligned with the SAC curve. In brown: "<- less off-policy" with "(i.e. no replay buffer)", aligned with the PPO/SQL curves.

Bottom, in green: "+ Off-policy with replay buffer (e.g. soft actor-critic) can be far more data efficient". In red: "- They can also generally be harder to tune hyperparameters, less stable (than e.g. PPO)".

The slide gives no source line for the charts. The chart axis values above are visual estimates.

## Slide 36 — The plan for today

Same outline as slide 5, all items in black, with bold names in parentheses added:

Actor critic methods
1. Improving policy gradients
2. How to estimate the value of a policy ("**policy evaluation**")
   a. Sample & directly supervise ("**Monte Carlo estimation**")
   b. Use your own estimate ("**bootstrapping**", "**temporal difference learning**")
   c. Something in-between? ("**N-step returns**")
3. Off-policy actor-critic
   a. Importance weights & constraining step size
   b. Full off-policy version with replay buffers

"Key learning goals:"
- How to estimate how good a state and action is for a policy
- How to use those estimates to form a better RL algorithm

## Slide 37 — Next Week

"Q-learning" "(last big online RL method)", then "+", then "Practical implementation of online RL algorithms".

"Course reminders"
- Form **final project groups & ideas** (survey due next Wednesday)
- **Homework 1** due next Friday
