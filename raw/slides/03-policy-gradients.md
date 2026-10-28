---
title: Lecture 3 — Policy Gradients (slide deck)
lecture: 3
slides: 28
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–28). Every page prints its number.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (6, 11, 12, 16, 23, 24) were checked by Claude Opus 5.5 against 1400px renders, and the printed slips on slides 10, 11, 16, 17 and 25 against the page. All agreed on structure, marks, arrows, colour coding and equations; no corrections were needed.
images: 11 slides are rendered and shown under their headings (5, 6, 7, 10, 11, 12, 14, 17, 19, 24, 26). Not rendered — slide 27, an exact repeat of slide 5; slides 13, 15 and 16, whose only picture is a thumbnail of slide 12's humanoid beside equations transcribed here; slides 8, 9, 21 and 22, whose images are pasted equations transcribed here; and the title, reminder, text and equation slides (1–4, 18, 20, 23, 25, 28).
---

# Lecture 3 — Policy Gradients: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 3 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 28 pages. This was confirmed from the text layer's footer spans, including pages 3 and 23, where equation subscripts sit near the footer and confuse an automatic number reader; both pages print "3" and "23".

The recording covers slides 3–28. It opens on the recap (slide 3) without discussing slide 2's reminders, and it ends at about 1:02:26 on the next-lecture slide (28). The KL constraint on slide 25 is shown but skipped for time (≈1:01:40: "we don't have time to cover KL constraints today"), so for it this file is the only source.

Companion pages: [wiki page for this lecture](../../wiki/03-policy-gradients.md) · [transcript](../transcripts/03-policy-gradients.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–5 | Title, course reminders, recap, imitation learning and offline versus online, plan for today |
| 6–11 | Online RL outline, evaluating the RL objective, deriving the policy gradient, estimating it (REINFORCE), what it means |
| 12–19 | What the gradient does (humanoid example), causality, baselines, implementation with a surrogate objective, summary so far |
| 20–26 | On-policy versus off-policy, importance sampling, off-policy policy gradient, KL constraint, review |
| 27–28 | Plan recap, next time (actor critic methods) and course reminders |

## Slide 1 — Policy Gradients

Title slide. Title: "Policy Gradients". Below: "CS 224R". No images.

## Slide 2 — Course reminders

- Start forming **final project groups** (survey due next Wednesday)
- **Homework 1** out, due Fri April 18

## Slide 3 — Recap

- *state* $\mathbf{s}_ t$ - the state of the "world" at time $t$
- *action* $\mathbf{a}_ t$ - the decision taken at time $t$
- *trajectory* $\tau$ - sequence of states/observations and actions

$$(\mathbf{s}_ 1, \mathbf{a}_ 1, \mathbf{s}_ 2, \mathbf{a}_ 2, \ldots, \mathbf{s}_ T, \mathbf{a}_ T)$$

- *reward function* $r(\mathbf{s}, \mathbf{a})$ - how good is $\mathbf{s}, \mathbf{a}$?
- *policy* $\pi(\mathbf{a} \mid \mathbf{s})$ or $\pi(\mathbf{a} \mid \mathbf{o})$ - behavior, usually what we are trying to learn

**Goal**: learn policy $\pi_\theta$ that maximizes *expected* sum of rewards:

$$\max_\theta \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \sum_{t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

where

$$p(\mathbf{s}_ 1, \mathbf{a}_ 1, \ldots, \mathbf{s}_ T, \mathbf{a}_ T) = p(\mathbf{s}_ 1) \prod_{t} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \thinspace p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

(In the printed equation the sum over $t$ has limits $t$ below and $T$ above, and the product index $t$ sits below the product sign; the subscript $t+1$ in $\mathbf{s}_ {t+1}$ is set in bold.)

## Slide 4 — Imitation learning from demonstrations (untitled slide)

The slide has no title; its first line is "Imitation learning from demonstrations $\mathcal{D}$".

$$\min_\theta - \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}} \left[ \log \pi_\theta(\mathbf{a} \mid \mathbf{s}) \right]$$

- (green) "+ Simple, scalable approach for learning performant behavior"
- (red) "- Cannot outperform demonstrator, doesn't allow improvement from practice"

A pink box with a red outline, headed "Definitions." in red:
- *offline*: using only an existing dataset, no new data from learned policy
- *online*: using new data from learned policy

## Slide 5 — The plan for today

![Slide 5 — The plan for today](../images/03-policy-gradients/slide-5.jpg)

"**Policy gradients**: our first **online RL** algorithm"

1. On-policy policy gradient
   a. Derivation and intuition of policy gradients
   b. Full algorithm
   c. How to make it better - causality and baselines
2. Off-policy policy gradients
   a. Importance sampling
   b. KL constraints

"**Key learning goals**:"
- Key intuition behind policy gradients
- How to implement, when to use policy gradients

Right side, headed "the basis for:": a photograph of a white and silver humanoid robot with a glowing blue head light, standing in a bright indoor hall with glass walls, one leg slightly raised as if mid-step, captioned "Source: Unitree". Overlapping the photograph's lower right is a screenshot of a dark-themed chat interface showing Python code (using numpy) and explanatory text; the text is too small to read [unclear]. Beside the screenshot: "+ part of default project".

In the lecture (≈1:40–2:27) these stand for the algorithms built on policy gradients: ones used to train legged robots (the photograph), ones used to train language models (the chat screenshot), and the default final project.

## Slide 6 — Online RL Outline

![Slide 6 — Online RL Outline](../images/03-policy-gradients/slide-6.jpg)

"First: Initialize the policy" followed by "(randomly, with imitation learning, with heuristics)".

Left: a two-box cycle. A light blue rounded box "Run policy to collect batch of data" and a light pink rounded box "Improve policy using batch of data", joined by two grey curved arrows forming a loop (blue box to pink box down the right side, pink box back to blue box up the left side).

Right: three grey ovals labelled "iteration 1", "iteration 2", "iteration 3", each showing blue curved trajectory arrows that start together at the lower left and end towards the upper right, with marks at their ends. In iteration 1 the four trajectories are spread out and end at different marks: two orange X marks and two yellow check marks. In iteration 2 the four trajectories are closer together: one orange X, one yellow check and two green checks. In iteration 3 the four trajectories are tightly bunched and all end at green check marks.

## Slide 7 — Evaluating the RL objective

![Slide 7 — Evaluating the RL objective](../images/03-policy-gradients/slide-7.jpg)

$$\theta^{\ast} = \arg\max_\theta E_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

A brace under the expectation term (from $E$ through the closing bracket) is labelled $J(\theta)$.

$$J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(\mathbf{s}_ t, \mathbf{a}_ t) \right] \approx \frac{1}{N} \sum_i \sum_t r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

A thick black curved arrow points from the text "sum over samples from $\pi_\theta$" up to the $\sum_i$ in the approximation.

Upper right: a 3D gridded plot (axis tick labels too small to read [unclear]) with three black curved trajectory arrows starting from a common point at the lower left and heading to the upper right, with a green check mark, an orange check mark and a red X mark near the middle of the trajectories.

Footer, bottom left: "Slide adapted from Sergey Levine".

## Slide 8 — Can we get the gradient of the RL objective?

Subtitle (grey): "Let's start in terms of trajectories."

$$\theta^{\ast} = \arg\max_\theta E_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

with the expectation term braced and labelled $J(\theta)$ as on slide 7.

$$J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ r(\tau) \right] = \int p_\theta(\tau) \thinspace r(\tau) \thinspace d\tau$$

A small brace under $r(\tau)$ points to $\sum_{t=1}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$.

Grey rounded box headed "a convenient identity":

$$p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) = p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta p_\theta(\tau)$$

Colour coding: the first term, $p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$, is underlined in yellow; the last term, $\nabla_\theta p_\theta(\tau)$, is underlined in blue.

Bottom equation:

$$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) \thinspace r(\tau) \thinspace d\tau = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) \thinspace r(\tau) \thinspace d\tau = E_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace r(\tau) \right]$$

Here $\nabla_\theta p_\theta(\tau)$ in the first integral is underlined in blue and $p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$ in the second integral is underlined in yellow, matching the identity.

Footer: "Slide adapted from Sergey Levine".

## Slide 9 — Can we get the gradient of the RL objective?

Subtitle (grey): "From trajectories to final form."

Top left:

$$\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace r(\tau) \right]$$

with $\log p_\theta(\tau)$ underlined in green. A green brace under that underlined term opens down to the next line:

$$\nabla_\theta \left[ \log p(\mathbf{s}_ 1) + \sum_{t=1}^{T} \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) + \log p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t) \right]$$

In this line the first term $\log p(\mathbf{s}_ 1)$ and the last term $\log p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$ are each crossed out with a red line; the middle sum is not.

Top right:

$$p_\theta(\tau) = p(\mathbf{s}_ 1) \prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \thinspace p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

$$\log p_\theta(\tau) = \log p(\mathbf{s}_ 1) + \sum_{t=1}^{T} \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) + \log p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$$

with the left-hand side, $\log p_\theta(\tau) =$, underlined in green.

Bottom (the resulting gradient):

$$\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right) \right]$$

Footer: "Slide adapted from Sergey Levine".

## Slide 10 — Estimating the gradient

![Slide 10 — Estimating the gradient](../images/03-policy-gradients/slide-10.jpg)

$$\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right) \right]$$

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) \right)$$

with $\frac{1}{N} \sum_{i=1}^{N}$ underlined in blue (the colour of the "Run policy to collect batch of data" box).

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

underlined in magenta/pink (the colour of the "Improve policy using batch of data" box).

"Full algorithm:"
1. sample $\lbrace \tau^i \rbrace$ from $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$
2. $\nabla_\theta J(\theta) \approx \sum_i \left( \sum_t \nabla_\theta \log \pi_\theta(\mathbf{a}_ t^i \mid \mathbf{s}_ t^i) \right) \left( \sum_t r(\mathbf{s}_ t^i, \mathbf{a}_ t^i) \right)$
3. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A curved grey arrow runs from step 3 back up to step 1, showing the loop.

Below: "REINFORCE algorithm", vanilla policy gradient" (printed as "“REINFORCE algorithm”, vanilla policy gradient").

Right: the 3D trajectory plot from slide 7 (three black trajectories with a green check, an orange check and a red X), above the same two-box cycle as slide 6: blue box "Run policy to collect batch of data" and pink box "Improve policy using batch of data" joined by grey curved arrows. The colour of the underlines on the left ties the blue underline to the first box and the pink underline to the second.

Footer: "Slide adapted from Sergey Levine".

## Slide 11 — What does the gradient mean?

![Slide 11 — What does the gradient mean?](../images/03-policy-gradients/slide-11.jpg)

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) \right)$$

The part $\frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right)$ is shaded in light teal. Below it: "Imitation gradient, but weighted by reward".

Right, in a light yellow rounded box headed "**Recall**: imitation learning":

$$\min_\theta - \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim \mathcal{D}} \left[ \log \pi_\theta(\mathbf{a} \mid \mathbf{s}) \right]$$

$$\nabla_\theta J_{\text{BC}}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t-1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right)$$

with the right-hand side shaded in the same light teal. (The printed lower sum limit reads $t-1$, with a minus sign, where the other slides have $t=1$.)

"**Intuition**:"
- (green text) "Increase likelihood" of actions you took in high reward trajectories.
- (red text) "Decrease likelihood" of actions you took in negative reward trajectories

"i.e. do more of the good stuff, less of the bad stuff."

"formalization of "trial-and-error""

Bottom right: a 3D surface plot, coloured as a heat map from blue (low) to red (high), with a ridge rising towards the upper right; the three black trajectory arrows from slide 7 lie on the surface, with a green check, an orange check and a red X (axis tick labels too small to read [unclear]).

## Slide 12 — What does the gradient do?

![Slide 12 — What does the gradient do?](../images/03-policy-gradients/slide-12.jpg)

"**Example**: learning humanoid walking in simulation"

Left: a still image of a black simulated humanoid figure with a small head, mid-stride on a white floor with a grey grid, casting a shadow (a single frame; the slide gives no sign it is a video).

"reward: $r(\mathbf{s}, \mathbf{a})$ = forward velocity of robot" and, right-aligned in italics, "(can be negative if robot goes backwards)".

1. sample $\lbrace \tau^i \rbrace$ from $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$

Five sampled trajectories, each with a mark:
- $\tau^1$: falls backwards (orange X)
- $\tau^2$: falls forwards (green check)
- $\tau^3$: manages to stand still (yellow check)
- $\tau^4$: one small step forward then falls backwards (orange X)
- $\tau^5$: one large step backwards then small step forwards (orange X)

2. $\nabla_\theta J(\theta) \approx \sum_i \left( \sum_t \nabla_\theta \log \pi_\theta(\mathbf{a}_ t^i \mid \mathbf{s}_ t^i) \right) \left( \sum_t r(\mathbf{s}_ t^i, \mathbf{a}_ t^i) \right)$

(teal) "Question: what will the gradient encourage the policy to do?"

"-> will encourage policy to fall forward, and not take step forward" followed by a wide-eyed fearful-face emoji.

Bottom right: "Policy gradient is **noisy / high-variance**" ("noisy / high-variance" in red).

## Slide 13 — Improving the gradient

Subtitle (grey): "Using causality"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right) \left( \sum_{t=1}^{T} r(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t}) \right)$$

Upper right: a small still of the humanoid from slide 12 with the label $\tau^5$ and the text "one large step backwards then small step forwards" and an orange X.

"Policy behavior at time $t$ does not affect rewards at time $t' \lt t$"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right)$$

A blue circle is drawn around the $=t$ in the lower limit $t'=t$ of the reward sum. Below, in blue: "sum of *future* rewards".

In the lecture Finn notes that this equation, unlike slide 18's, has the reward sum correctly inside the sum over $t$ (≈33:48).

## Slide 14 — What does the gradient do?

![Slide 14 — What does the gradient do?](../images/03-policy-gradients/slide-14.jpg)

"**Example**: learning humanoid walking in simulation"

Same humanoid still as slide 12.

"reward: $r(\mathbf{s}, \mathbf{a})$ = forward velocity of robot" and "(can be negative if robot goes backwards)".

1. sample $\lbrace \tau^i \rbrace$ from $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$

Four sampled trajectories, each marked with a check mark whose colour goes from yellow to dark green:
- $\tau^1$: falls forwards (yellow check)
- $\tau^2$: slowly stumbles forwards (light green check)
- $\tau^3$: steadily walks forwards (green check)
- $\tau^4$: runs forwards (dark green check)

2. $\nabla_\theta J(\theta) \approx \sum_i \left( \sum_t \nabla_\theta \log \pi_\theta(\mathbf{a}_ t^i \mid \mathbf{s}_ t^i) \right) \left( \sum_t r(\mathbf{s}_ t^i, \mathbf{a}_ t^i) \right)$

(teal) "Question: what will the gradient encourage the policy to do?"

"-> encourages policy to fall, stumble forward some of the time" followed by a wide-eyed fearful-face emoji.

Bottom right: "Policy gradient is **noisy / high-variance**" and "**sensitive to reward scale**" (both bold phrases in red).

## Slide 15 — Improving the gradient

Subtitle (grey): "Introducing baselines"

$$\nabla_\theta J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace r(\tau) \right]$$

Upper right: a small still of the humanoid from slide 12 with two labelled trajectories: $\tau^1$ "falls forwards" (yellow check) and $\tau^4$ "runs forwards" (dark green check). The rest of the slide is empty (the first step of a build continued on slide 16).

## Slide 16 — Improving the gradient

Subtitle (grey): "Introducing baselines"

Top right, grey rounded box headed "a convenient identity":

$$p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) = \nabla_\theta p_\theta(\tau)$$

$$\nabla_\theta J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) \thinspace (r(\tau) - b) \right]$$

with the $- b$ term shaded in light teal.

$$b = \frac{1}{N} \sum_{i=1}^{N} r(\tau)$$

Next to it, "subtracting a **constant**" ("constant" in teal) and, below, "baseline" in quotes (grey).

Right: the same humanoid still and the two labelled trajectories as on slide 15 ($\tau^1$ falls forwards, yellow check; $\tau^4$ runs forwards, dark green check).

Grey text: "If we subtract average reward, we get negative gradients for below-average behavior." followed by a party-face emoji. "But, can we even do that?" followed by a thinking-face emoji.

$$E\left[ \nabla_\theta \log p_\theta(\tau) b \right] = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) \thinspace b \thinspace d\tau = \int \nabla_\theta p_\theta(\tau) \thinspace b \thinspace d\tau = b \nabla_\theta \int p_\theta(\tau) \thinspace d\tau = b \nabla_\theta 1 = 0$$

"Subtracting a constant baseline does not change the gradient in expectation. It is **unbiased**." "and can reduce **variance** of the gradient"

"Average reward is a pretty good baseline."

## Slide 17 — What does the gradient do? (baseline edition)

![Slide 17 — What does the gradient do? (baseline edition)](../images/03-policy-gradients/slide-17.jpg)

Title: "What does the gradient do?" with "(baseline edition)" beside it.

"**Example**: learning to fold a jacket"

Left: a photograph of a grey humanoid robot with a dark head screen showing two blue oval eyes, torso on a dark column, two grey arms extended out to the sides and down; its left hand (image left) holds the corner of a grey jacket that lies on a white table. A white shelving unit with boxes is behind it.

Reward, as a brace with three cases: "reward: $r(\mathbf{s}, \mathbf{a}) =$" 1 "neatly folded"; 0.5 "folded with some wrinkles"; 0 "not folded".

1. sample $\lbrace \tau^i \rbrace$ from $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$

Four sampled trajectories (no marks on this slide):
- $\tau^1$: doesn't touch the jacket
- $\tau^2$: folds only the sleeves
- $\tau^3$: flattens the jacket but does not fold it
- $\tau^4$: folds the jacket

2. $\nabla_\theta J(\theta) \approx \sum_i \left( \sum_t \nabla_\theta \log \pi_\theta(\mathbf{a}_ t^i \mid \mathbf{s}_ t^i) \right) \left( \sum_t r(\mathbf{s}_ t^i, \mathbf{a}_ t^i) - b \right)$

(teal) "Question: what will the gradient encourage the policy to do?"

"-> will encourage folding, but gradient is constant for all but one trajectory" followed by a wide-eyed fearful-face emoji.

"Policy gradient is still **noisy / high-variance**" ("noisy / high-variance" in red).
"Best with **dense rewards, large batches**."

## Slide 18 — How to implement policy gradients?

"Our gradient"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

A curved black arrow points to the $\nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$ term from the text "Computing these individually is inefficient." and, below, "(N*T backwards passes)".

"Can we use automatic differentiation on full objective?"

"Implement "surrogate objective" whose gradient is the same as $\nabla J$" (printed with a bold $\nabla J$).

$$\tilde{J}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

The term $\log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$ is shaded light blue, and a blue arrow points up to it from the blue text "Cross-entropy for discrete action policy, squared error for Gaussian policy". To the right of the equation, in italics: "weighted maximum likelihood".

**Printed slip, acknowledged in the lecture.** In both equations on this slide the reward sum $\sum_{t'=t}^{T}$ refers to $t$ but sits outside the sum over $t$. Finn says "this second sum should be inside the first sum" (≈53:11), as on slide 13. The first equation on slides 20, 22 and 23 is printed with the same layout.

## Slide 19 — Summary so far

![Slide 19 — Summary so far](../images/03-policy-gradients/slide-19.jpg)

"**Estimating gradient of RL objective**"
- log gradient trick
- weigh policy likelihood by future rewards
- subtract baseline (e.g. average reward)
- even with tricks, gradient is (red) "noisy"

"**First reinforcement learning algorithm**"
- (blue) "collect batch of data", (pink) "improve policy by applying gradient"
- formalizes trial-and-error learning

"**Key intuition**: do more high reward stuff, less low reward stuff"

Right: the two-box cycle diagram as on slide 6, blue box "Run policy to collect batch of data" and pink box "Improve policy using batch of data" joined by grey curved arrows in a loop.

## Slide 20 — What else is troublesome about policy gradients?

"Latest version of our gradient:"

$$\nabla_\theta J(\theta) = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \right) - b \right) \right]$$

The subscript $p_\theta(\tau)$ of the expectation is shaded light blue. Beneath it, in blue: "assumes samples from current policy $\pi_\theta$".

"Full algorithm:"
1. sample $\lbrace \tau^i \rbrace$ from $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$
2. compute $\nabla_\theta J(\theta)$ using $\lbrace \tau^i \rbrace$
3. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ followed by blue text "<- we change $\theta$ right here"

A curved grey arrow runs from step 3 back up to step 1.

"Need to recollect data every gradient step!" followed by a screaming-face emoji.
"Vanilla policy gradient is **on-policy**."

Right: coral text "attribute of online RL algorithms" above a pink box with a coral outline headed "Definitions.":
- *on-policy*: update uses only data from current policy
- *off-policy*: update can reuse data from other, past policies

## Slide 21 — Off-policy version of policy gradient?

Subtitle (grey): "Importance sampling"

$$J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ r(\tau) \right]$$

"What if we want to use samples from $\bar{p}(\tau)$?" and, grey, right-aligned: "(e.g. previous policy)"

$$J(\theta) = E_{\tau \sim \bar{p}(\tau)} \left[ \frac{p_\theta(\tau)}{\bar{p}(\tau)} r(\tau) \right]$$

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \frac{p(\mathbf{s}_ 1) \prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \thinspace p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)}{p(\mathbf{s}_ 1) \prod_{t=1}^{T} \bar{\pi}(\mathbf{a}_ t \mid \mathbf{s}_ t) \thinspace p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)} = \frac{\prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)}{\prod_{t=1}^{T} \bar{\pi}(\mathbf{a}_ t \mid \mathbf{s}_ t)}$$

In the middle fraction the $p(\mathbf{s}_ 1)$ and the $p(\mathbf{s}_ {t+1} \mid \mathbf{s}_ t, \mathbf{a}_ t)$ factors in both numerator and denominator are crossed out with red lines.

Right, in a pink rounded box headed (coral) "Importance sampling", subtitle (grey) "Using proposal distribution $q$":

$$E_{x \sim p(x)} \left[ f(x) \right] = \int p(x) f(x) \thinspace dx$$
$$= \int \frac{q(x)}{q(x)} p(x) f(x) \thinspace dx$$
$$= \int q(x) \frac{p(x)}{q(x)} f(x) \thinspace dx$$
$$= E_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]$$

"**Note**: Important for $q$ to have non-zero support for high probability $p(x)$"

## Slide 22 — Off-policy version of policy gradient?

Subtitle (grey): "Importance sampling"

Top right, light green rounded box:

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \frac{\prod_{t=1}^{T} \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)}{\prod_{t=1}^{T} \bar{\pi}(\mathbf{a}_ t \mid \mathbf{s}_ t)}$$

"Say we want to update our latest policy $\pi_{\theta'}$ but we want to use samples from $\pi_\theta$"

Three equations in sequence:

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_ {\tau \sim p_{\theta'}(\tau)} \left[ \left( \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \right) - b \right) \right]$$

with the subscript $p_{\theta'}(\tau)$ shaded light blue;

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \frac{p_{\theta'}(\tau)}{p_\theta(\tau)} \left( \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \right) - b \right) \right]$$

with the subscript $p_\theta(\tau)$ and the ratio $\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}$ shaded light green;

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \prod_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ t \mid \mathbf{s}_ t)}{\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)} \left( \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \right) - b \right) \right]$$

with the product of ratios shaded green. A dark green curved arrow points to the product from the text "This can become very small or very large, for larger $T$".

## Slide 23 — Off-policy version of policy gradient?

Subtitle (grey): "Importance sampling"

"Say we want to update our latest policy $\pi_{\theta'}$ but we want to use samples from $\pi_\theta$"

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_ {\tau \sim p_\theta(\tau)} \left[ \prod_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ t \mid \mathbf{s}_ t)}{\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)} \left( \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ t \mid \mathbf{s}_ t) \right) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \right) - b \right) \right]$$

with the product of ratios shaded green and a dark green arrow from "This can become very small or very large, for larger $T$" pointing to it (same as slide 22).

"What if we consider the expectation *over timesteps* instead of trajectories?"

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})}{\pi_\theta(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

with the ratio $\frac{\pi_{\theta'}(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})}{\pi_\theta(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})}$ shaded green. A dark green arrow points to it from the text "Much less likely to explode/vanish ...but, hard to measure" (printed "...but, hard to measure"). To the right of that text, a small fraction $\frac{\pi_{\theta'}(\mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{s}_ {i,t})}$ followed by the italic text "often approximated as 1".

The ratio in the timestep version is written over state-action pairs, $\pi_{\theta'}(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$, deliberately: in the lecture Finn says the timestep form "needs to be over state action pairs rather than actions given states", and that the remaining ratio can be approximated as 1, which is incorrect but "the best that we can do" (≈1:00:54).

Bottom, in a light yellow rounded box headed "Common final form":

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

## Slide 24 — Off-policy policy gradient

![Slide 24 — Off-policy policy gradient](../images/03-policy-gradients/slide-24.jpg)

In a light yellow rounded box, "Common final form":

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

"Full algorithm:"
1. sample $\lbrace \tau^i \rbrace$ from $\pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t)$
2. compute $\nabla_\theta J(\theta)$ using $\lbrace \tau^i \rbrace$
3. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A grey curved arrow runs from step 3 back to step 1, and a second, orange curved arrow runs from step 3 back to step 2. Beside it: "Can take (orange) multiple gradient steps on the same batch".

Right: the two-box cycle diagram (blue "Run policy to collect batch of data", pink "Improve policy using batch of data", grey curved arrows).

"What if our policy changes **a lot** before sampling new data?"
"Data no longer reflects states that policy will visit. Gradient estimate less accurate."

## Slide 25 — Off-policy policy gradient

In a light yellow rounded box, "Common final form": the same equation as slide 24.

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})}{\pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})} \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

"What if our policy changes **a lot** before sampling new data?"
"Data no longer reflects states that policy will visit. Gradient estimate less accurate."

"**Can we constrain the policy to not stray too far during gradient updates?**" (the question is in a heavier weight than the surrounding text)

"One common choice:"

$$\mathbb{E}_ {\mathbf{s} \sim \pi_\theta} \left[ D_{KL}\left( \pi_{\theta'}(\cdot \mid \mathbf{s}) \Vert \pi_\theta(\cdot \mid \mathbf{s}) \right) \right] \leq \delta$$

The lecture does not discuss this constraint: "we don't have time to cover KL constraints today" (≈1:01:40).

## Slide 26 — Review

![Slide 26 — Review](../images/03-policy-gradients/slide-26.png)

"**Online RL via policy gradients**"
- (coral) "On-policy algorithm", differentiating the RL objective
- Baselines, causality for reducing gradient variance
- (blue) "collect batch of data", (pink) "improve policy by applying gradient"

"**Derived** (coral) off-policy **policy gradient**"
- importance sampling
- KL constraint on policy
- (blue) "collect batch of data", (pink) "apply multiple gradient updates"

"**Intuition**"
- Do more high reward stuff, less low reward stuff
- Gradient still very noisy, best with large batch sizes and dense rewards

Right: the two-box cycle diagram (blue "Run policy to collect batch of data", pink "Improve policy using batch of data", grey curved arrows).

## Slide 27 — The plan for today

Repeat of slide 5, unchanged: "**Policy gradients**: our first **online RL** algorithm", the same outline (1. On-policy policy gradient with a. Derivation and intuition of policy gradients, b. Full algorithm, c. How to make it better - causality and baselines; 2. Off-policy policy gradients with a. Importance sampling, b. KL constraints), "**Key learning goals**:" (Key intuition behind policy gradients; How to implement, when to use policy gradients), and on the right "the basis for:", the Unitree humanoid photograph captioned "Source: Unitree", the overlapping code-chat screenshot (too small to read [unclear]) and "+ part of default project".

## Slide 28 — Next time

*Actor critic methods*
- —> build closely on policy gradients!
- —> basis for popular algorithms like PPO

Second heading on the same slide, "Course reminders":
- Start forming **final project groups** (survey due next Wednesday)
- **Homework 1** out, due next Friday
