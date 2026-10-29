---
title: Lecture 7 — Offline RL (slide deck)
lecture: 7
slides: 33
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/07_cs224r_offline_rl_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–33). Page 28 prints no number; its label is inferred from position. Page 5 prints its number in the bottom-right corner.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (11, 14, 16, 21, 22, 31) were checked by Claude Opus 5.5 against 1400px renders. Slides 11, 22 and 31 agree, including all ten loss-curve endpoints of slide 22's left chart, the five fitted curves of its right chart, and every cell of slide 31's table. Corrections were small. On slides 14 and 16, $s_7$ is drawn beside $s_3$ rather than on it. On slide 16, an interpretation of the advantage-weight arrows was removed, and the advantage box's phrase "how much better $\mathbf{a}_ t$ is" is printed complete, not cut off. On slide 21, the second dashed line runs through the last bar, and the underline sits beneath $Q(\mathbf{s}', \mathbf{a}')$ only.
images: 14 slides are rendered and shown under their headings (3, 4, 5, 11, 14, 16, 17, 19, 21, 22, 23, 25, 26, 31). Not rendered — slides 8, 10, 15, 18, 20, 27 and 28, equation slides whose underlines, labels, strikethrough, loop arrows and circles are transcribed here; and the title, reminder, plan, text and summary slides (1, 2, 6, 7, 9, 12, 13, 24, 29, 30, 32, 33).
---

# Lecture 7 — Offline RL: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 7 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 33 pages. Page 28 prints no number at all; its label is inferred from its position between the numbered neighbours. Page 5 prints its number in the bottom-right corner rather than at the bottom centre; that number is real.

The recording covers slides 3–23 and 30–32. It opens on the recap (slide 3) without discussing slide 2's reminders. **The conservative-methods section, slides 24–28 (pushing down on large Q-values, conservative Q-learning (CQL), its full algorithm and how its objective is computed), is not taught in the recording**: after implicit Q-learning (slide 23, ≈1:05:10) the lecture goes straight to the summary (slide 30, ≈1:05:55), and the LinkedIn application on slide 31, which uses CQL, is passed over as an application "for the algorithm we didn't cover" (≈1:06:42). The recording ends at about 1:07:51 on slide 32's pointer to IDQL; slide 33's next-time line and reminders are not discussed. Much of the lecture is worked live, with the speaker writing or drawing as they talk ("maybe I'll do this in a new color", "I guess I drew it slightly wrong there"): the recap's value-function targets, the formal setting, a randomly initialized Q-function over a car's actions, the stitching example, the filtered and advantage-weighted objectives, AWR's advantage estimates, and the histogram and asymmetric loss behind expectile regression. This KB has that work only as the transcript describes it.

Companion pages: [wiki page for this lecture](../../wiki/07-offline-rl.md) · [transcript](../transcripts/07-offline-rl.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title, course reminders |
| 3–5 | Recap: policy-gradient and actor-critic methods, value and Q estimation, the full off-policy actor-critic method |
| 6 | Plan for today |
| 7–9 | Why offline RL: online versus offline process, the formal setting with a behavior policy, where the data comes from |
| 10–12 | Can we just use off-policy algorithms? Out-of-distribution actions and Q overestimation |
| 13–15 | Plan, offline RL versus imitation learning (stitching), filtered behavior cloning |
| 16–19 | Weighted imitation with advantage weights, advantage-weighted regression |
| 20–23 | Implicit policy constraints: AWAC, expectile regression, implicit Q-learning |
| 24–28 | Plan, conservative methods: Q-value penalty, conservative Q-learning (CQL) and its computation |
| 29–30 | Plan, summary |
| 31–32 | Example application (LinkedIn notifications), which offline RL algorithm to use |
| 33 | Next time, course reminders |

## Slide 1 — Offline Reinforcement Learning

Title slide. Title: "Offline Reinforcement Learning". Below: "CS 224R". No images.

## Slide 2 — Course reminders

Two columns.

Left, heading "Project":

- CA mentors assigned
- Fill out AWS form for GPU quota
- Proposal due Friday

Under the third bullet, in magenta: "graded fairly lightly- it's for your benefit!"

Right, heading "Homework":

- Homework 2 due next Friday. (start early!!)

## Slide 3 — Recap: Methods

![Slide 3 — Recap: Methods](../images/07-offline-rl/slide-3.jpg)

Left: the online RL cycle diagram. A blue box "Run policy to collect batch of data" (upper left) and a pink box "Improve policy" (lower left) are drawn at full strength; a middle box "Fit model to estimate expected return" and its label "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$" are drawn faded (nearly invisible). Faded grey arrows run from the blue box to the faded box and from it to the pink box; a solid dark-grey arrow runs from the pink box up the left side to the blue box. Under the pink box, in magenta: $\theta \leftarrow \theta + \nabla_\theta J(\theta)$.

Right, heading: "Online RL with policy gradients"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

The parts are colour-highlighted and labelled underneath: the two sums in a blue highlight labelled "samples from policy"; $\log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$ (with the $\nabla_\theta$ outside it) in a pink highlight labelled "policy log likelihood"; the inner reward sum in a green highlight labelled "reward to go"; and $-b$ in a teal highlight labelled "baseline".

Below it: "Do more of the" "above average" (green) "stuff, less of the" "below average" (red) "stuff."

Heading: "Online RL with actor-critic"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

The two sums are in a blue highlight and the $A^\pi$ term in a teal highlight.

"Estimate what is good and bad" (teal), ", then" (black) "do more of the good stuff." (magenta).

## Slide 4 — Recap: Methods

![Slide 4 — Recap: Methods](../images/07-offline-rl/slide-4.jpg)

Left: the online RL cycle diagram, now all at full strength. Blue box "Run policy to collect batch of data" (upper left), teal box "Fit model to estimate expected return" (middle right), pink box "Improve policy" (lower left). Grey arrows run blue to teal, teal to pink (labelled in teal "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$") and pink up to blue. Under the pink box, in magenta: $\theta \leftarrow \theta + \nabla_\theta J(\theta)$.

Below it, heading "Online RL with actor-critic" and the equation

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

(the two sums in a blue highlight, the $A^\pi$ term in a teal highlight), then "Estimate what is good and bad" (teal) ", then" "do more of the good stuff." (magenta).

Right, three numbered items, each with $V^\pi$ or $Q^\pi$ in teal:

1. Estimate $V^\pi$ with Monte Carlo

$$\min_\phi \sum_{\mathbf{s}_ t \sim \mathcal{D}} \left\lVert \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}_ t) - \sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right\rVert^2$$

(Printed slip, as on slides 4, 18 and 19: the sum runs over $t'$ but the reward inside it is written $r(\mathbf{s}_ t, \mathbf{a}_ t)$ rather than $r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$. The lecture describes this target as the sum of rewards from that state onward (≈1:38).)

2. Estimate $V^\pi$ with bootstrapped / TD updates

$$\min_\phi \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\lVert \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \hat{V}^{\pi_\theta}_ \phi(\mathbf{s}') \right) \right\rVert^2$$

3. Estimate $Q^\pi$ with bootstrapped / TD updates

$$\min_\phi \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\lVert \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}', \mathbf{a}') \right] \right) \right\rVert^2$$

## Slide 5 — Recap: Full Off-Policy Actor-Critic Method

![Slide 5 — Recap: Full Off-Policy Actor-Critic Method](../images/07-offline-rl/slide-5.jpg)

"online actor-critic algorithm:" (in a serif LaTeX-style font)

1. take action $\mathbf{a} \sim \pi_\theta(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i \rbrace$ from buffer $\mathcal{R}$
3. update $\hat{Q}^\pi_ \phi$ using targets $y_i = r_i + \gamma \hat{Q}^\pi_ \phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$ where $\mathbf{a}'_ i \sim \pi_\theta(\cdot \mid \mathbf{s}'_ i)$
4. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i)$ where $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

A thick pale-green arrow runs down the left side from the end of step 5 back to the start of step 1, marking the loop.

## Slide 6 — The plan for today

Heading "Offline RL", numbered:

1. Why offline RL? Can we just run off-policy methods?
2. Implicit policy constraint methods
3. Conservative methods

A black curly brace at the right, beside items 2 and 3, is labelled "Part of homework 3!"

"Key learning goals:"

- the **key challenges** (red) arising in offline reinforcement learning
- two approaches for offline RL (& why they work!)
- how offline RL (blue) can improve over imitation learning (magenta)

## Slide 7 — Why offline RL?

Two columns.

Left, "**Online RL** (teal) process (on-policy or off-policy)" with "on-policy" and "off-policy" in green:

- Collect data
- Update policy on latest data or data so far (the phrases "latest data" and "data so far" in green)

A black curved arrow at the left joins the two lines into a loop.

Right, "**Offline RL** (blue) process":

- Given static dataset
- Train policy on provided dataset

Centre: "Why, or when, might offline RL be more useful?" (offline RL in blue)

- leverage datasets collected by people, existing systems
- online policy collection may be risky, unsafe
- reuse previously collected data rather than recollecting (e.g. previous experiments, projects, robots, institutions)

Bottom: "**Note**: A blend of offline (blue) then online RL (green) is also possible!"

## Slide 8 — Why offline RL?

"**Offline RL** process" (blue):

- Given static dataset
- Train policy on provided dataset

"More formally:"

"Offline dataset $\mathcal{D} : \lbrace (\mathbf{s}, \mathbf{a}, \mathbf{s}', r) \rbrace$ sampled from some *unknown policy* $\pi_\beta$", with the label "“behavior policy”" under $\pi_\beta$.

$$\mathbf{s} \sim p_{\pi_\beta}(\cdot)$$

$$\mathbf{a} \sim \pi_\beta(\cdot \mid \mathbf{s})$$

$$\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a})$$

$$r = r(\mathbf{s}, \mathbf{a})$$

At right: "(**Note**: $\pi_\beta$ may be a mixture of policies)"

"Objective:"

$$\max_\theta \mathbb{E}_ {p_\theta(\tau)} \left[ \sum_t r(\mathbf{s}_ t, \mathbf{a}_ t) \right]$$

followed by "<- expectation under *the learned policy* $\pi_\theta$" with "the learned policy $\pi_\theta$" underlined in blue, and under it, in blue, "distribution shift".

## Slide 9 — Why offline RL?

"**Offline RL** process" (blue):

- Given static dataset
- Train policy on provided dataset

"Where does the data come from?"

- human collected data
- data from a hand-designed system / controller
- data from previous RL run(s)
- a mixture of sources

## Slide 10 — Can we just use off-policy algorithms?

"**Recall**: Off-policy actor & critic updates (e.g. SAC):"

"Q-function update:"

$$\min_\phi \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\lVert \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}', \mathbf{a}') \right] \right) \right\rVert^2$$

"Subsequent policy update:"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}_ i, \mathbf{a}^\pi_ i) \quad \text{where } \mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$$

Bottom: "What happens if you optimize this using a static dataset?" and, smaller, "(e.g. say data collected by a mediocre policy)".

## Slide 11 — Can we just use off-policy algorithms?

![Slide 11 — Can we just use off-policy algorithms?](../images/07-offline-rl/slide-11.png)

"**Recall**: Off-policy critic objective"

$$\min_\phi \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\lVert \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}', \mathbf{a}') \right] \right) \right\rVert^2$$

A pink hand-drawn ellipse circles the target's expectation term, $\gamma \mathbb{E}_ {\mathbf{a}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} [\hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}', \mathbf{a}')]$.

"What happens if you optimize this using a static dataset?" and "(e.g. say data collected by a mediocre policy)".

"What happens when evaluating $Q$ on actions $\mathbf{a}'$ not in the dataset?"

Lower left: "Randomly init. $Q$-function for state $\mathbf{s}'$". A plot with a vertical axis labelled $Q(\mathbf{s}', \mathbf{a}')$ and a horizontal axis labelled "action $\mathbf{a}'$" (both black, with arrowheads at both ends). One blue wavy curve with no numeric ticks: it starts low on the left (left of the vertical axis), rises to a small hump near the vertical axis, dips to a trough, rises to a larger hump in the right-centre, then falls steeply toward the right edge. A short green bracket on the horizontal axis just right of the vertical axis, under the first hump, is labelled in green "data support".

Right, three bullets:

- $Q$-function will be unreliable on OOD actions
- policy will seek out actions where $Q$-function is over-optimistic
- After policy update, $Q$-values will become substantially overestimated.

Bottom: "**Another perspective**: learned policy deviates too much from behavior policy."

## Slide 12 — How to mitigate overestimation in offline RL?

Centred title "How to mitigate overestimation in offline RL?" and below it "This is the **core goal** of offline RL methods!"

## Slide 13 — The plan for today

Same as slide 6's list, with item 2 in bold: "Offline RL"

1. Why offline RL? Can we just run off-policy methods?
2. **Implicit policy constraint methods**
3. Conservative methods

A black curly brace at the right, beside items 2 and 3, is labelled "Part of homework 3!". The key-learning-goals block of slide 6 is absent.

## Slide 14 — Why offline RL versus imitation learning?

![Slide 14 — Why offline RL versus imitation learning?](../images/07-offline-rl/slide-14.png)

Top right: "(**Recall**: Imitation methods can't outperform the expert.)"

"Offline data may not be optimal!"

- "—> Offline RL can leverage reward information to outperform behavior policy."
- "—> Good offline RL methods can *stitch* together good behaviors."

Left diagram: a state graph of two trajectories that cross. Trajectory 1 (teal dots, black arrows, black labels): $s_1$ at lower left, then $s_2$, then $s_3$ (at the crossing), then $s_4$, $s_5$, $s_6$ running up and to the right. Trajectory 2 (dark-blue dots, grey arrows, grey labels): starts at a grey $s_1$ at bottom centre, then $s_2$, $s_3$, $s_4$, $s_5$, $s_6$ running up and to the left, then a grey $s_7$ drawn just beside the first trajectory's $s_3$, the two dots touching but not at the same point (the crossing), then $s_8$ and $s_9$ going up. An orange star next to the final dark-blue dot is labelled in orange $s_g$ and the dot in grey $s_9$.

Right:

- $s_1$ -> $s_3$ "is good behavior" (black text)
- $s_7$ -> $s_9$ "is good behavior" ($s_7$ and $s_9$ in grey)
- "Offline RL methods can learn a policy that goes from $s_1$ to $s_9$!" ($s_9$ in grey)

## Slide 15 — A simple way to leverage rewards in imitation

"If we have reward labels: imitate only the good trajectories?"

"**Filtered behavior cloning**:"

1. Rank trajectories by return

$$r(\tau) = \sum_{(\mathbf{s}_ t, \mathbf{a}_ t) \in \tau} r(\mathbf{s}_ t, \mathbf{a}_ t)$$

2. Filter dataset to include top k% of data

$$\tilde{D} : \lbrace \tau \mid r(\tau) \gt \eta \rbrace$$

3. Imitate filtered dataset:

$$\max_\theta \sum_{(\mathbf{s}, \mathbf{a}) \in \tilde{D}} \log \pi_\theta(\mathbf{a} \mid \mathbf{s})$$

Bottom: "A very primitive approach to using reward information." and "Therefore, a **good baseline** to test against!"

## Slide 16 — Better way to do weighted imitation learning?

![Slide 16 — Better way to do weighted imitation learning?](../images/07-offline-rl/slide-16.png)

"Could we weight each transition depending on **how good the action is**?"

"How do you measure how good an action is?" and at right "Recall: advantage function $A$", with a grey-outlined box holding $A^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) - V^\pi(\mathbf{s}_ t)$: how much better $\mathbf{a}_ t$ is

$$\theta \leftarrow \arg\max_\theta E_{\mathbf{s}, \mathbf{a} \sim D} \left[ \log \pi_\theta(\mathbf{a} \mid \mathbf{s}) \exp(A(\mathbf{s}, \mathbf{a})) \right]$$

Under the equation, a coral underline beneath the part $\arg\max_\theta E_{\mathbf{s}, \mathbf{a} \sim D} [\log \pi_\theta(\mathbf{a} \mid \mathbf{s})$ is labelled in coral "standard imitation learning", and a magenta underline beneath the factor $\exp(A(\mathbf{s}, \mathbf{a}))$ is labelled in magenta "with advantage weights".

Lower left, a light-blue box with a blue outline: "**Aside**: Can show that advantage-weighted objective approximates KL-constrained objective."

$$\pi_{new} = \arg\max_\pi E_{\mathbf{a} \sim \pi(\cdot \mid \mathbf{s})} Q(\mathbf{s}, \mathbf{a}) \quad \text{s.t.} \quad D_{KL}(\pi \Vert \pi_\beta) \lt \epsilon$$

"See Peters et al. (REPS), Rawlik et al. ("psi-learning")"

Lower right: the two-trajectory state graph of slide 14 (teal trajectory $s_1$ to $s_6$ in black, dark-blue trajectory $s_1$ to $s_9$ in grey, crossing where $s_3$ and $s_7$ sit side by side, orange star $s_g$ by $s_9$). A magenta-outlined box at the right, "Adv. weights for $s_3$", holds a dark-blue dot labelled $s_3$ with two arrows leaving it: a short, thick grey arrow pointing steeply up (the direction of the grey trajectory's step toward $s_8$) and a long, thin black arrow pointing to the upper right (the direction of the teal trajectory's step toward $s_4$). The slide does not say what the arrows' thickness and length encode; the lecture explains the weights aloud (≈34:10–35:42).

## Slide 17 — Better way to do weighted imitation learning?

![Slide 17 — Better way to do weighted imitation learning?](../images/07-offline-rl/slide-17.png)

Same as slide 16 (question, advantage-function box, the weighted imitation equation with its coral and magenta underlines and labels, the state graph and the "Adv. weights for $s_3$" box) except that the blue "Aside" box is gone and two lines are added:

"Advantage of which policy? We'll use $A^{\pi_\beta}$ for now."

and at the bottom "**Key question**: How to estimate the advantage function?"

## Slide 18 — Advantage-weighted regression

"Could we weight each transition depending on **how good the action is**?"

$$\theta \leftarrow \arg\max_\theta E_{\mathbf{s}, \mathbf{a} \sim D} \left[ \log \pi_\theta(\mathbf{a} \mid \mathbf{s}) \exp(A(\mathbf{s}, \mathbf{a})) \right]$$

with the coral underline and label "standard imitation learning" and the magenta underline and label "with advantage weights", as on slide 16.

"**Key question**: How to estimate the advantage function?"

At left in blue: "First simple approach". Then:

"Estimate $V^{\pi_\beta}(s)$ with Monte Carlo:"

$$\min_\phi \sum_{\mathbf{s}_ t \sim \mathcal{D}} \left\lVert \hat{V}^{\pi_\beta}_ \phi(\mathbf{s}_ t) - \sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right\rVert^2$$

A black curved arrow from the label "empirical return" points to the reward sum $\sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t)$.

(Printed slip, as on slides 4, 18 and 19: the sum runs over $t'$ but the reward inside it is written $r(\mathbf{s}_ t, \mathbf{a}_ t)$ rather than $r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$. The lecture describes this target as the sum of rewards from that state onward (≈1:38).)

"Approximate"

$$\hat{A}^{\pi_\beta}(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) - \hat{V}^{\pi_\beta}_ \phi(\mathbf{s}_ t)$$

Lower right, in teal: "**Question**: What do you learn for deterministic policy $\pi_\beta$?"

Citation at lower left: "Peng, Kumar, Zhang, Levine. Advantage-Weighted Regression. '19"

## Slide 19 — Advantage-weighted regression

![Slide 19 — Advantage-weighted regression](../images/07-offline-rl/slide-19.png)

"**Full AWR algorithm**"

1. Fit value function:

$$\min_\phi \sum_{\mathbf{s}_ t \sim \mathcal{D}} \left\lVert \hat{V}^{\pi_\beta}_ \phi(\mathbf{s}_ t) - \sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) \right\rVert^2$$

2. Train policy:

$$\max_\theta \mathbb{E}_ {(\mathbf{s}_ t, \mathbf{a}_ t) \sim \mathcal{D}} \left[ \log \pi_\theta(\mathbf{a}_ t \mid \mathbf{s}_ t) \exp \left( \frac{1}{\alpha} \left( \sum_{t'=t}^{T} r(\mathbf{s}_ t, \mathbf{a}_ t) - \hat{V}^{\pi_\beta}_ \phi(\mathbf{s}_ t) \right) \right) \right]$$

A black curved arrow from the label "hyperparameter" points to the $\frac{1}{\alpha}$.

(Printed slip, as on slides 4, 18 and 19: the sum runs over $t'$ but the reward inside it is written $r(\mathbf{s}_ t, \mathbf{a}_ t)$ rather than $r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'})$. The lecture describes this target as the sum of rewards from that state onward (≈1:38).)

Bottom left, in green:

- \+ Simple
- \+ Avoids querying or training on any OOD actions!

Bottom right, in red:

- \- Monte Carlo estimation is noisy
- \- $\hat{A}^{\pi_\beta}$ is for weaker policy than $\hat{A}^{\pi_\theta}$

Citation at lower left: "Peng, Kumar, Zhang, Levine. Advantage-Weighted Regression. '19"

## Slide 20 — Can we do better?

"Want to estimate advantages" "using TD updates" (green) "," "without querying $Q$ on OOD actions." (red).

"Estimate $Q$-function:"

$$\min_\psi E_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( \hat{Q}^\pi_ \psi(\mathbf{s}, \mathbf{a}) - \left( r + \gamma E_{\mathbf{a}' \sim \pi(\cdot \mid \mathbf{s})} \left[ \hat{Q}^\pi_ \psi(\mathbf{s}', \mathbf{a}') \right] \right) \right)^2 \right]$$

In the printed equation the subscript $\mathbf{a}' \sim \pi(\cdot \mid \mathbf{s})$ of the inner expectation is struck through with a red line, and the replacement $\mathbf{a}' \sim D$ is written beneath it. So the corrected target is $r + \gamma E_{\mathbf{a}' \sim D} [\hat{Q}^\pi_ \psi(\mathbf{s}', \mathbf{a}')]$.

Below, centred: "advantage-weighted actor-critic (AWAC)"

"**Note**: Gives Q-function estimate for $\pi_\beta$ underlying the data, \*not\* $\pi_\theta$" (the asterisks around "not" are printed literally).

## Slide 21 — Can we do better?

![Slide 21 — Can we do better?](../images/07-offline-rl/slide-21.jpg)

"Want to estimate advantages using TD updates, without querying $Q$ on OOD actions." (same colouring as slide 20)

"AWAC update:"

$$\hat{Q}^{\pi_\beta} \leftarrow \arg\min_Q E_{(\mathbf{s}, \mathbf{a}, \mathbf{s}', \mathbf{a}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r + \gamma Q(\mathbf{s}', \mathbf{a}') \right) \right)^2 \right]$$

A blue underline beneath $Q(\mathbf{s}', \mathbf{a}')$ (after the $\gamma$) is labelled in blue "a sample of $V^{\pi_\beta}(\mathbf{s}')$".

Lower left: "Can we estimate $Q$ for a policy that is better than $\pi_\beta$?" and "**Idea**: Use an asymmetric loss function". Beneath it, a plot with a horizontal axis labelled $x$ and a vertical axis, no numeric ticks, with two curves through the origin: a teal symmetric parabola labelled in teal $\ell_2(x)$, and a coral curve labelled in coral $\ell_2^\tau(x)$ that coincides with the parabola near the bottom, is slightly steeper on the left side and flatter (lower than the teal curve) on the right side.

Lower right: heading "Histogram of $V(\mathbf{s})$". A histogram with six adjacent light-blue bars, horizontal axis labelled $V(\mathbf{s})$ and a vertical axis with no ticks; bar heights rise from the first bar to a peak at the third bar, then fall: the fourth is nearly as tall as the third, the fifth shorter, the sixth shortest. A dashed vertical line through the third bar is labelled at its top in grey $E_{\mathbf{a} \sim \pi_\beta(\cdot \mid \mathbf{s})}[Q(\mathbf{s}, \mathbf{a})]$; a second dashed vertical line through the last bar is labelled in grey, $V(\mathbf{s})$ "for best policy in data support". A curved arrow from the label $\ell_2$ "loss gives us this!" points to the first dashed line, and a curved arrow from "Can we use another loss to get this?" points to the second.

Citation at lower left: "Kostrikov, Nair, Levine. Implicit Q-Learning. ICLR '22"

## Slide 22 — Aside: Expectile regression

![Slide 22 — Aside: Expectile regression](../images/07-offline-rl/slide-22.jpg)

"Instead of getting the mean of a random variable, can we get a higher or lower expectile?"

Left, "Expectile regression loss:"

$$\ell_2^\lambda(x) = \begin{cases} (1 - \lambda)x^2 & \text{if } x \lt 0 \cr \lambda x^2 & \text{otherwise} \end{cases}$$

Beneath it, a line chart with y-axis labelled $\ell_2^\lambda(x)$ (ticks 0.0 to 1.0 in steps of 0.2) and x-axis labelled $x$ (ticks −1.0 to 1.0 in steps of 0.5). Five curves, each a parabola through (0, 0), with legend at upper centre:

- $\lambda = 0.01$ (blue): about 0.99 at $x = -1.0$, about 0.01 at $x = 1.0$.
- $\lambda = 0.1$ (green): about 0.9 at $x = -1.0$, about 0.1 at $x = 1.0$.
- $\lambda = 0.5$ (orange): about 0.5 at both $x = -1.0$ and $x = 1.0$.
- $\lambda = 0.9$ (pink): about 0.1 at $x = -1.0$, about 0.9 at $x = 1.0$.
- $\lambda = 0.99$ (yellow): about 0.01 at $x = -1.0$, about 0.99 at $x = 1.0$.

Right, "Example with a 2D random variable": a scatter plot with x-axis labelled $x$ (−1.0 to 1.0) and y-axis labelled $y$ (about −0.5 to 1.5). Many small blue dots form a thick arch (an inverted parabola) peaking near $y \approx 1.0$ to $1.2$ at $x = 0$ and falling to about −0.3 to 0.3 at the ends. Five arch-shaped fitted curves are overlaid, with a legend inside the plot:

- $m_{0.01}(x)$ (blue): the lowest curve, about 0.7 at $x = 0$ and about −0.3 at $x = -1.0$.
- $m_{0.1}(x)$ (green): second lowest, about 0.8 at $x = 0$.
- $m_{0.5}(x)$ (orange): about 1.0 at $x = 0$, about 0.0 at $x = \pm 1.0$.
- $m_{0.9}(x)$ (pink): about 1.2 at $x = 0$.
- $m_{0.99}(x)$ (yellow): the highest curve, about 1.25 at $x = 0$.

The legend box covers part of the scatter. The values above are read by eye from the curves.

Citation at lower left: "Kostrikov, Nair, Levine. Implicit Q-Learning. ICLR '22"

## Slide 23 — Can we do better?

![Slide 23 — Can we do better?](../images/07-offline-rl/slide-23.png)

"Want to estimate advantages using TD updates, without querying $Q$ on OOD actions." (same colouring)

"**Full algorithm**"

"Fit $V$ with expectile loss:"

$$\hat{V}(\mathbf{s}) \leftarrow \arg\min_V E_{(\mathbf{s}, \mathbf{a}) \sim D} \left[ \ell_2^\lambda \left( V(\mathbf{s}) - \hat{Q}(\mathbf{s}, \mathbf{a}) \right) \right] \quad \text{using small } \lambda \lt 0.5$$

"Update $Q$ with typical MSE loss:"

$$\hat{Q}(\mathbf{s}, \mathbf{a}) \leftarrow \arg\min_Q E_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r + \gamma \hat{V}(\mathbf{s}') \right) \right)^2 \right]$$

"Extract policy with AWR:"

$$\hat{\pi} \leftarrow \arg\max_\pi E_{\mathbf{s}, \mathbf{a} \sim D} \left[ \log \pi(\mathbf{a} \mid \mathbf{s}) \exp \left( \frac{1}{\alpha} \left( \hat{Q}(\mathbf{s}, \mathbf{a}) - \hat{V}(\mathbf{s}) \right) \right) \right]$$

A black curved arrow at the left runs from the "Update $Q$" line back up to the "Fit $V$" line, marking a loop over those two steps.

Bottom left, in green:

- \+ Never need to query OOD actions!
- \+ Policy (still) only trained on actions in data.
- \+ Decoupling actor & critic training —> computationally fast

Bottom right: "policy improvement is implicit" and in bold "-> implicit Q-learning (IQL)". In a blue dotted-outline box: "You will implement IQL in homework 3!"

Citation at lower left: "Kostrikov, Nair, Levine. Implicit Q-Learning. ICLR '22"

## Slide 24 — The plan for today

Same as slide 6's list, with item 3 in bold: "Offline RL"

1. Why offline RL? Can we just run off-policy methods?
2. Implicit policy constraint methods
3. **Conservative methods**

A black curly brace at the right beside items 2 and 3 is labelled "Part of homework 3!".

## Slide 25 — How to mitigate overestimation in offline RL?

![Slide 25 — How to mitigate overestimation in offline RL?](../images/07-offline-rl/slide-25.png)

"**Recall**: Randomly init. $Q$-function for state $\mathbf{s}'$" and the same plot as slide 11: vertical axis $Q(\mathbf{s}', \mathbf{a}')$, horizontal axis "action $\mathbf{a}'$", one blue wavy curve (a small hump near the vertical axis, a trough, a larger hump, then falling steeply to the right) and a green bracket on the horizontal axis labelled in green "data support".

Right: "Can we discourage overestimation?" and smaller "without explicitly modeling the behavior policy". Then "What if we just push down on large Q-values?"

$$\hat{Q}^\pi = \arg\min_Q \max_\mu E_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma E_\pi [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha E_{\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$$

A teal underline beneath the squared-error term is labelled in teal "standard critic update"; a blue underline beneath the $\alpha E_{\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})}[Q(\mathbf{s}, \mathbf{a})]$ term is labelled in blue "push down on large Q-values".

"Can show that $\hat{Q}^\pi \leq Q^\pi$ for large enough $\alpha$"

Credit at lower left: "Slide adapted from Sergey Levine"

## Slide 26 — How to mitigate overestimation in offline RL?

![Slide 26 — How to mitigate overestimation in offline RL?](../images/07-offline-rl/slide-26.png)

"Can we discourage overestimation?" and smaller "without explicitly modeling the behavior policy".

$$\hat{Q}^\pi = \arg\min_Q \max_\mu E_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma E_\pi [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha E_{\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})] - \alpha E_{(\mathbf{s}, \mathbf{a}) \sim D} [Q(\mathbf{s}, \mathbf{a})]$$

The last term is printed on a second line, under the $\alpha E_{\mathbf{s} \sim D, \ldots}$ term. Labels: a teal underline beneath the squared-error term, with "standard critic update" (teal) above it; a blue underline beneath the first $\alpha$ term, with "push down on large Q-values" (blue) above it; a magenta underline beneath the second-line term $-\alpha E_{(\mathbf{s}, \mathbf{a}) \sim D}[Q(\mathbf{s}, \mathbf{a})]$, labelled in magenta "push up on Q-values for $(\mathbf{s}, \mathbf{a})$ in the data".

"No longer guaranteed that $\hat{Q}^\pi \leq Q^\pi$ for all $(\mathbf{s}, \mathbf{a})$."

"BUT, guaranteed that $E_{\pi(\mathbf{a} \mid \mathbf{s})}[\hat{Q}^\pi(\mathbf{s}, \mathbf{a})] \leq E_{\pi(\mathbf{a} \mid \mathbf{s})}[Q^\pi(\mathbf{s}, \mathbf{a})]$ for all $\mathbf{s} \in D$."

In bold: "Conservative Q-learning (CQL)"

Credit at lower left: "Slide adapted from Sergey Levine"

## Slide 27 — How to mitigate overestimation in offline RL?

"**Conservative Q-learning (CQL)**" and "Full algorithm"

1. Update $\hat{Q}^\pi$ using $L_{CQL}$ using $D$
2. Update policy $\pi$

A black curved arrow at the left runs from step 2 back up to step 1.

"If actions are discrete:"

$$\pi(\mathbf{a} \mid \mathbf{s}) = \begin{cases} 1 & \text{if } \mathbf{a} = \arg\max_{\bar{\mathbf{a}}} \hat{Q}^\pi(\mathbf{s}, \bar{\mathbf{a}}) \cr 0 & \text{otherwise} \end{cases}$$

"If actions are continuous:"

$$\theta \leftarrow \theta + \eta \nabla_\theta E_{\mathbf{s} \sim D, \mathbf{a} \sim \pi_\theta(\cdot \mid \mathbf{s})} \left[ \hat{Q}^\pi(\mathbf{s}, \mathbf{a}) \right]$$

Credit at lower left: "Slide adapted from Sergey Levine"

## Slide 28 — How to mitigate overestimation in offline RL?

(No printed slide number; the label 28 is inferred from position.)

"**Conservative Q-learning (CQL)**"

1. Update $\hat{Q}^\pi$ using $L_{CQL}$ using $D$
2. Update policy $\pi$

(with the loop arrow from step 2 to step 1 at left)

"How compute objective $L_{CQL}$?"

$$\hat{Q}^\pi = \arg\min_Q \max_\mu E_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma E_\pi [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha E_{\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})] - \alpha E_{(\mathbf{s}, \mathbf{a}) \sim D} [Q(\mathbf{s}, \mathbf{a})] + R(\mu)$$

The last two terms are on a second line. A pink hand-drawn circle surrounds the $\max_\mu$; a blue underline sits beneath the first $\alpha$ term; a green underline beneath $R(\mu)$ is labelled in green "regularizer".

"Common choice:"

$$R(\mu) = E_{\mathbf{s} \sim D} \left[ \mathcal{H}(\mu(\cdot \mid \mathbf{s})) \right]$$

"With max entropy regularizer $R$, optimal"

$$\mu(\mathbf{a} \mid \mathbf{s}) \propto \exp \left( Q(\mathbf{s}, \mathbf{a}) \right)$$

"Then:"

$$E_{\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})] = \log \sum_{\mathbf{a}} \exp \left( Q(\mathbf{s}, \mathbf{a}) \right)$$

(a blue underline beneath the left-hand side). Then "Don't need to construct $\mu$ directly."

In a blue dotted-outline box at lower right: "You will implement CQL in homework 3!"

Credit at lower left: "Slide adapted from Sergey Levine"

## Slide 29 — The plan for today

Same as slide 6's list with no item bold: "Offline RL"

1. Why offline RL? Can we just run off-policy methods?
2. Implicit policy constraint methods
3. Conservative methods

A black curly brace at the right beside items 2 and 3 is labelled "Part of homework 3!".

## Slide 30 — Summary

"**Why offline RL?** Online data is expensive. *Reusing offline data* is good!"

"**Key challenge**: Overestimating Q-values because of shift between $\pi_\beta$ and $\pi_\theta$"

"Approaches:"

1. filtered or weighted imitation learning is a simple baseline
2. **implicitly constrain** the policy to $\pi_\beta$ by only supervising on actions in data
3. construct conservative objective by **penalizing Q-values**

"Trajectory stitching allows offline RL methods to improve over imitation."

## Slide 31 — An example application

![Slide 31 — An example application](../images/07-offline-rl/slide-31.jpg)

"Optimizing policy for sending notifications to users on LinkedIn"

Left, a system-architecture flow diagram (small print, drawn with boxes and arrows). Reading it by region:

- Top left: an orange cylinder "Offline Features" with an arrow to a green wavy box "Feature Push", which has an arrow into a dashed box titled "Nearline Samza".
- Inside the dashed "Nearline Samza" box: an orange cylinder "Online Feature Store"; a blue wavy box "Batch Notification" (linked to the feature store by a light-blue line); an orange cylinder "Notification Queue" with an arrow pointing left to "Batch Notification"; a green box "Policy"; and a diamond "Send now?". An arrow runs from the feature store down to "Policy"; an arrow runs from "Policy" to the "Send now?" diamond. From the diamond, "Yes" leads right, outside the dashed box, to a blue person icon labelled "User"; "No" (red) leads right to "Drop"; and a second "No" (blue) leads up to the "Notification Queue".
- Outside, upper right: a white box "Batch Notification Providers" with an arrow to the "Notification Queue".
- A red wavy box "Near-Real-Time Notification" at left, with a red line running up and across into "Policy".
- A second dashed box titled "Offline" at the bottom holds a green box "Offline Off-Policy Training" and a green box "Decision and Feature Snapshots". An arrow runs from "Policy" down to "Decision and Feature Snapshots", then left to "Offline Off-Policy Training"; an arrow runs from "Offline Features" down into "Offline Off-Policy Training"; and an arrow labelled "Deploy new policy" runs from "Offline Off-Policy Training" up to "Policy".

Right, definitions: "WAU: weekly active users", "Volume: total # of notifications", "CTR: click-through-rate of notifications".

A table, captioned "Table 1: Online A/B test results for DDQN with and without CQL":

| Metric | DDQN vs. Baseline | DDQN + CQL vs. Baseline |
| --- | --- | --- |
| Sessions | not stat sig | + 0.24% |
| WAU | -0.69% | + 0.18% |
| Volume | +7.72% | -1.73% |
| CTR | -7.79% | +2.26% |

Citation: "Prabhakar, Yuan, Yang, Sun, Muralidharan. Multi-Objective Optimization of Notifications Using Offline RL. '22"

## Slide 32 — Which offline RL algorithm to use?

"If you only want to train offline:"

- **Filtered behavior cloning**: Good first approach to using offline data.
- **Implicit Q-learning**: Can stitch data & explicitly constrained to data support
- **Conservative Q-learning**: Just one hyperparameter

"If you want offline pre-training + online fine-tuning:"

- **Implicit Q-learning**: Seems most performant.

"See also: IDQL (Hansen-Estruch et al. IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies. 2023)"

Lower right: "**Note**: Still an active area of research!"

## Slide 33 — Next time

"**Friday lecture:** How do we get rewards??"

Then a second heading, "Course reminders", with two columns.

Left, "Project":

- CA mentors assigned
- Fill out AWS form for GPU quota
- Proposal due Friday

Under the third bullet, in magenta: "graded fairly lightly, for your benefit!"

Right, "Homework":

- Homework 2 due next Friday. (start early!!)
