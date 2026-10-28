---
title: Lecture 6 — Q-Learning (slide deck)
lecture: 6
slides: 29
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–29). Pages 3, 15, 20, 26 and 27 print no number; their labels are inferred from position. Pages 5, 8, 12–14, 17, 19 and 21–25 print the number in the bottom-right corner.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (7, 15, 19, 20, 24, 26) were checked by Claude Opus 5.5 against 1400px renders, plus slide 25 for its printed exponent and a 500 dpi crop of slide 26's lower bars. Slides 7, 15, 24 and 25 agree, and slide 26's 49 printed percentages and its table agree cell by cell. Corrections were small, with one exception. Slide 26's human-level line had been placed at 100%; it sits between H.E.R.O. (76%) and Asterix (69%). On slide 19, two Q curves at epoch 25 were re-read; on slide 20, Space Invaders' DQN estimate at 100 million steps, where its two true-value lines sit, and when Zaxxon's Double DQN estimate leaves zero were re-read.
images: 12 slides are rendered and shown under their headings (3, 4, 5, 7, 8, 12, 15, 18, 19, 20, 24, 26). Not rendered — slides 9, 10, 11, 13, 14, 17, 21, 22, 23 and 25, equation slides whose strikethroughs, highlights, loop arrows, braces and annotation arrows are transcribed here; and the title, reminder, plan, advice and next-time slides (1, 2, 6, 16, 27, 28, 29), which are text.
---

# Lecture 6 — Q-Learning: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 6 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 29 pages. Pages 3, 15, 20, 26 and 27 print no number at all; their labels are inferred from their position between the numbered neighbours. Page 5, page 8, pages 12–14, page 17, page 19 and pages 21–25 print their number in the bottom-right corner rather than at the bottom centre; those numbers are real.

The recording covers slides 3–29. It opens on the recap (slide 3) without discussing slide 2's reminders, and it ends at about 1:01:12 on slide 29's first line, "Done with online RL methods!"; slide 29's next-week topics and reminders are not discussed. Slide 7's thought exercise is answered aloud over the slide's picture (≈7:08–12:32). The algorithm with its targets is written on the board (≈18:00–19:34), and a value-function lookahead is described without a slide (≈23:24–24:56); the recording does not show the board, so the transcript is the only source for those.

Companion pages: [wiki page for this lecture](../../wiki/06-q-learning.md) · [transcript](../transcripts/06-q-learning.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title, course reminders |
| 3–5 | Recap: value, Q and advantage objects, methods, off-policy policy evaluation |
| 6 | Plan for today |
| 7–11 | Thought exercise on a greedy policy from Q-values, omitting the policy gradient, policy iteration, from actor-critic to critic only |
| 12–13 | Q-learning: why it makes sense, the Bellman equations, convergence |
| 14–15 | What data to collect (epsilon-greedy, Boltzmann), full Q-learning with replay buffer |
| 16–18 | Q-learning in practice: making Q-learning stable with a frozen target network, the DQN algorithm |
| 19–20 | Are the Q-values accurate? Atari training curves and Q-value plots, DQN versus Double DQN value estimates |
| 21–23 | Overestimation in Q-learning, double Q-learning, double Q-learning with the current and target networks |
| 24–25 | N-step returns for Q-learning targets |
| 26–27 | Q-learning in practice (Atari, robot grasping), when to use PPO, DQN or SAC variants |
| 28–29 | Plan recap, next time (offline RL, reward learning) and course reminders |

## Slide 1 — Q-Learning

Title slide. Title: "Q-Learning". Below: "CS 224R". No images.

## Slide 2 — Course reminders

- Today
  - **Homework 1** due, **homework 2** out
  - Sending out 100 dollars of AWS credits + "form for GPU quota." (printed with a dollar sign and the figure 100; the words "form for GPU quota." sit in a pale pink highlight)
  - Extra Q-learning section (1:30-2:30 pm)

At lower right, in red-orange bold "Very important" followed in black by "if you want GPUs, especially for default project".

## Slide 3 — Recap: Some Useful Objects

![Slide 3 — Recap: Some Useful Objects](../images/06-q-learning/slide-3.jpg)

(No printed slide number; the label 3 is inferred from position.)

- *value function* $V^\pi(\mathbf{s})$ - future expected rewards starting at $\mathbf{s}$ and following $\pi$
- *Q-function* $Q^\pi(\mathbf{s}, \mathbf{a})$ - future expected rewards starting at $\mathbf{s}$, taking $\mathbf{a}$, then following $\pi$

Two diagrams side by side, each with its defining equation above it. Left equation:

$$V^\pi(\mathbf{s}_ t) = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t \right]$$

Left diagram: a blue dot labelled "s" at the lower left with five black wavy trajectory arrows fanning out from it to the upper right, labelled "run $\pi$" at the left. A thin red double-headed arrow running alongside, parallel to the trajectories, is labelled "expected sum of these rewards". To the right of it the label $V^\pi(\mathbf{s})$ with the s in blue.

Right equation:

$$Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{T} E_{\pi_\theta} \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right]$$

Right diagram: the same blue dot "s" with a similar fan of five trajectories, grey except for two that are drawn in green (with a short green arrow at the start); the label "take action **a**, then run $\pi$" (the a in green). A thin red double-headed arrow labelled "expected sum of these rewards" (green text on a light-grey band) runs alongside. To the right the label $Q^\pi(\mathbf{s}, \mathbf{a})$ with s in blue and a in green.

## Slide 4 — Recap: Methods

![Slide 4 — Recap: Methods](../images/06-q-learning/slide-4.jpg)

Left: the online RL cycle diagram. A blue box "Run policy to collect batch of data" (upper left) and a pink box "Improve policy" (lower left) are drawn at full strength; a middle box "Fit model to estimate expected return" and its label "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$" are drawn faded (nearly invisible). Faded grey arrows run from the blue box to the faded teal box and from it to the pink box; a solid dark-grey arrow runs from the pink box up the left side to the blue box. Under the pink box, in magenta: $\theta \leftarrow \theta + \nabla_\theta J(\theta)$.

Right, heading: "Online RL with policy gradients"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) \left( \left( \sum_{t'=t}^{T} r(\mathbf{s}_ {i,t'}, \mathbf{a}_ {i,t'}) \right) - b \right)$$

The parts are colour-highlighted and labelled underneath: the two sums in a blue highlight labelled "samples from policy"; $\log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t})$ in a pink highlight labelled "policy log likelihood"; the inner reward sum in a green highlight labelled "reward to go"; and $-b$ in a teal highlight labelled "baseline".

Below it: "Do more of the" "above average" (green) "stuff, less of the" "below average" (red) "stuff."

Heading: "Online RL with actor-critic"

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_ {i,t} \mid \mathbf{s}_ {i,t}) A^\pi(\mathbf{s}_ {i,t}, \mathbf{a}_ {i,t})$$

The two sums are in a blue highlight and the $A^\pi$ term in a teal highlight.

"Estimate what is good and bad" (teal), ", then" (black) "do more of the good stuff." (magenta).

## Slide 5 — Recap: Off-Policy Policy Evaluation

![Slide 5 — Recap: Off-Policy Policy Evaluation](../images/06-q-learning/slide-5.jpg)

Left: a rounded grey-outlined panel headed "Online RL with actor-critic" containing the small three-box cycle: blue box "Run policy to collect batch of data" (upper left), teal box "Fit model to estimate expected return" (middle right), pink box "Improve policy" (lower left), with grey arrows blue to teal, teal to pink (labelled in teal "Estimate $V^\pi$, $Q^\pi$, or $A^\pi$") and pink up to blue. Under the pink box, in magenta: $\theta \leftarrow \theta + \nabla_\theta J(\theta)$. Below, "Estimate what is good and bad" (teal) ", then" "do more of the good stuff." (magenta).

Right: "**For off-policy version**: can we estimate" "Q-function" (teal) "using past policy data?"

A fully connected neural-network diagram (four columns of nodes with dense connections). The input on the left is labelled $\mathbf{s}, \mathbf{a}$; the output on the right is labelled $\hat{Q}^\pi_\phi(\mathbf{s}, \mathbf{a})$ with "parameters $\phi$" beneath.

"For *any* $(\mathbf{s}, \mathbf{a})$:"

$$Q^{\pi_\theta}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a}), \bar{\mathbf{a}}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ Q^{\pi_\theta}(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

Numbered steps at bottom left:

1. Sample $(\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i)$ from the buffer
2. Sample $\bar{\mathbf{a}}'_ i \sim \pi_\theta(\cdot \mid \mathbf{s}'_ i)$ from *current* policy

$$Q^{\pi_\theta}(\mathbf{s}_ i, \mathbf{a}_ i) \approx r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma Q^{\pi_\theta}(\mathbf{s}'_ i, \bar{\mathbf{a}}'_ i)$$

A curly brace under the right-hand side $r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma Q^{\pi_\theta}(\mathbf{s}'_ i, \bar{\mathbf{a}}'_ i)$ is labelled $y_i$.

At right:

$$\mathcal{L}(\phi) = \frac{1}{N} \sum_i \left\Vert \hat{Q}^\pi_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - y_i \right\Vert^2$$

"**Note**: To be accurate, targets $y_i$ need sufficient action coverage".

## Slide 6 — The plan for today

"Value-based RL methods", then the outline exactly as printed:

1. Q-learning RL method
   a. Policy iteration
   b. Bellman optimality equation
   c. How to collect data for Q-learning methods
2. Q-learning in practice
   d. Target networks
   e. Double DQN
   f. N-step returns

At right, in bold italics: "Part of homework 2!"

"**Key learning goals**:"

- How Q-functions relate to policies
- How to do RL without learning an explicit policy
- How to stabilize Q-learning in practice

## Slide 7 — A thought exercise

![Slide 7 — A thought exercise](../images/06-q-learning/slide-7.jpg)

Top right, in a rounded grey box: "Recall: *Definition of Q-values*"

$$Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t) = \sum_{t'=t}^{T} E_\pi \left[ r(\mathbf{s}_ {t'}, \mathbf{a}_ {t'}) \mid \mathbf{s}_ t, \mathbf{a}_ t \right]$$

"“total reward we get if we take $\mathbf{a}_ t$ in $\mathbf{s}_ t$... ... and then follow the policy $\pi$”"

Main text: "For some policy $\pi$, say you have an accurate estimate $\hat{Q}^\pi(\mathbf{s}, \mathbf{a})$ for all $\mathbf{s}, \mathbf{a}$"

"Define a new policy"

$$\pi_{\text{new}}(\mathbf{a}_ t \mid \mathbf{s}_ t) = \begin{cases} 1 & \text{if } \mathbf{a}_ t = \arg\max_{\mathbf{a}} Q_\phi(\mathbf{s}_ t, \mathbf{a}) \cr 0 & \text{otherwise} \end{cases}$$

Figure: a "2D navigation example". A rectangle (white inside, thin dark outline) represents the 2D world. A small light-blue square near the lower left is labelled in blue to the left "Initial state distribution $p(\mathbf{s}_ 1)$". A yellow-orange star sits at the right side, a little above the bottom. Under the rectangle: "Reward is 1 at ★, 0 elsewhere" (the star in orange). To the right of the rectangle: "Current policy $\pi$ goes right in all states" with a large dark-grey right-pointing arrow beside it. Further right, two vertical bracket marks (a shorter orange one and a taller magenta one) next to the text "takes" "1 timestep" (orange) ", " "2 timesteps" (magenta) "to travel these distances". The brackets are drawn at the right edge of the world next to the star's height region, the short one spanning a small vertical distance and the long one a larger vertical distance; the slide does not label what they measure beyond this text.

Bottom, in teal: "**Questions:**" "Is the new policy better, worse, or the same as the original policy? Is it the optimal policy? Why or why not?"

## Slide 8 — Can we omit policy gradient completely?

![Slide 8 — Can we omit policy gradient completely?](../images/06-q-learning/slide-8.jpg)

Top left: $Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t)$: expected reward from taking $\mathbf{a}_ t$ and subsequently following $\pi$

Below: $\arg\max_{\mathbf{a}_ t} Q^\pi(\mathbf{s}_ t, \mathbf{a}_ t)$: best action from $\mathbf{s}_ t$, if we then follow $\pi$ afterwards

At right of that line: "*at least* as good as any $\mathbf{a}_ t \sim \pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$ *regardless* of what $\pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$ is!"

A thick black curved arrow points from the text "forget policies, let's just do this!" up to the $\arg\max$ expression.

Lower left:

$$\pi'(\mathbf{a}_ t \mid \mathbf{s}_ t) = \begin{cases} 1 \text{ if } \mathbf{a}_ t = \arg\max_{\mathbf{a}} Q^\pi(\mathbf{s}_ t, \mathbf{a}) \cr 0 \text{ otherwise} \end{cases}$$

A thick black curved arrow points from the text "as good as $\pi$ (probably better)" up to this definition.

Lower right: the three-box cycle diagram. Blue box "Run policy to collect batch of data" (upper), teal box "Fit model to estimate expected return" (right), pink box "Improve policy" (lower), with grey arrows blue to teal, teal to pink, pink up to blue. Beside the teal-to-pink arrow, in teal: "Estimate $Q^\pi$". Under the pink box, in magenta: $\pi \leftarrow \pi'$.

Footer, bottom left: "Slide adapted from Sergey Levine".

## Slide 9 — From actor-critic to critic only

The title line reads "~~Off-policy actor critic with replay buffer~~" (struck through with a red line) followed by "Q-learning". A green bracket arrow loops from step 5 (bottom) back around the left side to step 1 (top). The steps, with strikethroughs in red:

1. take action $\mathbf{a} \sim \pi(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i \rbrace$ from buffer $\mathcal{R}$
3. update $\hat{Q}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$ where $\mathbf{a}'_ i \sim \pi(\cdot \mid \mathbf{s}'_ i)$
4. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}^\pi_ i \mid \mathbf{s}_ i) \hat{Q}^\pi(\mathbf{s}_ i, \mathbf{a}^\pi_ i)$ where $\mathbf{a}^\pi_ i \sim \pi_\theta(\mathbf{a} \mid \mathbf{s}_ i)$ (red strikethrough)
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ (red strikethrough)

A new step is written beneath the struck ones, numbered 4:

4. define new policy $\pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$, printed as a cases expression: 1 if $\mathbf{a}_ t = \arg\max_{\mathbf{a}} \hat{Q}_ \phi(\mathbf{s}_ t, \mathbf{a})$, 0 otherwise

On this page the word "Q-learning" beside the title is visible; the struck-through steps 4 and 5 remain visible.

## Slide 10 — From actor-critic to critic only

Same as slide 9 (struck-through title "Off-policy actor critic with replay buffer", the green loop arrow, steps 1–3, struck-through old steps 4 and 5, new step 4 "define new policy" with the arg max definition), with these changes. The word "Q-learning" next to the title is gone. Annotations are added at the right: "<- policy *evaluation*" beside step 3, with "(**Note**: can do multiple gradient steps here)" beneath it, and "<- policy *improvement*" beside the new step 4. Below the steps, centred: "“Policy iteration”". At the bottom: "Can we improve the policy in the Q-function update?"

## Slide 11 — From actor-critic to critic only

Same layout as slide 9 (title with struck-through "Off-policy actor critic with replay buffer" and "Q-learning"; green loop arrow; steps 1 and 2 unchanged; struck-through old steps 4 and 5; new step 4 "define new policy"). Changes: in step 3 the text now reads "update $\hat{Q}^\pi_\phi$ using targets" with the $\pi$ superscript on $\hat{Q}$ struck through in red, and the old target $y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_ i, \mathbf{a}'_ i)$ where $\mathbf{a}'_ i \sim \pi(\cdot \mid \mathbf{s}'_ i)$ is struck through in red. The new target is written to its right:

$$y_i = r_i + \gamma \max_{\mathbf{a}'} \hat{Q}_ \phi(\mathbf{s}'_ i, \mathbf{a}')$$

A black curly brace under $r_i + \gamma \max_{\mathbf{a}'} \hat{Q}_ \phi(\mathbf{s}'_ i, \mathbf{a}')$ is labelled "Q-values for new policy!".

## Slide 12 — From actor-critic to critic only

![Slide 12 — From actor-critic to critic only](../images/06-q-learning/slide-12.jpg)

Heading "Q-learning" (the struck-through lines of slides 9–11 are gone). Steps in a green loop-arrow (from step 4 back to step 1):

1. take action $\mathbf{a} \sim \pi(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\lbrace \mathbf{s}_ i, \mathbf{a}_ i, r_i, \mathbf{s}'_ i \rbrace$ from buffer $\mathcal{R}$
3. update $\hat{Q}_ \phi$ using targets $y_i = r_i + \gamma \max_{\mathbf{a}'} \hat{Q}_ \phi(\mathbf{s}'_ i, \mathbf{a}')$ (this whole line sits in a light-blue highlight)
4. define new policy $\pi(\mathbf{a}_ t \mid \mathbf{s}_ t)$, printed as a cases expression: 1 if $\mathbf{a}_ t = \arg\max_{\mathbf{a}} \hat{Q}_ \phi(\mathbf{s}_ t, \mathbf{a})$, 0 otherwise

At right, a rounded orange-outlined box headed "Terminology" (orange) containing "Bellman equation" (orange-red) and "Bellman optimality equation" (magenta).

"Why does make sense?" (sic, as printed)

"Recall:" 

$$Q^\pi(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a}), \bar{\mathbf{a}}' \sim \pi(\cdot \mid \mathbf{s}')} \left[ Q^\pi(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

"for *all* $(\mathbf{s}, \mathbf{a})$". The equation is underlined in orange-red (matching "Bellman equation" in the terminology box). Beneath: "-> This holds for any policy $\pi$ (including the optimal policy $\pi^\ast$)".

"If $\pi^\ast$ is the optimal policy, we also get:"

$$Q^{\pi^\ast}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a})} \left[ \max_{\bar{\mathbf{a}}'} Q^{\pi^\ast}(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

"for *all* $(\mathbf{s}, \mathbf{a})$". This equation is underlined in magenta (matching "Bellman optimality equation"). A black curved arrow from the text "the action $\pi^\ast$ would take in state $\mathbf{s}'$" points to the $\max$ term under the underline.

Last line: "When we optimize" "step 3," (in a light-blue highlight) "we are trying to make this equation hold!"

## Slide 13 — From actor-critic to critic only

Same top part as slide 12, shifted up (heading "Q-learning", the four steps in the green loop with step 3 in the light-blue highlight), but the "Terminology" box and the "Recall" Bellman equation are gone. At right: "**Note**: Q-learning is" "off-policy" (orange-red).

"Why does make sense?" (sic)

"If $\pi^\ast$ is the optimal policy, we also get:"

$$Q^{\pi^\ast}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{s}' \sim p(\cdot \mid \mathbf{s}, \mathbf{a})} \left[ \max_{\bar{\mathbf{a}}'} Q^{\pi^\ast}(\mathbf{s}', \bar{\mathbf{a}}') \right]$$

"for" "*all* $(\mathbf{s}, \mathbf{a})$" (the words "all (s, a)" in a pale red highlight).

"When we optimize" "step 3," (light-blue highlight) "we are trying to make this equation hold!"

"**Will this algorithm converge to optimal $Q^{\pi^\ast}$?**"

"Yes, if you maintain a table of Q-values for every state and action. More generally, no." followed by a crying-face emoji. "Can construct scenarios where it diverges, even with linear Q. *But*, it can be made to work well".

## Slide 14 — What kind of data should we collect?

"Q-learning" with the four-step loop as on slide 13 (steps 1–4 in the green loop arrow). In step 1, the policy $\pi(\mathbf{a} \mid \mathbf{s})$ is struck through with a red line and a red annotation above reads "from some exploration policy". A blue curved arrow from the blue text "want coverage for many actions $\mathbf{a}$" points at the target in step 3, $y_i = r_i + \gamma \max_{\mathbf{a}'} \hat{Q}_ \phi(\mathbf{s}'_ i, \mathbf{a}')$. At right: "**Note**: Q-learning is" "off-policy" (orange-red).

Blue heading: "What are good choices for data collection?"

1. With probability $\epsilon$, take uniformly random action

$$\pi(\mathbf{a}_ t \mid \mathbf{s}_ t) = \begin{cases} 1 - \epsilon & \text{if } \mathbf{a}_ t = \arg\max_{\mathbf{a}_ t} Q_\phi(\mathbf{s}_ t, \mathbf{a}_ t) \cr \epsilon / (\vert \mathcal{A} \vert - 1) & \text{otherwise} \end{cases}$$

labelled to the right “*epsilon-greedy*”. Grey text below: "Often start with larger $\epsilon$, decrease over training."

2. Take actions with probability proportional to their Q-value.

$$\pi(\mathbf{a}_ t \mid \mathbf{s}_ t) \propto \exp(Q_\phi(\mathbf{s}_ t, \mathbf{a}_ t))$$

labelled to the right “*Boltzmann exploration*”.

## Slide 15 — Putting it together

![Slide 15 — Putting it together](../images/06-q-learning/slide-15.jpg)

(No printed slide number; the label 15 is inferred from position.)

"full Q-learning with replay buffer:"

1. collect data $\lbrace (\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i) \rbrace$ using some policy, add it to $\mathcal{R}$ (the first phrase in a light-blue highlight, $\mathcal{R}$ in a light-green highlight)
2. sample a batch $(\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i)$ from $\mathcal{R}$
3. $\phi \leftarrow \phi - \alpha \sum_i \frac{dQ_\phi}{d\phi}(\mathbf{s}_ i, \mathbf{a}_ i) \left( Q_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - \left[ r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \max_{\mathbf{a}'} Q_\phi(\mathbf{s}'_ i, \mathbf{a}') \right] \right)$

Steps 2 and 3 together sit in a pink highlighted band. Two green loop arrows: an outer one from step 3 back to step 1, and a small inner one labelled $K \times$ looping from step 3 back to step 2. At right of the band: "K = 1 is common, though larger K more efficient".

"result: $Q_\phi$" and "final policy"

$$\pi(\mathbf{a}_ t \mid \mathbf{s}_ t) = \begin{cases} 1 & \text{if } \mathbf{a}_ t = \arg\max_{\mathbf{a}} Q_\phi(\mathbf{s}_ t, \mathbf{a}) \cr 0 & \text{otherwise} \end{cases}$$

Diagram at bottom: on the left an Earth globe icon (the world); a blue arrow pointing right from the globe to a green cylinder, labelled above $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$; a blue arrow pointing left from the cylinder back to the globe, labelled below $\pi(\mathbf{a} \mid \mathbf{s})$ (e.g., $\epsilon$-greedy). The green cylinder is labelled "dataset of transitions (“replay buffer”)". To its right a thick magenta curved arrow leaves the top of the cylinder and curves back into its bottom, labelled "off-policy Q-learning".

Footer, bottom left: "Slide adapted from Sergey Levine".

## Slide 16 — The plan for today

Same outline as slide 6, with progress shown by fading: the heading "Value-based RL methods" stays black; item "1. Q-learning RL method" and its sub-items "a. Policy iteration", "b. Bellman optimality equation" and "c. How to collect data for Q-learning methods" are faded to light grey; item "**2. Q-learning in practice**" is in bold black, with "d. Target networks", "e. Double DQN" and "f. N-step returns" in normal black. The "Key learning goals" block and the "Part of homework 2!" note of slide 6 are not on this page.

## Slide 17 — How to make Q-learning stable?

"full Q-learning with replay buffer:"

1. collect data $\lbrace (\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i) \rbrace$ using some policy, add it to $\mathcal{R}$ (first phrase in a light-blue highlight, $\mathcal{R}$ in light green)
2. sample a batch $(\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i)$ from $\mathcal{R}$
3. $\phi \leftarrow \phi - \alpha \sum_i \frac{dQ_\phi}{d\phi}(\mathbf{s}_ i, \mathbf{a}_ i) \left( Q_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - \left[ r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \max_{\mathbf{a}'} Q_\phi(\mathbf{s}'_ i, \mathbf{a}') \right] \right)$

Steps 2 and 3 sit in a pink band, with the same pair of green loop arrows as slide 15 (outer loop back to step 1, inner loop labelled $K \times$ back to step 2). A black curly brace under the bracketed target $\left[ r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \max_{\mathbf{a}'} Q_\phi(\mathbf{s}'_ i, \mathbf{a}') \right]$ is labelled "this is a moving target" and, in italics, "can lead to unstable optimization". (The "K = 1 is common" note and the replay-buffer diagram of slide 15 are not on this page.)

"Can we change the target Q-values more slowly?"

"Simple idea: freeze parameters used for the target Q-values, update periodically", preceded by a light-bulb emoji.

Footer, bottom left: "Slide adapted from Sergey Levine".

## Slide 18 — How to make Q-learning stable?

![Slide 18 — How to make Q-learning stable?](../images/06-q-learning/slide-18.jpg)

"Q-learning with replay buffer and target network:"

0. save target network parameters: $\phi' \leftarrow \phi$
1. collect data $\lbrace (\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i) \rbrace$ using some policy, add it to $\mathcal{R}$ (blue and green highlights as on slide 17)
2. sample a batch $(\mathbf{s}_ i, \mathbf{a}_ i, \mathbf{s}'_ i, r_i)$ from $\mathcal{R}$
3. $\phi \leftarrow \phi - \alpha \sum_i \frac{dQ_\phi}{d\phi}(\mathbf{s}_ i, \mathbf{a}_ i) \left( Q_\phi(\mathbf{s}_ i, \mathbf{a}_ i) - \left[ r(\mathbf{s}_ i, \mathbf{a}_ i) + \gamma \max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}'_ i, \mathbf{a}') \right] \right)$

Steps 2 and 3 are in the pink band. Three nested green loop arrows run down the left: the outermost, unlabelled, goes from step 3 back to step 0; a middle one labelled $N \times$ goes back to step 1; the innermost labelled $K \times$ goes back to step 2. A black square bracket at the right spans steps 1 to 3 and is labelled "Inner loop is doing supervised learning". The term $Q_{\phi'}(\mathbf{s}'_ i, \mathbf{a}')$ in step 3 is underlined in magenta with the magenta label "frozen target network". A black curly brace under the bracketed target is labelled "labels don't change in inner loop".

"Corresponds to DQN algorithm (“deep Q network”)"

In italics: "Mnih et al. Playing Atari with Deep Reinforcement Learning. 2013."

Bottom right: "Along with an actor-critic method, you'll implement this algorithm in HW 2!"

## Slide 19 — Are the Q-values accurate?

![Slide 19 — Are the Q-values accurate?](../images/06-q-learning/slide-19.jpg)

Title question "Are the Q-values accurate?". The slide shows figures of Atari training curves and Q-value plots, with the right-hand comment "As predicted Q increases, so does the return". Footer: "Slide adapted from Sergey Levine". The slide has two rows of figures.

Top row: four line charts in a row, each a single blue noisy curve. All four share the x-axis "Training Epochs", 0 to 100 in steps of 10.

- Panel "Average Reward on Breakout": y-axis "Average Reward per Episode", 0 to 250. One series (blue): near 0 at epoch 0, climbing to about 100 by epoch 40, with spikes to about 230 near epoch 52, hovering around 100–170 from epoch 60 to 100 and ending near 170.
- Panel "Average Reward on Seaquest": y-axis "Average Reward per Episode", 0 to 1800. One series (blue): roughly 200 at epoch 0, rising with large oscillations to about 1000–1600 between epochs 50 and 90, with deep dips (to about 0 near epoch 78 and to a few hundred near epochs 98–100) and a peak near 1700 at about epoch 95.
- Panel "Average Q on Breakout": y-axis "Average Action Value (Q)", 0 to 4. One series (blue): smooth, rising from 0 to about 1.5 at epoch 20, about 2 at epoch 25, about 2.5 at epoch 45, about 3 at epoch 60, about 3.4 at epoch 90 and about 3.9 at epoch 100.
- Panel "Average Q on Seaquest": y-axis "Average Action Value (Q)", 0 to 9. One series (blue): smooth and concave, rising from 0 to about 4 at epoch 20, about 5 at epoch 25, about 7 at epoch 50 and about 8.8 at epoch 100.

Bottom left, labelled "a": four Breakout game frames numbered with blue circles 1 to 4 (the Breakout screens show a rainbow brick wall, a ball with a dotted path and a red paddle at the bottom; the score strip reads, from the figure, 052, 056, 065, 072). Below them a light-blue area chart: y-axis "Value (V)", ticks 15 to 25; x-axis "Frame #", 0 to 120 in steps of 5. One series: starts near 16, drifts around 16–17 until about frame 40, with marker 1 on a small peak (about 17.3) at about frame 22 and marker 2 on a small peak (about 17) at about frame 38 followed by a sharp dip to about 15; then rises with fluctuations, marker 3 at about frame 85 (about 21.5), reaching a peak near 24 at about frame 105–108 (marker 4 at about frame 107), and ends near 22 at frame 120.

Bottom right, labelled "b": four frames numbered with black circles 1 to 4, each a brown playing field with a small orange bar at left, a small green bar at right and a white ball with a dashed trajectory line; a score strip at the top reads an orange "0" and a green "2" in each. Below each frame is a three-bar chart, y-axis "Action-Values (Q)", from -1 to 1, with the legend at right: light-blue "NO-OP", purple "UP", pink "DOWN". Frame 1: all three bars positive at about 0.75–0.8. Frame 2: NO-OP about -0.9, UP about 0.7, DOWN about -0.9. Frame 3: NO-OP about -1, UP about 0.8, DOWN about -1. Frame 4: all three bars positive and filling the chart to about 1.

## Slide 20 — Are the Q-values accurate?

![Slide 20 — Are the Q-values accurate?](../images/06-q-learning/slide-20.jpg)

No text other than the title and the footer "Slide adapted from Sergey Levine". The slide holds a figure of four line-chart panels titled "Alien", "Space Invaders", "Time Pilot" and "Zaxxon". The shared y-axis label (left of the first panel) is "Value estimates"; the shared x-axis label (under the last panel) is "Training steps (in millions)", ticks 0, 50, 100, 150, 200 in every panel. Four series are labelled to the right of the Zaxxon panel: "DQN estimate" (orange-red, noisy curve with a pale orange band), "Double DQN estimate" (blue, noisy curve with a pale blue band), "Double DQN true value" (a thin horizontal blue line) and "DQN true value" (a thin horizontal red line). Per panel, approximately:

- Alien (y ticks 10, 15, 20): DQN estimate shoots up to about 20 at about 30 million steps, then falls to about 14 around 100 million and stays about 14–15 to 200. Double DQN estimate climbs to about 10–11 within the first 20 million and stays there, about 11 at 200. Double DQN true value line at about 10; DQN true value line at about 8.8.
- Space Invaders (y ticks 4, 6, 8): DQN estimate rises from about 4 to about 6 at 20 million, about 8 at 100 million and about 9 at 200. Double DQN estimate rises to about 5.8 by 60 million and stays about 5.5–5.8. The two true-value lines lie almost on top of each other at about 4.9, the blue drawn over the red.
- Time Pilot (y ticks 1.0, 1.5, 2.0, 2.5): DQN estimate jumps to about 2.1, dips to about 1.7, then hovers about 2.0–2.3 to 200. Double DQN estimate rises steadily from about 0.7 to about 1.3 at 200. Double DQN true value line at about 1.3; DQN true value line at about 0.9.
- Zaxxon (y ticks 0, 2, 4, 6, 8): DQN estimate rises steeply to about 6 by about 40 million (with a wide pale band between about 2 and 6 around 40–60 million) and reaches about 7.5 at 200. Double DQN estimate stays near 0 until about 75 million, then rises to about 3 at 150–200 million. Double DQN true value line at about 1.2; DQN true value line at about 0.8.

## Slide 21 — Overestimation in Q-learning

$$\text{target value } y_j = r_j + \gamma \max_{\mathbf{a}'_ j} Q_{\phi'}(\mathbf{s}'_ j, \mathbf{a}'_ j)$$

A thin black arrow from the text "this last term is the problem" points at the $\max$ term.

The line $Q_{\phi'}(\mathbf{s}', \mathbf{a}')$ "is not perfect – it looks “noisy”"

"hence $\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}')$ *overestimates* the next value!"

"note that"

$$\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}') = Q_{\phi'}\left(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}')\right)$$

The outer $Q_{\phi'}$ is underlined, with the label "value *also* comes from $Q_{\phi'}$" beneath; the $\arg\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}')$ part is underlined, with the label "action selected according to $Q_{\phi'}$" beneath.

Footer: "Slide adapted from Sergey Levine".

## Slide 22 — Double Q-learning

The first lines repeat the end of slide 21: "note that $\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}') = Q_{\phi'}(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}'))$" with the same two underlines and the labels "value *also* comes from $Q_{\phi'}$" and "action selected according to $Q_{\phi'}$". Two thin black arrows point up from the text "if the noise in these is decorrelated, the problem goes away!" to the two labels.

"idea: don't use the same network to choose the action and evaluate value!"

"“double” Q-learning: use two networks:"

$$Q_{\phi_A}(\mathbf{s}, \mathbf{a}) \leftarrow r + \gamma Q_{\phi_B}\left(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi_A}(\mathbf{s}', \mathbf{a}')\right)$$

$$Q_{\phi_B}(\mathbf{s}, \mathbf{a}) \leftarrow r + \gamma Q_{\phi_A}\left(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi_B}(\mathbf{s}', \mathbf{a}')\right)$$

Two thin black arrows point up from the text "if the two Q's are noisy in *different* ways, there is no problem" to the outer $Q_{\phi_B}$ of the first line and the outer $Q_{\phi_A}$ of the second (the arrow tips sit under the outer-Q terms of the lower equation, one at its left end and one near the second line's $\arg\max$).

Footer: "Slide adapted from Sergey Levine".

## Slide 23 — Double Q-learning in practice

"where to get two Q-functions?"

"just use the current and target networks!"

"standard Q-learning:" $y = r + \gamma Q_{\phi'}\left(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi'}(\mathbf{s}', \mathbf{a}')\right)$

"double Q-learning:" $y = r + \gamma Q_{\phi'}\left(\mathbf{s}', \arg\max_{\mathbf{a}'} Q_{\phi}(\mathbf{s}', \mathbf{a}')\right)$

A green circle is drawn around the $\phi$ subscript of the inner $Q_\phi$ (the one that selects the action). Below, two lines: "just use current network (not target network) to evaluate action" and "still use target network to evaluate value!"

Footer: "Slide adapted from Sergey Levine".

## Slide 24 — N-step returns?

![Slide 24 — N-step returns?](../images/06-q-learning/slide-24.jpg)

"Q-learning target:" $y_{j,t} = r_{j,t} + \gamma \max_{\mathbf{a}_ {j,t+1}} Q_{\phi'}(\mathbf{s}_ {j,t+1}, \mathbf{a}_ {j,t+1})$

Two thin black arrows annotate it: the text "these are the only values that matter if $Q_{\phi'}$ is bad!" has an arrow pointing to the $r_{j,t}$ term; the text "these values are important if $Q_{\phi'}$ is good" has an arrow pointing to the $\gamma \max Q_{\phi'}$ term.

"**Recall**: fitting value functions with rewards + bootstrapped $V$"

Diagram at lower left: a blue dot (the start state) with five black wavy trajectory arrows fanning up and to the right. Three short coloured marks cross the fan: an orange line near the dot (one step out), a blue line further up (several steps out), and a long red measuring line running alongside the trajectories to their ends labelled in red "sum rewards"; a purple and a magenta bracket sit beside the red line near the bottom, marking the short and the middle spans. Labels:

- "*Monte Carlo:* using" $\sum_t^T r$ (the sum in red)
- "*Bootstrapped:* using" $r$ (magenta) $+ V$ (orange), at the bottom left under the dot
- "*How about*" $\sum_t^{t+n-1} r + V_{t+n}$ (sum and $r$ in purple, $V_{t+n}$ in blue), with two annotations to the right: "-> *less variance than MC*" and "-> *lower bias than 1-step bootstrap*"

"**Can we construct these multi-step targets for Q-learning?**"

$$y_{j,t} = \sum_{t'=t}^{t+N-1} \gamma^{t-t'} r_{j,t'} + \gamma^N \max_{\mathbf{a}_ {j,t+N}} Q_{\phi'}(\mathbf{s}_ {j,t+N}, \mathbf{a}_ {j,t+N})$$

**Printed slip:** the discount on the rewards is printed $\gamma^{t-t'}$, an exponent that is negative for every $t' \gt t$. Lecture 5's slides 12–13, which discount a reward $t'-t$ steps ahead, print $\gamma^{t'-t}$. The lecture reads the target aloud without the discounts (≈55:02). Slide 25 prints the same exponent.

## Slide 25 — N-step returns

"Q-learning target:" $y_{j,t} = r_{j,t} + \gamma \max_{\mathbf{a}_ {j,t+1}} Q_{\phi'}(\mathbf{s}_ {j,t+1}, \mathbf{a}_ {j,t+1})$ with the same two annotation arrows and texts as slide 24 ("these are the only values that matter if $Q_{\phi'}$ is bad!" and "these values are important if $Q_{\phi'}$ is good").

"**Can we construct these multi-step targets for Q-learning?**"

"N-step target:"

$$y_{j,t} = \sum_{t'=t}^{t+N-1} \gamma^{t-t'} r_{j,t'} + \gamma^N \max_{\mathbf{a}_ {j,t+N}} Q_{\phi'}(\mathbf{s}_ {j,t+N}, \mathbf{a}_ {j,t+N})$$

The sum of discounted rewards $\sum_{t'=t}^{t+N-1} \gamma^{t-t'} r_{j,t'}$ is in a pale pink highlight, labelled below in magenta "these are the rewards for policy that collected the data".

The exponent $\gamma^{t-t'}$ is printed as on slide 24 (see the slip noted there).

At right, three annotations: in green "+ (far) less biased target values when Q-values are inaccurate" and "+ typically faster learning, especially early on"; in red "- only actually correct when learning on-policy", then in black, right-aligned "(not an issue when N=1!)".

"**Ways to fix?**" "Most commonly: ignore the problem & still use N > 1". "Can also:" followed by the list:

- dynamically choose N to only use data that follows current policy (if data mostly on-policy, action space is small)
- use importance sampling

## Slide 26 — Q-learning in practice?

![Slide 26 — Q-learning in practice?](../images/06-q-learning/slide-26.jpg)

(No printed slide number; the label 26 is inferred from position.)

Two examples side by side.

Left heading: "Q-learning learns to outperform people on most Atari games", with "Mnih et al. 2015" at upper right of the chart. The figure is a horizontal bar chart of about 49 Atari games, sorted from highest to lowest DQN score. The x-axis runs 0 to 600 in steps of 100 then, after an axis break, 1,000 and 4,500%. The percentage printed at each bar's left end is the value for that game (the slide does not say what it is a percentage of); the legend at lower right gives light-blue "DQN" and grey "Best linear learner". A thin horizontal line, labelled above it "At human-level or above" and below it "Below human-level", sits between H.E.R.O. (76%) and Asterix (69%), so the games above it include several printed below 100% (Enduro 97% down to H.E.R.O. 76%); the slide does not state the threshold the line marks. Games and the printed percentages, top to bottom: Video Pinball 2539%, Boxing 1707%, Breakout 1327%, Star Gunner 598%, Robotank 508%, Atlantis 449%, Crazy Climber 419%, Gopher 400%, Demon Attack 294%, Name This Game 278%, Krull 277%, Assault 246%, Road Runner 232%, Kangaroo 224%, James Bond 145%, Tennis 143%, Pong 132%, Space Invaders 121%, Beam Rider 119%, Tutankham 112%, Kung-Fu Master 102%, Freeway 102%, Time Pilot 100%, Enduro 97%, Fishing Derby 93%, Up and Down 92%, Ice Hockey 79%, Q*bert 78%, H.E.R.O. 76%, Asterix 69%, Battle Zone 67%, Wizard of Wor 67%, Chopper Command 64%, Centipede 62%, Bank Heist 57%, River Raid 57%, Zaxxon 54%, Amidar 43%, Alien 42%, Venture 32%, Seaquest 25%, Double Dunk 17%, Bowling 14%, Ms. Pac-Man 13%, Asteroids 7%, Frostbite 6%, Gravitar 5%, Private Eye 2%, Montezuma's Revenge 0%. Light-blue bars are DQN and grey bars (best linear learner) sit behind or beside them; for Double Dunk the grey bar extends to about 175 while the blue bar is short. Error whiskers extend from many bars.

Right heading: "Q-learning trains robot grasping system", with "Kalashnikov et al. 2018" under it. Below it a photograph (appears to be a video still) of a row of robot stations, with grey robot arms with orange joints over the stations numbered 2, 3 and 4 on their bases (the arm at 2 is cut off at the left edge), each reaching over a bin; two red empty trays and two grey bins full of colourful toys alternate along the bottom. A white overlay title "Comparison" at the top and "1x" at the lower left.

Below the photograph, a table (its right-hand edge is clipped, so any further columns are not visible):

| Method | Dataset | Test |
| --- | --- | --- |
| QT-Opt (ours) | 580k off-policy + 28k on-policy | **96%** |
| Levine et al. [27] | 900k grasps from Levine et al. [27] | 78% |
| QT-Opt (ours) | 580k off-policy grasps only | 87% |
| Levine et al. [27] | 400k grasps from our dataset | 67% |

## Slide 27 — When to use one online RL algorithm vs. another?

(No printed slide number; the label 27 is inferred from position.)

Subtitle: "Chelsea's advice"

| Method | When |
| --- | --- |
| **PPO & variants** | When you care about stability, ease-of-use. When you don't care about data efficiency |
| **DQN & variants** | When you have discrete actions or low-dimensional continuous actions |
| **SAC & variants** | When you care most about data efficiency. When you are okay with tuning hyperparameters, less stability |

(Printed as three left-aligned rows of bold method names with the conditions beside them, not as a bordered table.)

## Slide 28 — The plan for today

Same as slide 6 with nothing faded: "Value-based RL methods"; "1. Q-learning RL method" with a. Policy iteration, b. Bellman optimality equation, c. How to collect data for Q-learning methods; "2. Q-learning in practice" with d. Target networks, e. Double DQN, f. N-step returns; then "**Key learning goals**:" "How Q-functions relate to policies", "How to do RL without learning an explicit policy", "How to stabilize Q-learning in practice". The "Part of homework 2!" note is not on this page.

## Slide 29 — Next time

"Done with online RL methods!"

"**Next week:**" "Offline RL" "(can we learn behavior from existing datasets?)" and, on the next line, "Reward learning + start of RL for LLMs".

Lower half, heading "Course reminders":

- Today
  - **Homework 1** due, **homework 2** out
  - Sending out 100 dollars of AWS credits + "form for GPU quota." (printed with a dollar sign and the figure 100; these words in a pale pink highlight)
  - Extra Q-learning section (1:30-2:30 pm)

At lower right, in red-orange bold "Very important" followed in black by "if you want GPUs, especially for default project" (as on slide 2).
