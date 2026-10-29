---
title: Lecture 8 — Conservative Offline RL and Reward Learning (slide deck)
lecture: 8
slides: 34
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf
numbering: printed number = PDF page number, 1:1 (slides 1–34). Pages 4, 8, 19, 27, 28, 29 and 30 print no number; their labels are inferred from position. Pages 2 and 34 print their number in the bottom-right corner.
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (9, 15, 23, 27, 30, 32) were checked by Claude Opus 5.5 against 1400px renders. Slides 9 (every table cell and the architecture's arrows) and 32 agree. Corrections: on slide 30, the dark-grey "With Chain of Thought" series' last three points were re-read; on slide 27, the green Heading curve's peak is inside the plot (the curve is cut off after it, not at it), and the road's lane count is not legible; on slide 15, the second classifier photograph shows the pencil case to the right of the notebook and further back, not in front of it; on slide 23, the green trajectory ends level with the star, well to its left. Slide 29's repeated sub-line was confirmed on the page.
images: 18 slides are rendered and shown under their headings (4, 5, 6, 8, 9, 12, 15, 17, 19, 20, 23, 25, 26, 27, 28, 29, 30, 32). Not rendered — slides 2 and 34, whose only pictures are photographs of the two guest lecturers; slide 13, whose embedded video frame is black; slides 7, 18 and 24, equation and text slides whose loop arrow, emoji and brace are transcribed here (24's content is repeated on 25, which is rendered); and the title, plan, summary and recap slides (1, 3, 10, 11, 14, 16, 21, 22, 31, 33).
---

# Lecture 8 — Conservative Offline RL and Reward Learning: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025, instructor Chelsea Finn) Lecture 8 deck, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

## Slide numbers vs PDF pages

Printed slide number equals PDF page number, 1:1, for all 34 pages. Pages 4, 8, 19, 27, 28, 29 and 30 print no number at all; their labels are inferred from their position between the numbered neighbours. Pages 2 and 34 print their number in the bottom-right corner rather than at the bottom centre; those numbers are real.

The deck's title slide reads "Reward Learning" with "Conservative Offline RL and" inserted above it; the course schedule and the video call the lecture "Reward Learning". The recording covers slides 3–32. It opens on the plan (slide 3) without discussing slide 2's reminders, and it ends at about 1:05:46 on slide 32's goal-setting game, without slides 33–34, so for slides 1–2 and 33–34 this file is the only source. **Slides 4–10 finish Lecture 7's offline RL material and teach conservative Q-learning (CQL) aloud**, which [Lecture 7's recording](../../wiki/07-offline-rl.md) skipped; slides 5–8 here match that deck's slides 25–28. Much of the CQL part and the goal-classifier discussion is worked live, with the lecturer writing or drawing as they talk: the IQL recap's example Q-values, the CQL objective written in terms of the critic's parameters ("the notation on the slides is slightly different", ≈13:52), a plot of a lower-bound Q-function, and a 2D picture of positive and negative goal states with a classifier's decision boundary. This KB has that work only as the transcript describes it.

Companion pages: [wiki page for this lecture](../../wiki/08-reward-learning.md) · [transcript](../transcripts/08-reward-learning.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–3 | Title, course reminders, plan for today |
| 4–10 | Offline RL recap and conservative methods: expectile loss and IQL, pushing down on large Q-values, conservative Q-learning (CQL), its algorithm and computation, LinkedIn application, summary |
| 11–13 | Plan, where the reward comes from, direct imitation learning |
| 14–21 | Learning rewards from example goals and behaviors: goal classifiers, adding visited states as negatives, robotic example, the GAN aside, recap of reward classifiers |
| 22–30 | Learning rewards from human preferences: feedback on roll-outs, the preference-learning objective and algorithm, human feedback for control and driving, RLHF for LLMs, AI feedback |
| 31–32 | Summary of reward learning, can RL agents propose their own goals |
| 33–34 | Plan for today (repeated), course reminders |

## Slide 1 — Conservative Offline RL and Reward Learning

Title slide. In pink at upper left: "Conservative Offline RL and", with a pink caret mark "^" below it pointing at the main title, which reads in black: "Reward Learning". Below: "CS 224R". No images.

## Slide 2 — Course reminders

Two columns.

Left, heading "Project":

- Fill out AWS form for GPU quota
- Proposal due today

Under the second bullet, in magenta: "graded fairly lightly- it's for your benefit!"

Right, heading "Homework":

- Homework 2 due next Friday. (start early!!)

Below, centred: "Guest lectures next week!" Two photographs with captions:

- Left, a photograph of Archit Sharma, smiling, against a rocky, tree-covered landscape. Text: "Archit Sharma", "Researcher on Gemini team", "Lead author of DPO", and underneath "Preference optimization for LLMs".
- Right, a photograph of Aviral Kumar in a grey collared shirt against a blurred building. Text: "Aviral Kumar", "Prof at CMU", "At forefront of LLM reasoning", and underneath "RL for LLM reasoning".

Bottom line: "Both very useful for default project." and, in bold, "Please join lecture in person!"

## Slide 3 — The plan for today

Numbered list:

1. Offline reinforcement learning: recap & conservative methods
2. Reward learning:
   - a. Where do rewards come from?
   - b. Learning rewards from example goals, behaviors
   - c. Learning rewards from human preferences

At right, in bold: "Part of HW3" beside item 1, and "Part of default project" beside item 2 (its sub-items a to c).

Below, "Key learning goals:" with two bullets:

- why task specification is hard (& why naïve methods fail)
- methods for learning rewards from human supervision

## Slide 4 — Recap: Offline RL

![Slide 4 — Recap: Offline RL](../images/08-reward-learning/slide-4.png)

"Problem: Can we learn a policy from a batch of existing data, without online data collection?"

"Batch of data collected by unknown" "behavior policy $\pi_\beta$," (in coral red) ", want to maximize rewards under $\pi_\theta$".

"Recall: Off-policy critic objective"

$$\min_\phi \sum_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left\lVert \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi_\theta(\cdot \mid \mathbf{s}')} \left[ \hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}', \mathbf{a}') \right] \right) \right\rVert^2$$

The subscript $\mathbf{a}' \sim \pi_\theta(\cdot \mid \mathbf{s}')$ and the bracketed term $\hat{Q}^{\pi_\theta}_ \phi(\mathbf{s}', \mathbf{a}')$ are in a light-blue highlight. Under it, in blue: "Querying Q-function on OOD actions causes overestimation."

Lower left: "Randomly init. $Q$-function for state $\mathbf{s}'$". A plot with a vertical axis labelled $Q(\mathbf{s}', \mathbf{a}')$ and a horizontal axis labelled "action $\mathbf{a}'$". One blue wiggly curve: it starts at the left slightly below the level of the vertical axis, rises to a small hump just right of the axis, falls to a dip, rises to a large hump further right, then declines steeply toward the right end. A green bracket on the horizontal axis, just right of the vertical axis and spanning the small hump and the dip, is labelled in green "data support".

Lower right: "IQL avoids queries beyond dataset actions altogether."

"Fit $V^\pi$ for policy $\pi$ better than $\pi_\beta$ using expectile loss:"

$$\hat{V}(\mathbf{s}) \leftarrow \arg\min_V \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim D} \left[ \ell_2^\lambda \left( V(\mathbf{s}) - \hat{Q}(\mathbf{s}, \mathbf{a}) \right) \right]$$

"Fit $Q$ using $V$:"

$$\hat{Q}(\mathbf{s}, \mathbf{a}) \leftarrow \arg\min_Q \mathbb{E}_ {(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r + \gamma \hat{V}(\mathbf{s}') \right) \right)^2 \right]$$

## Slide 5 — How to mitigate overestimation in offline RL?

![Slide 5 — How to mitigate overestimation in offline RL?](../images/08-reward-learning/slide-5.png)

"Recall: Randomly init. $Q$-function for state $\mathbf{s}'$". The same plot as slide 4, redrawn: vertical axis $Q(\mathbf{s}', \mathbf{a}')$, horizontal axis "action $\mathbf{a}'$", one blue wiggly curve (small hump, dip, large hump, then decline), a green bracket on the horizontal axis labelled "data support" (green).

At right: "Can we discourage overestimation?" and in smaller text "while still estimating value of the learned policy". Then "What if we just push down on large Q-values?"

$$\hat{Q}^\pi = \arg\min_Q \max_\mu \mathbb{E}_ {(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi} [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$$

The squared-error term (inside the bracket, from $Q(\mathbf{s}, \mathbf{a})$ through the inner expectation) is underlined in teal and labelled in teal "standard critic update". The final term $\alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$ is underlined in blue and labelled in blue "push down on large Q-values".

Below: "Can show that $\hat{Q}^\pi \leq Q^\pi$ for large enough $\alpha$".

Footer, bottom left: "Slide adapted from Sergey Levine".

## Slide 6 — How to mitigate overestimation in offline RL?

![Slide 6 — How to mitigate overestimation in offline RL?](../images/08-reward-learning/slide-6.png)

"Can we discourage overestimation?" and in smaller text "while still estimating value of the learned policy".

$$\hat{Q}^\pi = \arg\min_Q \max_\mu \mathbb{E}_ {(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\mathbf{a}' \sim \pi} [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$$

$$- \alpha \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim D} [Q(\mathbf{s}, \mathbf{a})]$$

The squared-error term is underlined in teal and labelled above it in teal "standard critic update". The $+\alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu}$ term is underlined in blue and labelled above it in blue "push down on large Q-values". The new second-line term $-\alpha \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim D}[Q(\mathbf{s}, \mathbf{a})]$ is underlined in magenta and labelled below it in magenta "push up on Q-values for $(\mathbf{s}, \mathbf{a})$ in the data".

Below:

"No longer guaranteed that $\hat{Q}^\pi \leq Q^\pi$ for all $(\mathbf{s}, \mathbf{a})$."

"BUT, guaranteed that $\mathbb{E}_ {\pi(\mathbf{a} \mid \mathbf{s})} [\hat{Q}^\pi(\mathbf{s}, \mathbf{a})] \leq \mathbb{E}_ {\pi(\mathbf{a} \mid \mathbf{s})} [Q^\pi(\mathbf{s}, \mathbf{a})]$ for all $\mathbf{s} \in D$."

In bold: "Conservative Q-learning (CQL)".

Footer: "Slide adapted from Sergey Levine".

## Slide 7 — How to mitigate overestimation in offline RL?

Bold: "Conservative Q-learning (CQL)"; below: "Full algorithm".

1. Update $\hat{Q}^\pi$ using $L_{CQL}$ using $D$
2. Update policy $\pi$

A curved black arrow runs from the left of item 2 up to item 1, showing the loop repeats.

"If actions are discrete:"

$$\pi(\mathbf{a} \mid \mathbf{s}) = \begin{cases} 1 & \text{if } \mathbf{a} = \arg\max_{\bar{\mathbf{a}}} \hat{Q}^\pi(\mathbf{s}, \bar{\mathbf{a}}) \cr 0 & \text{otherwise} \end{cases}$$

"If actions are continuous:"

$$\theta \leftarrow \theta + \eta \nabla_\theta \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \pi_\theta(\cdot \mid \mathbf{s})} \left[ \hat{Q}^\pi(\mathbf{s}, \mathbf{a}) \right]$$

Footer: "Slide adapted from Sergey Levine".

## Slide 8 — How to mitigate overestimation in offline RL?

![Slide 8 — How to mitigate overestimation in offline RL?](../images/08-reward-learning/slide-8.jpg)

Bold: "Conservative Q-learning (CQL)".

1. Update $\hat{Q}^\pi$ using $L_{CQL}$ using $D$
2. Update policy $\pi$

with the same curved arrow from item 2 back to item 1.

"How compute objective $L_{CQL}$?"

$$\hat{Q}^\pi = \arg\min_Q \max_\mu \mathbb{E}_ {(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim D} \left[ \left( Q(\mathbf{s}, \mathbf{a}) - \left( r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_ {\pi} [Q(\mathbf{s}', \mathbf{a}')] \right) \right)^2 \right] + \alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$$

$$- \alpha \mathbb{E}_ {(\mathbf{s}, \mathbf{a}) \sim D} [Q(\mathbf{s}, \mathbf{a})] + R(\mu)$$

A hand-drawn pink circle surrounds "max" and its subscript $\mu$. The term $\alpha \mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})]$ is underlined in blue. In the second line, $R(\mu)$ is underlined in green and labelled in green "regularizer".

"Common choice:" $R(\mu) = \mathbb{E}_ {\mathbf{s} \sim D} \left[ \mathcal{H}(\mu(\cdot \mid \mathbf{s})) \right]$

"With max entropy regularizer $R$, optimal $\mu(\mathbf{a} \mid \mathbf{s}) \propto \exp \left( Q(\mathbf{s}, \mathbf{a}) \right)$"

"Then:" $\mathbb{E}_ {\mathbf{s} \sim D, \mathbf{a} \sim \mu(\cdot \mid \mathbf{s})} [Q(\mathbf{s}, \mathbf{a})] = \log \sum_{\mathbf{a}} \exp \left( Q(\mathbf{s}, \mathbf{a}) \right)$, with the left-hand side underlined in blue.

"Don't need to construct $\mu$ directly."

At lower right, in blue inside a dotted blue box: "You will implement CQL in homework 3!"

Footer: "Slide adapted from Sergey Levine".

## Slide 9 — An example application

![Slide 9 — An example application](../images/08-reward-learning/slide-9.jpg)

"Optimizing policy for sending notifications to users on LinkedIn"

At left, a system-architecture diagram with small text. Two dashed boxes: the upper one labelled "Nearline Samza", the lower one labelled "Offline". Components, with arrows:

- Outside the Nearline box at upper left, a tan cylinder "Offline Features" feeds (arrow right) a green banner "Feature Push", which feeds a tan cylinder "Online Feature Store" inside the Nearline box.
- Inside Nearline: the Online Feature Store feeds a green box "Policy". A tan cylinder "Notification Queue" receives from a white box "Batch Notification Providers" (outside the dashed box, at right) and feeds a blue banner "Batch Notification" (arrow leftward), which feeds the Policy box. A pink/red banner "Near-Real-Time Notification" also feeds the Policy box, through a red arrow.
- The Policy box feeds a diamond "Send now?". From the diamond, "Yes" leads right to a blue person icon labelled "User"; "No" leads up to the Notification Queue (blue arrow labelled "No") and a red "No" arrow leads right to "Drop".
- The Policy box also sends an arrow down to a green box "Decision and Feature Snapshots" in the Offline box. Offline Features also feeds, via a long line down the left side, a green box "Offline Off-Policy Training" in the Offline box. Decision and Feature Snapshots feeds Offline Off-Policy Training (arrow leftward), and Offline Off-Policy Training sends a line up and right to the Policy box, labelled "Deploy new policy".

At right: "WAU: weekly active users" and "CTR: click-through-rate of notifications"; "Volume: total # of notifications".

Table (three columns):

| Metric | DDQN vs. Baseline | DDQN + CQL vs. Baseline |
| --- | --- | --- |
| Sessions | not stat sig | + 0.24% |
| WAU | -0.69% | + 0.18% |
| Volume | +7.72% | -1.73% |
| CTR | -7.79% | +2.26% |

Caption in bold serif: "Table 1: Online A/B test results for DDQN with and without CQL".

Citation: "Prabhakar, Yuan, Yang, Sun, Muralidharan. Multi-Objective Optimization of Notifications Using Offline RL. '22"

## Slide 10 — Summary

"Why offline RL?" (bold) "Online data is expensive." "Reusing offline data" (italic) "is good!"

"Key challenge: Overestimating Q-values because of shift between $\pi_\beta$ and $\pi_\theta$"

"Approaches:"

1. filtered or weighted imitation learning is a simple baseline
2. "implicitly constrain" (bold) the policy to $\pi_\beta$ by only supervising on actions in data
3. construct conservative objective by "penalizing Q-values" (bold)

Bottom: "Trajectory stitching allows offline RL methods to improve over imitation."

## Slide 11 — The plan for today

Numbered list:

1. Offline reinforcement learning: recap & conservative methods
2. Reward learning: (bold)
   - a. Where do rewards come from?
   - b. Learning rewards from example goals, behaviors
   - c. Learning rewards from human preferences

## Slide 12 — Where does the reward come from?

![Slide 12 — Where does the reward come from?](../images/08-reward-learning/slide-12.jpg)

Two groups of pictures.

Left, heading "Computer Games": a screenshot of the Atari Breakout game (black playfield with grey walls at the sides and top, a rainbow-coloured band of bricks (red, orange, yellow, green) with a blue row beneath, a small pink paddle at the bottom, a small ball). The score digits at the top are boxed in orange and labelled in orange "reward". Caption beneath: "Mnih et al. '15".

Right, heading "Real World Scenarios", three examples:

- "robotics": a photograph of a toddler with light hair at a wooden table, looking at a glass pitcher and a glass.
- "dialog": a clip-art picture of two speech bubbles, one blue and one orange, over a blue and an orange stick figure.
- "autonomous driving": an aerial photograph of a multi-lane motorway with a white car at the bottom, a green arrow curving from the car around a truck and onward ahead.

Beneath the dialog picture: "what is the" "reward" (orange) "?" and "often use a proxy".

Bottom: "Are there other easier ways of providing task supervision?" and "We've seen one alternative approach so far."

## Slide 13 — Where does the reward come from?

"Direct imitation learning:" (bold) "Mimic actions of expert"

In red, three sub-bullets:

- but no reasoning about outcomes or dynamics
- the expert might have different degrees of freedom
- might not be possible to provide demonstrations

"Can we reason about what an expert is trying to achieve?"

At lower right, an embedded video. The frame held on the page is entirely black, with the text "© Warneken & Tomasello" at its lower right. Nothing else is visible in this frame.

What the video showed is not on the page. In the recording (≈27:53) the lecturer points to "lots of fun videos" where "even very small kids are able to figure out what is the reward function, what is the objective" and optimize it "to help the person"; the recording does not describe this video further.

## Slide 14 — The plan for today

Numbered list, as on slide 3 without the side labels:

1. Offline reinforcement learning: recap & conservative methods
2. Reward learning:
   - a. Where do rewards come from?
   - b. Learning rewards from example goals, behaviors (bold)
   - c. Learning rewards from human preferences

## Slide 15 — Goal classifiers

![Slide 15 — Goal classifiers](../images/08-reward-learning/slide-15.jpg)

"Key idea: Learn to discern goal states from other states"

"Example task: put pencil case behind notebook"

Left, two groups of photographs of a robot workspace seen from above (a light wooden table with a green notebook, a striped pencil case and other objects, a robot arm with a gripper at the edge):

- "Positive examples" (green): two photographs in which the pencil case lies behind (above, in the image) the green notebook. In the second of them a yellow can also sits to the left of the notebook.
- "Negative examples" (red): two photographs in which the pencil case is not behind the notebook. In the first, the pencil case is at left beside the notebook, with a Rubik's cube above the notebook and a ball below it; in the second, the pencil case lies to the right and below the notebook, with a Rubik's cube near it.

Right, "Trained binary classifier": two photographs of the same workspace, each followed by an arrow. The first, showing the pencil case behind the notebook with the red robot arm reaching in, maps (arrow symbol) to "1". The second, showing the notebook at lower left and the pencil case to its right and further back, not behind it, maps to "0". Beneath: "Use output as reward signal."

Bottom, numbered list:

1. Collect examples of "successful" (green) & "unsuccessful" (red) states (states inside and outside of goal set $G$)
2. Train binary classifier. Beneath it, in parentheses: with inputs $s_i$ and labels $\mathbb{1}(s_i \in G)$.
3. Run RL with classifier as reward.

## Slide 16 — Goal classifiers

"Key idea: Learn to discern goal states from other states"

1. Collect examples of "successful" (green) & "unsuccessful" (red) states (states inside and outside of goal set $G$)
2. Train binary classifier. Beneath it, in parentheses: with inputs $s_i$ and labels $\mathbb{1}(s_i \in G)$.
3. Run RL with classifier as reward.

"What can go wrong?" (bold)

"The RL algorithm will seek out states that the classifier thinks are good."

"It may simply find states that the classifier wasn't trained on!"

"—> exploiting the classifiers weaknesses" (at lower right, as printed, with no apostrophe in "classifiers")

## Slide 17 — Goal classifiers

![Slide 17 — Goal classifiers](../images/08-reward-learning/slide-17.png)

"Can we prevent the RL algorithm from exploiting the classifier's weaknesses?"

"A proposition: Add states that RL visits as negative examples for the classifier"

"Specifically:" with a numbered list:

1. Collect initial set of "successful states $D_+$" (green) and "unsuccessful states $D_-$" (red).
2. Update classifier using $D_+$ (green) and $D_-$ (red) (balance dataset 50/50)
3. Collect experience $s_t, a_t, \ldots$ using policy $\pi$
4. Update policy $\pi$ using classifier-based reward.
5. Add visited states to negatives: $D_- \leftarrow D_- \cup \lbrace s_t \rbrace$ (the $D_-$ terms in red)

A curved black arrow on the left runs from item 5 up to item 2, showing that items 2 to 5 repeat.

"Why might this work or not work?" (bold) Three questions beneath, in two columns:

- "Will it learn an accurate classifier?"
- "What will the classifier output for successful states?"
- "Do we expect the policy to work?"

## Slide 18 — Goal classifiers

"A proposition: Add states that RL visits as negative examples for the classifier"

"Specifically:" with the same numbered list as slide 17 (items 1 to 5 as above), and the same curved arrow from item 5 to item 2.

"Why might this work or not work?" (bold)

- "Classifier can't be exploited." followed by a relieved-face emoji.
- "But, what if some of the visited states" "are" (italic) "successful?" followed by a flushed-face emoji.
- "As long as batches are" "balanced" (blue) ", classifier will output p≥0.5 for successful states." followed by a grinning-face-with-sweat emoji.

## Slide 19 — Goal classifiers for robotic reinforcement learning

![Slide 19 — Goal classifiers for robotic reinforcement learning](../images/08-reward-learning/slide-19.jpg)

"Collect 50 demonstrations:"

- use final states as "success state examples" (green)
- initialize RL replay buffer with demos

At upper right: "Important to regularize the classifier!"

Two video frames side by side, each a photograph of a robot workspace seen from the front: a dark enclosure with a pale-grey table, a white robot arm with a black gripper above it, a metal stand (a hook-shaped pole on a round base) in the centre-right, and a lime-green cloth on the table. The word "Nest" is printed at the top right of each frame.

- Left, headed "Directly imitating demos:": the cloth lies spread out and crumpled at the left of the stand, with the gripper just above it. Caption: "26% success rate".
- Right, headed "RL policy trained with learned classifier:": the cloth is gathered and hangs more compactly near the gripper at the left of the stand. Caption: "62% success rate".

(The page holds one frame for each video; what the videos showed is not on the page, and the recording (≈41:02–41:49) gives only the two success rates.)

Citation at the bottom: "Sharma, Ahmed, Ahmad, Finn. Self-Improving Robots: End-to-End Autonomous Visuomotor Reinforcement Learning. '23"

## Slide 20 — Aside: Generative adversarial networks

![Slide 20 — Aside: Generative adversarial networks](../images/08-reward-learning/slide-20.jpg)

"This is also how GANs work!"

1. Train classifier to discriminate between "real data" (green) and "generated data" (red)
2. Train generator to generate data that the classifier thinks is real.

Left, heading: "“Yorkshire terrier” generated by VQ-GAN". Three images of Yorkshire terrier dogs: a dog lying and looking at the camera against a white background; a dog with a flying topknot running on a beach under a blue sky; a close-up of a small dog with its mouth open and a blue collar, with a blue pool and a grass strip behind. Citation under them: "Yu et al. ViT-VQGAN. ICLR '22."

Right, heading: "Phenaki video generation (uses GAN-loss)". One frame showing a figure in a white spacesuit walking in a reddish-brown landscape, reflected in a shallow puddle. Beside it, the prompt text: "Side view of an astronaut is walking through a puddle on mars. The astronaut is dancing on mars. The astronaut walks his dog on mars. The astronaut and his dog watch fireworks." Citation: "Villegas et al. Phenaki. ICLR '23."

The Phenaki frame is from a video. In the recording (≈42:36–43:23) the lecturer calls it "a cool video of an astronaut dancing on Mars and later walking his dog on Mars", generated with a GAN loss.

Bottom: "At convergence: generator will match data distribution p(x)"

## Slide 21 — Recap of reward classifiers

- "Pre-trained classifiers can be" "exploited" (red) "when optimized against."
- "Solution" (green): "Update the classifier" "during RL" (italic) ", using policy data as negatives"
- "Can learn goal classifier with success examples, full reward with demos."

Then three lines:

- "+ practical framework for task specification" (green)
- "~ adversarial training can be unstable" (orange), with a grey sub-line "(though variety of regularization tricks from GAN literature)"
- "- requires examples of desired behavior or outcomes" (red)

## Slide 22 — The plan for today

Numbered list, as on slide 3 without the side labels:

1. Offline reinforcement learning: recap & conservative methods
2. Reward learning:
   - a. Where do rewards come from?
   - b. Learning rewards from example goals, behaviors
   - c. Learning rewards from human preferences (bold)

## Slide 23 — Can humans provide feedback on policy roll-outs?

![Slide 23 — Can humans provide feedback on policy roll-outs?](../images/08-reward-learning/slide-23.png)

"instead of requiring demos or example goals" "(or in addition to!)"

"A couple options"

Two panels.

- Left, "“How good is this trajectory?”": a blue dot (start) at lower left with one wavy magenta curve running up and to the right, ending with an arrowhead near a yellow star (goal) at upper right.
- Right, inside a rounded grey box, "“Which trajectory is better?”": a blue dot at lower left from which three curves leave: a green curve that rises steeply and ends with an arrowhead well to the left of the star, at about its height; a blue curve that rises more gently and ends with an arrowhead just left of the star; and a wavy magenta curve ending with an arrowhead at the star. The yellow star is at upper right.

Beneath the right panel: "Relative" "preferences" (bold) "are easier to provide!"

## Slide 24 — How to learn a reward function from human preferences?

"Human says $\tau_w$ is better than $\tau_l$. (can write it as $\tau_w \succ \tau_l$)"

In a grey box at upper right: "Note: $\tau$ could be a" "full" (italic) "or" "partial" (italic) "roll-out!"

"We want a reward $r_\theta$ such that"

$$\sum_{(s, a) \in \tau_w} r_\theta(s, a) > \sum_{(s, a) \in \tau_l} r_\theta(s, a).$$

A curly brace under the left sum is labelled "shorthand $r_\theta(\tau_w)$".

"Humans are classifying which trajectory is better."

"Reward should be discriminative as well."

## Slide 25 — How to learn a reward function from human preferences?

![Slide 25 — How to learn a reward function from human preferences?](../images/08-reward-learning/slide-25.png)

The content of slide 24 (the "Human says" line, the grey note box, the "We want a reward" inequality with its "shorthand $r_\theta(\tau_w)$" brace) is repeated, and two lines are added.

"We can define $\sigma\left( r_\theta(\tau_a) - r_\theta(\tau_b) \right)$ to be the estimated probability that $\tau_a \succ \tau_b$."

"Then, we can maximize log probability:"

$$\max_\theta \mathbb{E}_ {\tau_w, \tau_l} \left[ \log \sigma \left( r_\theta(\tau_w) - r_\theta(\tau_l) \right) \right]$$

## Slide 26 — How to learn a reward function from human preferences?

![Slide 26 — How to learn a reward function from human preferences?](../images/08-reward-learning/slide-26.png)

"Complete reward learning algorithm" (bold)

1. Given dataset $\lbrace \tau_i \rbrace$, sample batches of $k$ trajectories and ask humans to rank. (for LLMs, these $k$ trajectories all have the same prompt)
2. Compute $r_\theta(\tau_1), \ldots, r_\theta(\tau_k)$ under current reward model $r_\theta$
3. For all $\binom{k}{2}$ pairs per batch, compute $\nabla_\theta \mathbb{E}_ {\tau_w, \tau_l} \left[ \log \sigma \left( r_\theta(\tau_w) - r_\theta(\tau_l) \right) \right]$ where $\tau_w \succ \tau_l$
4. Update $\theta$ using computed gradient.

A curved black arrow on the left runs from item 4 up to item 2, showing that items 2 to 4 repeat.

Bottom: "Note: This can be done in the loop of online RL."

## Slide 27 — Learning rewards from human feedback

![Slide 27 — Learning rewards from human feedback](../images/08-reward-learning/slide-27.jpg)

Two panels.

Left, "Learning rewards in the loop of online RL" and "uses 900 human preference queries": a simulation screenshot of a jointed brown robot (a few cylindrical limbs forming a hopper-like body) standing on a black-and-white checkerboard floor under a dark blue-grey sky.

Right, "Learning rewards for driving" and "to weight different factors": a line chart and a driving diagram.

Chart: horizontal axis from -1.0 to 1.0 (ticks at -1.0, -0.5, 0.0, 0.5, 1.0), no axis title; vertical axis labelled "PDF", from 0.0 to 4.0 (ticks every 0.5). Five bell-shaped curves, each with a dashed vertical line of the same colour at its peak. The legend, at upper right of the chart:

- blue: $w_1$ for Road Boundary. Peak about 1.7, at about -0.1.
- red: $w_2$ for Staying within Lanes. Peak about 2.3, at about 0.15.
- light blue: $w_3$ for Keeping Speed. Narrow peak about 3.5, at about -0.08.
- green: $w_4$ for Heading. Peak about 2.3, at about 0.87; the curve then falls to about 1.0 at the plot's right edge (1.0), where it is cut off.
- purple: $w_5$ for Collision Avoidance. Peak about 1.4, at about -0.48.

Below the chart: a small top-down diagram of a straight road with grass on both sides (the lane markings are too small to count), a white car and an orange car side by side near the bottom, and a white dotted path ahead of the white car, curving slightly, and a red path ahead of the orange car.

The left screenshot is from a video. In the recording (≈56:34–57:20) the lecturer says the simulated agent learns to do a backflip, for which it is "very hard to write a reward function", from 900 human preference queries asked in the loop of RL, learning a reward function for "a human preferred backflip" in the process.

Citations at the bottom: "Christiano et al. Deep Reinforcement Learning from Human Preferences. NeurIPS 2017." and "Sadigh et al. Active Preference-Based Learning of Reward Functions. RSS 2017."

## Slide 28 — Learning rewards from human feedback for LLMs

![Slide 28 — Learning rewards from human feedback for LLMs](../images/08-reward-learning/slide-28.jpg)

"Prompt x" (the text prints a plain "x"). A screenshot of a dark chat box with the prompt "How do I cook an omelette?" preceded by the label "x :".

"Sample replies": two speech-bubble icons. The left one, containing text lines and the letter "y", has a gold trophy at its lower left. The right one is labelled "y'". Beneath the left icon, a dark screenshot of a long, small-print reply: "Cooking an omelette is quick and easy! Here's a simple step-by-step guide for a classic 2-egg omelette:" followed by an "Ingredients:" list (2 eggs; Salt and pepper (to taste); 1-2 tbsp milk or water (optional, for fluffiness); 1 tsp butter or oil; Optional fillings: cheese, chopped veggies, ham, herbs, etc.) and an "Instructions:" list of seven numbered steps (1. Crack the eggs into a bowl ... 7. Slide it onto a plate and serve hot!). Beneath the right icon, a shorter dark screenshot of a one-paragraph reply that begins "Sure! To cook an omelet, start by whisking a couple of eggs with a pinch of salt and pepper." The body text of both screenshots is too small to read in full.

"Ask human which response is better" and "y ≻ y'"

Bottom: "Train reward model $r(x, y)$ that judges how good a response $y$ is for a prompt $x$."

## Slide 29 — Learning rewards from human feedback for LLMs

![Slide 29 — Learning rewards from human feedback for LLMs](../images/08-reward-learning/slide-29.png)

Left, three stacked rounded boxes:

1. Light blue: "1. Large-scale pre-training" with "next token prediction, mixed quality"
2. Teal: "2. Supervised fine-tuning" with "on higher-quality (prompt, response)"
3. Pink: "3. RL from human feedback" with "on higher-quality (prompt, response)" (as printed, the same sub-line as box 2, apparently a copy slip; the lecturer describes this step as fine-tuning "on higher quality preference data", ≈59:38)

Right, three steps:

- "3a. Gather preference data." (bold): "Prompt x" with a chat-box screenshot "x : How do I cook an omelette?", and "Sample replies" with two speech-bubble icons "y" (with a trophy at its lower left) and "y'", and beneath them "y ≻ y'".
- "3b. Train reward model." (bold): "Train reward model $r(x, y)$ that judges how good a response $y$ is for a prompt $x$."
- "3c. Reinforcement learning to maximize reward." (bold): "Finetune model to predict $y$ for prompt $x$ with high reward"

## Slide 30 — Learning rewards from AI feedback

![Slide 30 — Learning rewards from AI feedback](../images/08-reward-learning/slide-30.jpg)

Title with the word "human" struck through: "Learning rewards from ~~human~~ AI feedback".

"Reinforcement learning with AI feedback (RLAIF)"

"Ask another language model “which of these responses is less harmful?”"

"Key insight: critique is easier than generation!"

Chart (from the cited paper). Horizontal axis "Helpfulness Elo", from about -150 to 150 (ticks -150, -100, -50, 0, 50, 100, 150); vertical axis "Harmlessness Elo", from -100 to 200 (ticks every 50). Several coloured line series with dots, plus two marked points and two arcs:

- Red cross labelled "Pretrained Base" (red text) at about (-145, -95), the common start of the blue and orange series.
- Blue series labelled "Helpful-Only" (blue): from the Pretrained Base to about (65, -45), then (87, -10), then (120, -62), then (147, -47).
- Orange series labelled "Helpful + Harmless" (orange): from the Pretrained Base to about (12, 33), then (75, 77), then (100, 30), then (103, -5).
- Green X labelled "Constitutional SL" (green) at about (0, 0), the start of the grey series.
- Dark grey series labelled "With Chain of Thought" (grey): from Constitutional SL to about (43, 52), then (49, 114), then (78, 151), then (84, 191).
- Light grey series, with no label of its own: from Constitutional SL to about (57, 60), then (103, 97), then (108, 120), then (110, 142).
- A black arc labelled "Standard RLHF" (bold black) curving over the lower-right region, from about (45, 90) down to about (150, -65).
- A magenta arc labelled "Constitutional RL (Pareto Improvement)" (bold magenta, with "(Pareto Improvement)" in italics), lying above and to the right of the black arc, from about (45, 200) to about (125, 95).

The black and magenta arcs are drawn over the chart to show the frontier of the methods; the magenta one lies beyond the black one.

Citation: "Bai et al. Constitutional AI: Harmlessness from AI Feedback. 2022."

## Slide 31 — Summary of Reward Learning

"#1 Takeaway:" (bold) "Rewards can't be taken for granted!"

Two columns.

Left, "Learning rewards from goals, demos":

- "+ practical framework for task specification" (green)
- "~ adversarial training can be unstable" (orange), with a grey sub-line "(though variety of regularization tricks from GAN literature)"
- "- requires examples of desired behavior or outcomes" (red)

Right, "Learning rewards from human preferences":

- "+ pairwise preferences easy to provide" (green), with a grey sub-line "(doesn't require example goals, demos!)"
- "+ has been deployed at scale!" (green)
- "- may require supervision in the loop of RL" (red), with a grey sub-line "(usually requires more human time)"

Bottom: "Thought exercise:" (bold) "Are there other forms of feedback, supervision that might be helpful?"

## Slide 32 — Can RL agents propose their own goals?

![Slide 32 — Can RL agents propose their own goals?](../images/08-reward-learning/slide-32.jpg)

"Entire area of “unsupervised” RL"

"One example:" (bold) "Formulate two-player game, with a goal-setter and a goal reacher"

Diagram from the cited paper, three grid-world panels (each a grid of grey cells with a vertical orange brick wall in the middle, interrupted by a brown arched door at the wall's middle row; a golden key near the bottom of the left side of the wall; a light bulb to the right of the wall):

- A bracket above the first two panels is labelled "Self Play Episode (no supervision -- internal reward only)". The first panel is captioned "Alice's turn" and a black arrow points from it to the second panel, captioned "Bob's turn".
- Alice's turn (red arrows): an "Alice" face icon at upper left. A red path runs straight down from Alice to the key at lower left, then in steps up and right to the door, through it, up and right to the light bulb, then up and right to a "STOP" label at the top right.
- Bob's turn (blue arrows): a "Bob" face icon at the top right. A blue path runs left from Bob and down to the light bulb, then left through the door, down to the key, and then a blue arrow runs up the far left of the grid.
- Right panel, a bracket label "Target Task Episode (supervision from external reward)", captioned "Bob applied to target task": a "Bob" face icon at left of centre. A blue path runs down and right to the key, then up and right through the door, to the light bulb, then down and right to a chequered flag at the lower right.

Citation: "Sukhbaatar et al. Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play. ICLR 2018"

## Slide 33 — The plan for today

Numbered list, as on slide 3 without the side labels:

1. Offline reinforcement learning: recap & conservative methods
2. Reward learning:
   - a. Where do rewards come from?
   - b. Learning rewards from example goals, behaviors
   - c. Learning rewards from human preferences

"Key learning goals:" with two bullets:

- why task specification is hard (& why naïve methods fail)
- methods for learning rewards from human supervision

## Slide 34 — Course reminders

The content is identical to slide 2: Project (Fill out AWS form for GPU quota; Proposal due today; in magenta "graded fairly lightly- it's for your benefit!"), Homework (Homework 2 due next Friday. (start early!!)), "Guest lectures next week!", the photograph and caption for "Archit Sharma" (Researcher on Gemini team; Lead author of DPO; Preference optimization for LLMs) and for "Aviral Kumar" (Prof at CMU; At forefront of LLM reasoning; RL for LLM reasoning), and the bottom line "Both very useful for default project." with, in bold, "Please join lecture in person!"
