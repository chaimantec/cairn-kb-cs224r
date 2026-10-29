---
title: "Lecture 10 — Reinforcement Learning for LLM Reasoning (slide deck)"
lecture: 10
slides: 41
printed_numbers: 1–41
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf
numbering: 1:1 with PDF pages, printed bottom right; pages 6, 31, 32, 35, 36 and 40 print no number in the text layer, labels inferred from position, unambiguous
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (11, 17, 23, 27, 35, 40) were checked by Claude Opus 5.5 against 1400px renders, with slide 19 for its printed title. Slides 11 (every series and value), 17 (every Q-value, advantage and arrow), 23, 40 (every bar) agree, and slide 27 agrees but for where one arrow starts. Correction: on slide 35, the first chart's early points had been placed one step late; the Verifications and Backtracking values are re-read at every labelled step.
images: 27 slides are rendered and shown under their headings (3, 4, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 34, 35, 36, 39, 40, 41). Not rendered: panels of text transcribed here in full (2, 5, 31); slide 26, a text pop-up over a faded copy of the earlier slides; the build steps 33 (slide 34 adds its constraint), 37 and 38 (slide 39 shows both diagrams); and the title, outline, section-divider and text slides (1, 6, 8, 13, 29, 30, 32). Slides 10, 19 and 23 hold only the first frame of an animation.
---

# Lecture 10 — Reinforcement Learning for LLM Reasoning: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025) Lecture 10 deck, "Reinforcement Learning for LLM Reasoning", given by guest lecturer Aviral Kumar (Carnegie Mellon University, School of Computer Science), transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

The recording (about 70 minutes) follows the deck from slide 2 through the online-RL slides (24–27) and then jumps: at ≈1:06:56 Kumar says "I'm going to skip some of these things" and turns to the thinking models of Part 2, discussing slide 32's "what changed" from ≈1:07:42 until the recording ends at about 1:10:30. Slides 28 (solving hard questions), 29 (Part 1 takeaways), 30 (the Part 2 divider), 31 (an example thinking trace) and 33–41 (the formulation, meta strategies, response length, test-time scaling, the leaderboard results and the open questions) are not discussed aloud, so for them this file is the only source. Slide 14's apples-and-pears example and slide 26's pop-up on the optimal rollout policy are shown but not read out; Kumar mentions an exact expression for the best rollout policy at ≈53:50 without covering it. Kumar works three things on a board the recording does not show: the language MDP (≈10:03–10:50 and ≈46:58), the rejection finetuning algorithm (≈13:54–18:29) and the per-step DPO algorithm (≈55:22–56:09). At ≈58:27 Kumar says the result plots "were supposed to be gifs": the PDF holds only their first frame, which is why slides 10, 19 and 23 show a few points at the left edge and no curves.

## Slide numbers vs PDF pages

The deck is 1:1 with its printed numbers: PDF page N is printed slide N, for all 41 pages, and the number prints in the bottom-right corner. Six pages (6, 31, 32, 35, 36 and 40) print no number in the text layer; their labels follow the 1:1 sequence and are unambiguous.

| PDF page | Printed slide |
| -------- | ------------- |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 (inferred from position) |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |
| 10 | 10 |
| 11 | 11 |
| 12 | 12 |
| 13 | 13 |
| 14 | 14 |
| 15 | 15 |
| 16 | 16 |
| 17 | 17 |
| 18 | 18 |
| 19 | 19 |
| 20 | 20 |
| 21 | 21 |
| 22 | 22 |
| 23 | 23 |
| 24 | 24 |
| 25 | 25 |
| 26 | 26 |
| 27 | 27 |
| 28 | 28 |
| 29 | 29 |
| 30 | 30 |
| 31 | 31 (inferred from position) |
| 32 | 32 (inferred from position) |
| 33 | 33 |
| 34 | 34 |
| 35 | 35 (inferred from position) |
| 36 | 36 (inferred from position) |
| 37 | 37 |
| 38 | 38 |
| 39 | 39 |
| 40 | 40 (inferred from position) |
| 41 | 41 |

## Sections

| Section | Slides |
| ------- | ------ |
| Introduction: setup (solving math problems), conventional next-token training, limited data, why current training is not enough | 1–5 |
| Outline | 6 |
| Problem setup (reasoning as a sparse-reward MDP) | 7 |
| Part 1, Classical RL Methods for Reasoning: divider and main papers | 8 |
| Part 1: data-scaling analysis (SFT, RFT, RL axes; why SFT on on-policy solutions degrades) | 9–12 |
| Part 1: addressing spurious steps with step-level credit | 13–14 |
| Part 1: negative on-policy data, advantages, and filtering steps by advantage | 15–19 |
| Part 1: offline RL (DPO with per-step preference pairs) and its scaling results | 20–23 |
| Part 1: online RL, process advantage verifiers (PAVs) and their results | 24–28 |
| Part 1: takeaways | 29 |
| Part 2, Training "Thinking" Models via RL: divider and main papers | 30 |
| Part 2: an example thinking trace and what changed | 31–32 |
| Part 2: formulation (test-compute-constrained objective) | 33–34 |
| Part 2: meta strategies in the action space | 35 |
| Part 2: RL amplifies response length | 36 |
| Part 2: test-time scaling | 37–39 |
| Part 2: results on leaderboards | 40 |
| Open questions | 41 |

## Slide 1 — Reinforcement Learning for LLM Reasoning

Title slide. "Reinforcement Learning for LLM Reasoning" in blue; "Aviral Kumar" below; the Carnegie Mellon University School of Computer Science logo at the bottom.

## Slide 2 — Setup for Today: Solving Math Problems!

The slide shows an AIME problem with two worked solutions. Footer: "Source: AIME questions, 2025."

**Problem 1.** Find the sum of all integer bases $b \gt 9$ for which $17_b$ is a divisor of $97_b$.

**Solution 1 (thorough).** We are tasked with finding the number of integer bases $b \gt 9$ such that $\frac{9b+7}{b+7} \in \mathbf{Z}$. Notice that

$$\frac{9b+7}{b+7} = \frac{9b+63-56}{b+7} = \frac{9(b+7)-56}{b+7} = 9 - \frac{56}{b+7}$$

so we need only $\frac{56}{b+7} \in \mathbf{Z}$. Then $b+7$ is a factor of $56$.

The factors of $56$ are $1, 2, 4, 7, 8, 14, 28, 56$. Of these, only $8, 14, 28, 56$ produce a positive $b$, namely $b = 1, 7, 21, 49$ respectively. However, we are given that $b \gt 9$, so only $b = 21, 49$ are solutions. Thus the answer is $21 + 49 =$ boxed 070. ~eevee9406

**Solution 2 (quick).** We have, $b + 7 \mid 9b + 7$ meaning $b + 7 \mid -56$ so taking divisors of $56$ under bounds to find $b = 49, 21$ meaning our answer is $49 + 21 =$ boxed 070.

## Slide 3 — The Conventional Way of Training

![Slide 3 — The Conventional Way of Training](../images/10-rl-for-reasoning/slide-3.png)

Left: a diagram. A row of seven peach input boxes at the bottom, labelled "text, image, video, audio tokens" with a double-headed horizontal arrow beneath. Seven blue arrows go up into a dashed grey rounded box "Foundation model (transformer)". Four blue arrows go up from it to a row of four green output boxes, labelled in red "predict the next token" with a double-headed horizontal arrow above.

Right: an orange box with the question "Why does next token prediction work?" Below it the equation

$$\left| \hat{p}_ \theta(\mathbf{y} \mid \mathbf{x}) - p^\ast(\mathbf{y} \mid \mathbf{x}) \right| \propto \frac{1}{|\mathcal{D}(\mathbf{y} \mid \mathbf{x})|^\alpha}$$

with blue labels and arrows: "ground-truth model" points at $p^\ast(\mathbf{y} \mid \mathbf{x})$, "Learned model" points at $\hat{p}_ \theta(\mathbf{y} \mid \mathbf{x})$, and "Error reduces with" followed in red by "more data similar to the target x" points at the denominator $|\mathcal{D}(\mathbf{y} \mid \mathbf{x})|^\alpha$.

## Slide 4 — But in Many Problems, Data is Limited/Biased…

![Slide 4 — But in Many Problems, Data is Limited/Biased…](../images/10-rl-for-reasoning/slide-4.jpg)

Two dashed grey panels side by side, each with a red-outlined heading and an orange caption at the bottom.

Left panel, "Math reasoning": a line chart. The y-axis is "Effective stock (number of tokens)" on a log scale from $10^{11}$ to $10^{15}$; the x-axis is "Year", 2020 to 2034. The legend lists four items: "Stock of data" (teal solid line), "Median date of full stock utilization" (pink dashed vertical line), "Dataset size projection" (blue solid line), "Median date of full stock utilization (5x overtraining)" (purple dashed vertical line). Filled grey-blue dots mark named models: GPT-3 (about $4 \times 10^{11}$, 2020), FLAN (about $2 \times 10^{12}$, 2021), PaLM (about $6 \times 10^{11}$, 2022), Falcon-180B (about $3 \times 10^{12}$, 2023), DBRX and Llama 3 (about $1$ to $2 \times 10^{13}$, 2024). The teal stock line rises gently from about $2.5 \times 10^{14}$ in 2020 to about $7 \times 10^{14}$ in 2034, inside a teal band from about $10^{14}$ to $3 \times 10^{15}$. The blue dataset-size projection rises steeply from about $6 \times 10^{11}$ in 2020, crossing the stock line around 2027, and flattens near $7 \times 10^{14}$, inside a blue band. The vertical dashed lines sit at about 2027.5 (purple) and 2028.6 (pink). Caption: "It is estimated we will run out of high-quality Internet text by 2028".

Right panel, "Embodied AI": two pictures from the DROID dataset. The left one is headed "DROID, Distributed Robot Interaction Dataset" with the list "76k Episodes", "564 Scenes", "52 Buildings", "13 Institutions", "86 Tasks / Verbs". The right one is a coloured-border infographic: "1M Episodes from 311 Scenes", "34 Research Labs across 21 Institutions", "22 Embodiments" (a row of robot-arm pictures), "527 Skills" (three photos labelled pour, stack, route), "60 Datasets" (three photos), "1,798 Attributes · 5,228 Objects · 23,486 Spatial Relations". Caption: "Real-world is stochastic, this is little data compared to LLMs!"

## Slide 5 — And Training on Current Data is Not Enough

Left, in a dashed grey box, a problem: "Show that the inequality $\sum_{i=1}^{n} \sum_{j=1}^{n} \sqrt{|x_i - x_j|} \leq \sum_{i=1}^{n} \sum_{j=1}^{n} \sqrt{|x_i + x_j|}$ holds for all real numbers $x_1, x_2, \ldots, x_n$."

Right, a model solution beginning with a red-boxed heading "Analyzing $\sqrt{|x_i - x_j|}$ and $\sqrt{|x_i + x_j|}$":

1. **Absolute Value Cases:** When $x_i = x_j$, $\sqrt{|x_i - x_j|} = \sqrt{0} = 0$ and $\sqrt{|x_i + x_j|} = \sqrt{|2x_i|}$. When $x_i \neq x_j$, we need to analyze how $|x_i - x_j|$ compares to $|x_i + x_j|$.
2. **Comparing $|x_i - x_j|$ and $|x_i + x_j|$:** Consider two cases for $(x_i, x_j)$.
   - **Case 1: $x_i \geq 0$ and $x_j \geq 0$:** Here both $x_i$ and $x_j$ are non-negative. We have $|x_i - x_j| = x_i - x_j$ and $|x_i + x_j| = x_i + x_j$. Clearly, $x_i + x_j \geq x_i - x_j$ if $x_i \geq x_j$ or $x_j \geq x_i$.
   - **Case 2: $x_i \leq 0$ and $x_j \leq 0$:** Both are non-positive. We have $|x_i - x_j| = -(x_i - x_j) = x_j - x_i$ and $|x_i + x_j| = -(x_i + x_j) = -(x_i + x_j)$. Similarly, $|x_i + x_j|$ would be $|-x_i - x_j|$ and the same argument holds.
   - **Case 3: $x_i \geq 0$ and $x_j \leq 0$ or vice versa:** (red-boxed) This makes $(x_i - x_j)$ and $(x_i + x_j)$ different in signs or their modulus. The absolute values may change based on these signs but we still have the result as: $\sqrt{|x_i + x_j|} \geq \sqrt{|x_i - x_j|}$.

Two pink annotation boxes with red arrows: "Analyzing individual terms is not good enough" points at the red-boxed heading, and "The model "**asserts**" its way out, no realization of its mistake / backtracking" points at the Case 3 box.

## Slide 6 — Outline: RL for Reasoning

A horizontal blue double-headed arrow labelled "time" at the right end, with a red tick mark near the middle labelled "DeepSeek-R1 / "thinking" models". Below are two dashed grey boxes with orange headings.

Left box, "Classical RL techniques for LLM reasoning": the list 1. Imitation Learning, 2. Offline reinforcement learning, 3. Online reinforcement learning. Then in red "Takeaway:" "Training with RL can help improve efficiency of learning!"

Right box, "Modern extensions of these RL techniques": the list 1. Online reinforcement learning, 2. Some Extensions. Then in red "Summary:" "Still the old recipes and RL ideas are helpful!"

## Slide 7 — Problem Setup: LLM Math Reasoning Problems

![Slide 7 — Problem Setup: LLM Math Reasoning Problems](../images/10-rl-for-reasoning/slide-7.jpg)

A blue label "Initial state" at upper right sits over a blue-shaded text block: "**Problem:** Suppose $a$ and $b$ are positive real numbers with $a \gt b$ and $ab = 8$. Find the minimum value of $\frac{a^2 + b^2}{a - b}$."

Below is the "**Ground truth solution:**" with three orange-shaded phrases (the steps): "We can write $\frac{a^2+b^2}{a-b} = \frac{a^2+b^2-2ab+16}{a-b} = \frac{(a-b)^2+16}{a-b} = a - b + \frac{16}{a-b}$. By AM-GM, $a - b + \frac{16}{a-b} \geq 2\sqrt{(a-b) \cdot \frac{16}{a-b}} = 8$." Then "Equality occurs when $a - b = 4$ and $ab = 8$." Then "We can solve these equations to find $a = 2\sqrt{3} + 2$ and $b = 2\sqrt{3} - 2$." The ending "Thus, the minimum value is" boxed 8 is circled in a green oval.

Three orange arrows point up from the orange label "Steps = actions" at the three shaded phrases. A green label "reward = 1 if answer is correct" sits by the green oval. At the bottom a red-outlined pink box: "**Intuition:** Sparse-reward MDP with deterministic dynamics".

## Slide 8 — Classical RL Methods for Reasoning

Section divider. A dashed grey box with an orange label "Part 1", then the title "Classical RL Methods for Reasoning" in blue.

"Main papers covered:" (red)
- RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold. Setlur, Garg, Geng, Garg, Smith, **Kumar**. NeurIPS 2024.
- Rewarding Progress: Scaling up Automated Process Supervision for LLM Reasoning. Setlur, Nagpal, Fisch, Geng, Eisenstein, R. Agarwal, A. Agarwal, Berant, **Kumar**. ICLR 2025.

The second paper's footer on slides 24–28 cites it differently, as "Rewarding Progress: Scaling Automated Process Supervision for LLM Reasoning. arXiv 2024." (no "up", and arXiv 2024 rather than ICLR 2025).

## Slide 9 — Setup: Data-Scaling Analysis

![Slide 9 — Setup: Data-Scaling Analysis](../images/10-rl-for-reasoning/slide-9.png)

Top left, a dashed box headed "Basic Approach" (pink): "Collect **problems** and corresponding oracle **solutions** to train on".

A three-axis diagram from a common origin. A black vertical axis pointing up is labelled "Number of questions"; a blue dashed curved arrow links it to the blue text "**SFT**: Obtained by asking bigger models / human to write questions and one ref answer". A red horizontal axis pointing right is labelled in red "Number of correct solutions (all produce the same final answer)"; a red dashed arrow links it to the red text "**RFT**: Multiple solutions sampled on-policy from the learner". A red diagonal axis pointing down-left is labelled "Number of incorrect solutions"; a red dashed arrow links it to the red text "**RL**: Also using bad solutions produced by the learner on some questions".

## Slide 10 — Warmup: Scale #{Questions, Oracle Ans}

![Slide 10 — Warmup: Scale #{Questions, Oracle Ans}](../images/10-rl-for-reasoning/slide-10.jpg)

Footer: "Setlur et al. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.** NeurIPS 2024."

Top right, a small dashed corner diagram of the three axes: "Q" (blue arrow up), "A" (black arrow right), "-ve" (black arrow down-left).

Two line-chart panels, titled "GSM8K" and "MATH". Both have x-axis "#questions" with ticks 8k, 16k, 32k, 64k, 128k and y-axis "Test error". Each legend lists "SFT DeepSeek" (solid line with dot marker) and "RFT DeepSeek" (dashed line). Only dots near the left edge (about 8k questions) are drawn; no line segments and no RFT points are visible. The chart is the first frame of an animation: Kumar says at ≈58:27 that these plots "were supposed to be gifs".
- GSM8K panel (y ticks 0.16, 0.19, 0.22, 0.25, 0.28): two blue dots at the leftmost x position, at about 0.285 and about 0.268.
- MATH panel (y ticks 0.52, 0.55, 0.58, 0.61, 0.64): two blue dots at the leftmost x position, at about 0.625 and about 0.606.

Below the panels, large equations: under GSM8K $|\mathcal{D}|^{-0.15}$, under MATH $|\mathcal{D}|^{-0.05}$.

## Slide 11 — But.. We Can't Just SFT On-Policy Solutions

![Slide 11 — But.. We Can't Just SFT On-Policy Solutions](../images/10-rl-for-reasoning/slide-11.png)

Left: a pink box with a red outline: "**Observation:** On-policy imitation eventually degrades if you train too much on it". Below it: "Fitting self-generated data on limited initial states can hurt generalization on new initial states (prompts)."

Right: a chart titled (in light grey) "RFT Scaling (fixed questions/prompts)". The x-axis is "#correct answers per question" with ticks 10k, 20k, 30k, 40k, 50k, 60k and 128k (the axis skips between 60k and 128k). The y-axis is "Test error", split into two stacked panels with a break: the upper panel for MATH (ticks 0.58, 0.60, 0.62), the lower for GSM8K (ticks 0.22, 0.24, 0.26, 0.28). The legend shows "MATH 8k prompts" (solid) and "16k prompts" (dashed) in red; "GSM8K 8k prompts" (solid) and "16k prompts" (dashed) in purple. Four series, values approximate at 10k / 20k / 30k / 40k / 50k / 60k / 128k:
- MATH 8k prompts (red, solid, light pink to dark red markers): 0.620, 0.614, 0.606, 0.600, 0.599, 0.599, 0.606.
- MATH 16k prompts (red, dashed): 0.607, 0.599, 0.592, 0.589, 0.589, 0.588, 0.597.
- GSM8K 8k prompts (purple, solid): 0.280, 0.270, 0.263, 0.264, 0.270, 0.275, 0.290 (the 128k point is partly hidden by the legend).
- GSM8K 16k prompts (purple, dashed): 0.260, 0.249, 0.241, 0.235, 0.247, 0.246, 0.263.
In every series the error falls and then rises again at larger numbers of correct answers. Footer: "Setlur et al. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.** NeurIPS 2024."

## Slide 12 — Mental Picture: Spurious Steps Derail the Model

![Slide 12 — Mental Picture: Spurious Steps Derail the Model](../images/10-rl-for-reasoning/slide-12.jpg)

Left, a diagram. A tall black-outlined oval on the left holds three light-blue dots labelled "prompts". From each prompt a black curved-arrow path goes through three or four dark-blue dots (steps). From the end of each path a solid red curved arrow goes up to a black-outlined grey oval at the top holding three light-blue dots. From those, red dashed arrows lead down to two light-blue dots in a green-outlined oval labelled "Correct" in green, at the right.

Right of the diagram, red text: "“Spurious” steps" with bullets "Can somehow learn to recover on train" and "But *not on* test, where it will derail".

Far right, a black-titled panel "RFT data has spurious steps." containing: "**Question:** Betty wants a new wallet costing 100 dollars (printed with a dollar sign) and has only half the money she needs. Her parents give her 15 dollars (printed with a dollar sign), and her grandparents twice as much as her parents. How much more money does she need?" and "**Model generated solution:** Betty's parents gave her 15, so her grandparents gave her 2 \* 15 = «15\*2=30»30. In total, Betty received 30 + 15 = «30+15=45»45." Then a highlighted (blue) passage "The total amount of money Betty needs is 100 \* 2 = «100\*2=200»200. Betty needs 100 / 2 = «100/2=50»50 in total." Then "Betty still needs to save 50 - 45 = «50-45=5»5. Answer is 5."

Bottom, a pink box: "Akin to “**causal confusion**” in imitation when conditioned on history of past actions". Footer: "Setlur et al. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.** NeurIPS 2024."

## Slide 13 — How Can We Address This?

A white slide with only the centred text "How Can We Address This?"

## Slide 14 — An Idea: Assigning "Credit" to Steps

![Slide 14 — An Idea: Assigning "Credit" to Steps](../images/10-rl-for-reasoning/slide-14.png)

Top left, a grey box: "**Question**: 4 apples and 3 pears cost 25 dollars, but 7 apples and 6 pears cost 49 dollars (printed with dollar signs). What is the cost of 1 apple?" Top right, a red-outlined pink box: "**Key insight:** If we can identify spurious steps somewhat *precisely*, then we can avoid this issue".

A chain of six blue ovals joined by blue curved arrows, labelled "Start" beside the first:
- $C_1$: Let cost of 1 apple be x and 1 pear be y.
- $C_2$: 4x+3y=25 (Eq 1) and 7x+6y=49 (Eq 2).
- $C_3$: To remove y, multiple Eq1 by 2 and subtract Eq 2 from it. ("multiple" for "multiply", printed so.)
- $C_4$: (8x+6y)-(7x+6y)=50-49.
- $C_5$: x=1.
- $C_6$: Answer: 1 apple costs 1 dollar (printed with a dollar sign).

A green check mark with "Correct solution" at the bottom right. Footer: "Setlur et al. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.** NeurIPS 2024."

## Slide 15 — Negative On-Policy Data Gives Advantages

![Slide 15 — Negative On-Policy Data Gives Advantages](../images/10-rl-for-reasoning/slide-15.png)

Top right: the corner diagram of three axes ("Q" up, "A" right, "-ve" down-left), with the "-ve" arrow drawn in blue here (and on slides 16 to 19 and 23 below) and the others black.

A horizontal black arrow runs left to right through blue dots (the steps of one solution), ending in a red dot (a failed final answer). Four of the blue dots (the second to fifth) are each enclosed in a red dotted square. From these dots orange dashed curves fan out to end points: red filled dots (failures) and green open circles (successes), some above the line and some below. A blue bracket on the right groups the end points above and below, labelled "Rollout succeeds ➔ likely not a spurious step". A blue arrow from the text "Sometimes succeeds, sort of OK step" points at the second blue dot; a blue arrow from "Always fails, not OK" points at a red end point at the lower right. At the bottom, a pink box: "**Connection:** This is equivalent to the value function of the *rollout* policy…." Footer: "Setlur et al. **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold.** NeurIPS 2024."

## Slide 16 — Negative On-Policy Data Gives Advantages

![Slide 16 — Negative On-Policy Data Gives Advantages](../images/10-rl-for-reasoning/slide-16.jpg)

The same diagram as slide 15 without the red dotted squares, and with notation. The first blue dot is labelled $\mathbf{x}$; the arrows between the second and third dots and between the third and fourth dots are labelled $\hat{\mathbf{y}}_ {i-1}$ and $\hat{\mathbf{y}}_ i$. Same bracket ("Rollout succeeds ➔ likely not a spurious step") and the same two annotations ("Sometimes succeeds, sort of OK step", "Always fails, not OK").

At the bottom the equation

$$Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\ \hat{\boldsymbol{y}}_ i) = \mathbb{E}_ {\boldsymbol{y}^{\text{new}}_ {i+1:L} \sim \tilde{\pi}(\cdot \mid \boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i})} \left[ r\left( [\hat{\boldsymbol{y}}_ {1:i}, \boldsymbol{y}^{\text{new}}_ {i+1:L}], \boldsymbol{y} \right) \right]$$

with $(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1})$ braced and labelled "state", $\hat{\boldsymbol{y}}_ i$ braced and labelled "action", the expectation subscript highlighted in orange, and the reward term $r(\ldots)$ in a blue oval; under the right-hand side the brace label reads "expected future reward under new actions (i.e., steps) sampled by policy $\tilde{\pi}$". Footer as on slide 15.

## Slide 17 — Negative On-Policy Data Gives Advantages

![Slide 17 — Negative On-Policy Data Gives Advantages](../images/10-rl-for-reasoning/slide-17.jpg)

Top right, the corner diagram with the blue "-ve" arrow. At the top, two small labels: "Q-value after step $i$" above the first $Q$ term and "Q-value after step $i-1$" above the second. The equation:

$$A_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\ \hat{\boldsymbol{y}}_ i) = Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\ \hat{\boldsymbol{y}}_ i) - Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-2};\ \hat{\boldsymbol{y}}_ {i-1}).$$

A pink box on the right: "**Advantage:** relative change in the value function having committed to a step".

A worked example: a chain of five blue dots ending in a red dot, joined by black arrows. Below it, the row $Q_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\ \hat{\boldsymbol{y}}_ i) =$ (with "state" and "action" braces as on slide 16) followed by the values 0.66, 0.0, 1.0, 0.25, 0.0. Beneath, the row $A_{\tilde{\pi}}(\boldsymbol{x}, \hat{\boldsymbol{y}}_ {1:i-1};\ \hat{\boldsymbol{y}}_ i) =$ followed by -0.66, 1.0, -0.75, -0.25. Blue arrows run from each Q value down to the advantage under it and the one to its right (0.66 to -0.66; 0.0 to -0.66 and 1.0; 1.0 to 1.0 and -0.75; 0.25 to -0.75 and -0.25; 0.0 to -0.25), so each advantage is a Q value minus the previous Q value. Footer as on slide 15.

## Slide 18 — Using Advantages for Training

![Slide 18 — Using Advantages for Training](../images/10-rl-for-reasoning/slide-18.png)

Top right, the corner diagram with the blue "-ve" arrow. A pink box: "**Option 1:** Filter steps by advantages directly".

Two solutions drawn as rows of blue dots joined by black arrows. Top row, labelled "Incorrect solution from base $\pi$", ends in a red dot; a red rounded box around its second and third dots (the arrow between them) is labelled $A_{\tilde{\pi}} \gg 0$. Orange dashed rollout curves fan around that box. A thick black arrow with a green check mark runs from the box to a blue cylinder labelled "Training dataset". Bottom row, labelled "Correct solution from base $\pi$", ends in a green open circle; a red box around its fourth and fifth dots is labelled $A_{\tilde{\pi}} \ll 0$, and a thick black arrow to the training dataset is crossed out by a red X. Footer as on slide 15.

## Slide 19 — Results: Filtering Steps w/ Advantages

![Slide 19 — Results: Filtering Steps w/ Advantages](../images/10-rl-for-reasoning/slide-19.jpg)

Top right, the corner diagram with the blue "-ve" arrow. Two chart panels, "Advantage filtered RFT (GMS8k)" (printed so) and "Advantage filtered RFT (MATH)". Both have x-axis "#correct answers per question" with ticks 10k, 20k, 30k, 40k, 50k, 60k, and y-axis test error. Their legends show "RFT" (solid) and "per-step RFT" (dashed); only two dots at x = 10k are drawn, with no lines (the first frame of an animation; see slide 10).
- GSM8K panel (y from 0.250 to 0.280 in steps of 0.005): two purple dots at 10k, at about 0.2805 and about 0.2725.
- MATH panel (y ticks 0.58 to 0.62): two red dots at 10k, at about 0.615 and about 0.6075.

A large black downward arrow in each panel (above the empty lower right region), with the text "Advantage filtering helps!" beside the left one. Bottom: a pink box "**Finding:** Advantage-filtered RFT does not fall prey to spurious steps". Footer as on slide 15.

## Slide 20 — Using Advantages for Offline RL (DPO)

![Slide 20 — Using Advantages for Offline RL (DPO)](../images/10-rl-for-reasoning/slide-20.png)

Top right, the corner diagram with the blue "-ve" arrow. A pink box: "**Option 2:** Retain partial rollouts from $\tilde{\boldsymbol{\pi}}$ for training". Text at left: "Incorrect solution from base $\pi$". A row of six dots, five blue and the last red, joined by black arrows. From the second and third blue dots, orange dashed rollout curves go up and down, each pair ending in either a green open circle (success) or a red dot (failure). Footer as on slide 15.

## Slide 21 — Using Advantages for Offline RL (DPO)

![Slide 21 — Using Advantages for Offline RL (DPO)](../images/10-rl-for-reasoning/slide-21.png)

Same header, option box and incorrect-solution row as slide 20, but all rollouts except one are faded. The one highlighted rollout (bold orange dashed) leaves the second dot and ends at a green open circle at the top; a thick black double-headed arrow connects that green end point to the grey (faded) failure end point next to it and down to the original solution's fourth dot. A thin black arrow labelled "Preference pair" runs from there to the blue cylinder "Training dataset". Footer as on slide 15.

## Slide 22 — Using Advantages for Offline RL (DPO)

![Slide 22 — Using Advantages for Offline RL (DPO)](../images/10-rl-for-reasoning/slide-22.png)

Same header and option box. The text now reads "Correct solution from base $\pi$", and the row of dots ends in a green filled dot. The highlighted rollout (bold orange dashed) leaves the third dot downward and ends at a red dot; a thick black double-headed arrow joins that red end point to the green end of the main solution, with a thin black arrow labelled "Preference pair" to the "Training dataset" cylinder. Other rollouts are faded. Footer as on slide 15.

## Slide 23 — Results: Scaling for Offline RL

![Slide 23 — Results: Scaling for Offline RL](../images/10-rl-for-reasoning/slide-23.jpg)

Top right, the corner diagram with the blue "-ve" arrow. Two chart panels, "GSM8K" and "MATH", each split by axis breaks into three stacked sub-panels sharing the x-axis "#questions" (ticks 8k, 16k, 32k, 64k, 128k) and the y-axis label "Test error". Only dots at the leftmost position (about 8k questions) are drawn; the legend lines show the series names but no lines or further points are drawn (the first frame of an animation, which Kumar says at ≈58:27 was supposed to be a gif).

GSM8K:
- Top sub-panel, orange, legend "SFT Llama2" (solid with dot) and "per-step DPO" (dashed); y ticks 0.35, 0.43, 0.51, 0.59: two dots at about 0.63 (cut off at the top edge) and about 0.52.
- Middle sub-panel, blue, legend "SFT DeepSeek" and "per-step DPO"; y ticks 0.13, 0.17, 0.21, 0.25, 0.29: two dots at about 0.285 and about 0.23.
- Bottom sub-panel, green, legend "per-step DPO DeepSeek Pass@5" (dashed); y ticks 0.04, 0.08: one dot at about 0.078.

MATH:
- Top sub-panel, orange, legend "SFT Llama2" and "per-step DPO"; y ticks 0.72, 0.79, 0.86: two dots at about 0.90 (cut off at the top edge) and about 0.865.
- Middle sub-panel, blue, legend "SFT DeepSeek" and "per-step DPO"; y ticks 0.52, 0.55, 0.58, 0.61: two dots at about 0.63 (cut off at the top edge) and about 0.58.
- Bottom sub-panel, green, legend "per-step DPO Pass@5"; y ticks 0.32, 0.36, 0.40, 0.44: one dot at about 0.405.

Bottom: a pink box "**Finding:** Using offline RL (via per-step RL) gives you 8x data efficiency in performance over imitation only." Footer as on slide 15.

## Slide 24 — Online RL: Basic Recipe with 0/1 rewards

![Slide 24 — Online RL: Basic Recipe with 0/1 rewards](../images/10-rl-for-reasoning/slide-24.png)

A pink box at upper right: "**Key idea:** Train with a binary 0/1 reward using policy gradients". Left text: "A solution from *current* $\pi$" (the second line in blue italics). The equation

$$\mathbb{E}_ {\mathbf{x} \sim \mathcal{D}_ {\text{train}}, \mathbf{y} \sim \pi(\cdot \mid \mathbf{x})} \left[ \sum_j \nabla_\pi \log \pi(\mathbf{y}_ j \mid \mathbf{x}, \mathbf{y}_ {0:j-1}) \cdot r(\mathbf{x}, \mathbf{y}) \right]$$

Below it, a chain starting at the label $\mathbf{x}$, with five blue dots and a final grey dot joined by black arrows; the arrows into the third and fourth dots are labelled $\hat{\mathbf{y}}_ {i-1}$ and $\hat{\mathbf{y}}_ i$. A black arrow points from the text "r(x, y) = {0, 1}" to the grey final dot. At lower left, an orange box: "*Can also use other policy gradient methods (PPO, GRPO, etc)!*" Footer: "Setlur et al. **Rewarding Progress: Scaling Automated Process Supervision for LLM Reasoning.** arXiv 2024."

## Slide 25 — Per-Step Advantages in Online RL

![Slide 25 — Per-Step Advantages in Online RL](../images/10-rl-for-reasoning/slide-25.jpg)

A pink box at upper right: "**Question:** Should we do rollouts from $\tilde{\boldsymbol{\pi}}$ on the fly?" Left text: "A solution from" with "base model" struck through in red, and below it in blue italics "*current* $\pi$". A row of five blue dots and a grey final dot; orange dashed rollouts fan out from the second and third dots, ending in green open circles and red dots (as on slide 20). At right, in blue: "Fit a parametric model to advantage predictions", above a small drawn neural network (inputs at the bottom, three layers narrowing to a single output with an arrow up). Footer: "Setlur et al. **Rewarding Progress: Scaling Automated Process Supervision for LLM Reasoning.** arXiv 2024."

## Slide 26 — Approach: Process Advantage Verifiers (PAVs)

A pop-up box covers a faded copy of the previous slides' content. The pop-up has a blue title bar "[Informal] Optimal Rollout Policy, $\tilde{\pi}$" and the text: "The optimal rollout policy $\tilde{\pi}$ is one which produces advantages that **most effectively distinguish** good and bad steps from *all* $\pi$." (the bold phrase is red).

The faded background, partly legible, shows: a pink box "Key idea: Advantage as dense reward bonus in RL"; "A solution from *current* $\pi$"; an orange box beginning "Challenge: These" whose remaining text is covered except fragments "ues once", "some $\tilde{\pi}$ are", "fixed in RL"; the label $\mathbf{x}$; the expression $A_{\tilde{\pi}}(\boldsymbol{x}, \boldsymbol{y}_ {0:i-1}; \boldsymbol{y}_ i)$ with a small neural network under it; red dots; and "r(x, y) = {0, 1}". The faded text is not fully legible. Footer: "Setlur et al. **Rewarding Progress: Scaling Automated Process Supervision for LLM Reasoning.** arXiv 2024."

## Slide 27 — Results: Dense-Reward RL Performance

![Slide 27 — Results: Dense-Reward RL Performance](../images/10-rl-for-reasoning/slide-27.jpg)

Two line charts with shaded confidence bands, each with series "ORM-RL" (dashed, light orange) and "PAV-RL" (solid, darker orange). Values are approximate.

Left, "Sample Efficiency (2B)": y-axis "Accuracy" (ticks 0.15, 0.20, 0.25); x-axis "Training Iterations ($\times 10^3$)" ticks 0 to 10. Two series:
- PAV-RL: about 0.147 at 0, 0.197 at 1, 0.19 at 2, 0.19 at 3, 0.23 at 4, 0.21 at 5, 0.22 at 6, 0.255 at 7, 0.265 at 8, 0.26 at 9, 0.252 at 10.
- ORM-RL: about 0.147 at 0, 0.165 at 1, 0.15 at 2, 0.16 at 3, 0.175 at 4, 0.18 at 5, 0.19 at 6, 0.195 at 7, 0.187 at 8, 0.20 at 9, 0.188 at 10.
Annotations: a horizontal dash-dotted arrow at accuracy about 0.197 from about x = 7 back to x = 1, labelled "6 ×"; a vertical dashed arrow at x = 8 from the ORM-RL curve up to the PAV-RL curve, labelled "7%".

Right, "Sample Efficiency (9B)": y ticks 0.40, 0.45, 0.50, 0.55; x-axis "Training Iterations ($\times 10^2$)" ticks 2, 10, 18, 26, 34, 42, 50. Two series:
- PAV-RL: about 0.385 at 2, 0.475 at 10, 0.515 at 18, 0.53 at 26, 0.53 at 34, 0.55 at 42, 0.548 at 50.
- ORM-RL: about 0.385 at 2, 0.425 at 10, 0.42 at 18, 0.43 at 26, 0.435 at 34, 0.48 at 42, 0.488 at 50.
Annotations: a horizontal dash-dotted arrow at accuracy about 0.49 from about x = 50 back to x = 10, labelled "5 ×"; a vertical dashed arrow near x = 43 from about 0.49 up to the PAV-RL curve, labelled "6%".

Bottom: a pink box "**5-6x** sample efficiency than outcome rewards only, **6-7%** improvement". Footer as on slide 24.

## Slide 28 — Results: Discovers Solutions to Hard Questions

![Slide 28 — Results: Discovers Solutions to Hard Questions](../images/10-rl-for-reasoning/slide-28.jpg)

Left, a diagram: an input box "input (x)" (peach) with an arrow up into a dashed grey box "Base policy", from which an arrow goes up to a column of four blue output boxes (with dots between them), beside a double-headed blue arrow labelled "N = 256 independent trials".

Right, a line chart "Solving Hard Questions". The y-axis is "Success Rate on Problems Unsolved by SFT @256" (ticks 0.00, 0.05, 0.10, 0.15); the x-axis is "N" with ticks $2^1$ to $2^8$. Two series with shaded bands, approximate values at $2^1$ to $2^8$:
- $A_{\tilde{\pi}}$ (orange, solid, star markers): 0.017, 0.028, 0.059, 0.082, 0.090, 0.128, 0.149, 0.152.
- $A_{\pi}$ (green, dashed, square markers): 0.004, 0.009, 0.013, 0.019, 0.023, 0.024, 0.029, 0.032.

Bottom: a pink box "**Insight:** Specifically, PAVs give exploration crucial for solving hard questions". Footer as on slide 24.

## Slide 29 — Takeaways

Section summary for Part 1. At upper left an orange label "Part 1" in a dashed grey box; the title "Takeaways" in blue. Three paragraphs, each opening with an arrow bullet and a coloured lead word:

- **Challenge:** (red) Imitating positive data scales (slowly), but it induces spurious steps, that are memorized.
- **Algorithm:** (blue) Careful, model-specific negative data can help with addressing this issue via offline / online RL
- **Result:** (green) Offline RL gives **8x** sample efficiency vs imitation. Online RL w/ dense advantage reward improves **5-6x** (the numbers 8x and 5-6x are blue)

## Slide 30 — Training "Thinking" Models via RL

Section divider. A dashed grey box with an orange label "Part 2", then the title "Training “Thinking” Models via RL" in blue.

"Main papers covered:" (red)
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *2025*
- Kimi K1.5: Scaling Reinforcement Learning with LLMs.
- **[Optional]** Optimizing Test-Time Compute requires Solving a Meta-RL Problem. *Setlur, Qu, Yang, Zhang, Smith, Kumar. CMU MLD Blog, 2025*

## Slide 31 — Example: A Trace From Thinking Models

A figure from another paper. A yellow rounded box under the label "Problem:" reads "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?" Below it, a large light-green rounded box holds the model's output. A double-headed vertical arrow on its left, labelled "Thinking Process", spans the text from the opening tag to the closing tag; a shorter double-headed arrow labelled "Answer" spans the text after the closing tag. The tags `<think>` and `</think>` are red. In the text, the words that mark the model's meta behaviours are bold and underlined (shown here in bold). The text reads:

"<think> Okay, let me try to figure out this problem. So, James is writing a 3-page letter to two different friends twice a week. We need to find out how many pages he writes in a year. **Hmm**, let's break this down step by step. First, **let me understand the details**. […] **Wait,** if he writes to 2 friends twice a week, does that mean he writes 2 letters each time, and he does this twice? […] So per week, he writes 2 friends * 3 pages * 2 times. **Let me check that.** […] **Alternatively,** maybe it's 3-page letter to 2 different friends each time he writes, and he does that twice a week. So each time he writes, he's writing two letters (each 3 pages), so that's 2 \* 3 = 6 pages per writing session. Then, twice a week would be 6 \* 2 = 12 pages per week. Then, multiply that by the number of weeks in a year. But how many weeks are in a year? Typically, 52 weeks. So 12 pages/week * 52 weeks = 624 pages. **Is that right?** **Wait, let me confirm.** So, per week: […] **That makes sense.** But maybe **I need to double-check** if the problem is interpreted differently. […] That seems consistent. Wait, but wait a second. Is there any chance that the problem might be considering a year as 52 weeks, **but some people might account for leap years or something else?** But generally, in problems like this, unless specified, we take 52 weeks per year. So that's standard. So, conclusion: […] So the answer is 624. So I think that's correct. </think>"

Then the answer part: "James writes 3-page letters to 2 friends twice a week. Here's the breakdown: […] Answer: James writes" followed by a purple-boxed "624 pages in a year."

Footer: "*Figure from:* Marjanovi´c et al. **DeepSeek-R1 Thoughtolgy.** arXiv 2025." (printed so)

## Slide 32 — What Changed?

Subtitle in blue italics: "(Disclaimer: this is my version!)". Arrow-bulleted text:

- **The RL training objectives remained the same:** (red)
  - DeepSeek-R1 uses GRPO, another policy gradient method
  - Kimi-1.5 uses some version of APA (advantage-induced policy alignment)
  - Some minor changes in reward function
- ***Crudely*, the action space changed:** (red)
  - These base models became better at performing "macro" actions, such as revisiting a given answer, verification, backtracking, planning, etc
  - Crudely, we can now think of operating in this new action space that presents several "meta" actions
- We **increased the token budget drastically** (red phrase) to chain multiple macro actions

## Slide 33 — Formulation: Training Thinking Models via RL

Bold italic line: "Let's start from the final goal". The displayed objective

$$\max_\pi \ \mathbb{E}_ {\mathbf{x} \sim \mathcal{D}_ {\text{test}}} \left[ \mathbb{E}_ {\mathbf{z} \sim \pi(\cdot \mid \mathbf{x})} \left[ r(\mathbf{x}, \mathbf{z}) \right] \right]$$

with three labels: a green oval around $\mathbf{x} \sim \mathcal{D}_ {\text{test}}$ labelled in green "on test problems"; a blue oval around $\mathbf{z} \sim \pi(\cdot \mid \mathbf{x})$ labelled in blue "response sampled from model (longer than typical solution)"; and, at right in orange, "Total compute constraint per problem" (no equation part yet attached to it).

## Slide 34 — Formulation: Training Thinking Models via RL

![Slide 34 — Formulation: Training Thinking Models via RL](../images/10-rl-for-reasoning/slide-34.jpg)

The objective now carries a constraint:

$$\max_\pi \ \mathbb{E}_ {\mathbf{x} \sim \mathcal{D}_ {\text{test}}} \left[ \mathbb{E}_ {\mathbf{z} \sim \pi(\cdot \mid \mathbf{x})} \left[ r(\mathbf{x}, \mathbf{z}) \right] \right] \ \text{s.t.} \ \forall \mathbf{x}, \ \mathbb{E}_ {\pi(\cdot \mid \mathbf{x})} |\mathbf{z}| \leq C_0$$

with an orange circle around $C_0$ and the orange text "This compute budget is fixed!" A green down arrow from the circled $\mathcal{D}_ {\text{test}}$ (green oval) points to a second line replacing it with the training distribution:

$$\max_\pi \ \mathbb{E}_ {\mathbf{x} \sim \mathcal{D}_ {\text{train}}} \left[ \mathbb{E}_ {\mathbf{z} \sim \pi(\cdot \mid \mathbf{x})} \left[ r(\mathbf{x}, \mathbf{z}) \right] \right] \ \text{s.t.} \ \ldots$$

Below, "**Can optimize this via:**" with two arrow bullets: "RL (like DeepSeek-R1): outcome-reward RL" and "SFT / RFT: collect data, filter by correctness, maximize likelihood".

## Slide 35 — "Action" Space: Incorporating Meta Strategies

![Slide 35 — "Action" Space: Incorporating Meta Strategies](../images/10-rl-for-reasoning/slide-35.jpg)

A large grey rounded panel headed "A contrast in behaviors explored by the two models". On the left, four white cards giving example behaviours: "Verifications" ("Let me check my answer …"), "Subgoal Setting" ("Let's try to get to a multiple of 10"), "Backtracking" ("Let's try a different approach, what if we …"), "Backward Chaining" ("Working backwards, 24 is 8 times 3").

Two line charts, each headed by a model logo (the slide does not name the models: a purple logo over the first chart in a purple frame, a blue infinity-shaped logo over the second in a blue frame). Both charts have the x-axis "RL steps" (0 to 250) and the y-axis "average count" (0 to 12) and four series: Verifications (blue), Backtracking (orange), Subgoal Setting (green; "Subgoals" in the second chart's legend), Backwards Chaining (red).
- First chart (purple frame), approximate values; points every 10 RL steps. Verifications (blue): about 0.7 at 0, 0.3 to 0.4 at 10 and 20, then 5.6 at 30, 7.0 at 40, 10.5 at 50 and a peak of about 11.5 at 60, falling to 8.7 at 70 and 5.1 at 80 and 90, rising again to 6.0 at 100, 7.5 at 110 and 7.7 at 120, then 5.8 at 130, 3.8 at 140, 0.9 at 150, and about 0.2 to 0.3 from 160 to 240. Backtracking (orange): about 0.6 at 0, 0.2 to 0.3 at 10 and 20, 3.0 at 30, 4.2 at 40, 7.0 at 50, a peak of about 8.6 at 60, 7.7 at 70, 4.9 at 80, 4.5 at 90, 5.0 at 100, 3.9 at 110, about 3.3 to 3.5 from 120 to 150, then falling to about 2.9 at 170, 2.3 at 180 and 1.4 at 190, between 1.1 and 2.0 from 200 to 230, and about 1.7 at 240. Subgoal Setting (green) and Backwards Chaining (red): about 0.5 at 0, then near 0 throughout.
- Second chart (blue frame): all four series are near 0 across 0 to 250 (a brief value of about 0.3 near step 0 to 10); a red X and the text "no behaviors shown" sit in the plot.

At the lower right, blue text: "This model does not improve during RL" (referring to the second chart). Footer: "Gandhi et al. **Cognitive Behaviors that Enable Self-Improving Reasoners.** arXiv 2025."

## Slide 36 — Longer Length: RL Training Amplifies Length

![Slide 36 — Longer Length: RL Training Amplifies Length](../images/10-rl-for-reasoning/slide-36.jpg)

Eight small line-chart panels in two rows of four, from the Kimi 1.5 report; the figure text is very small, so the values below are approximate. Each panel has x-axis "Iterations" (0 to about 155), a left y-axis "Accuracy" and a right y-axis "Token Length", and two series with a legend "Performance" and "Token Length": Performance (blue line with open circles) and Token Length (orange line with open squares, with a light-orange shaded band behind it). In the panels for Biology and KAOYAN an extra column of black axis numbers (0.0 to 1.0) overlaps the right axis; it is a rendering artifact of the figure.

Row 1: panel titles AIME2024, ChatGLMMath, GAOKAO, GPQA. Row 2: Biology, Chemistry, Physics, KAOYAN. In all eight panels both series trend upward over the iterations. Approximate first and last values:
- AIME2024: accuracy from about 0.13 to about 0.40 (reaching about 0.5 in between, noisy); token length from about 2,000 to about 19,000 (axis to 30,000).
- ChatGLMMath: accuracy 0.66 to 0.91; token length about 1,500 to 9,000 (axis to 17,500).
- GAOKAO: accuracy 0.81 to 0.93; token length about 1,500 to 8,500 (axis to 16,000).
- GPQA: accuracy 0.12 to 0.50; token length about 1,000 to 11,000 (axis to 20,000).
- Biology: accuracy 0.71 to 0.88; token length about 1,000 to 4,500 (axis to 12,000).
- Chemistry: accuracy 0.46 to 0.60 (peak about 0.68); token length about 1,000 to 6,500 (axis to 14,000).
- Physics: accuracy 0.62 to 0.65 (peak about 0.77); token length about 1,500 to 10,500 (axis to 20,000).
- KAOYAN: accuracy 0.62 to 0.88; token length about 2,000 to 11,500 (axis to 18,000).

At the right edge, blue text: "Why?" Footer: "Kimi team. **Kimi 1.5 Technical Report.** arXiv 2025."

## Slide 37 — A New Paradigm: Test-Time Scaling

A diagram. A row of five peach boxes at the bottom-left labelled "input problem (x)" with a double-headed arrow below. A black arrow goes up from them into a dashed grey box "Language model". To the right of the peach boxes, four blue boxes follow in the same row (the generated tokens); dotted grey arrows lead from each blue box at the bottom up around the language model to a row of five blue boxes at the top, one black arrow leaves the model's top right into the last top box. This is the autoregressive loop: outputs fed back as inputs. Footer: "Snell et al. **Scaling LLM Test-Time Compute Optimally can be more Effective than Scaling Model Parameters.** ICLR 2025 (Oral)."

## Slide 38 — A New Paradigm: Test-Time Scaling

The same diagram as slide 37 with the model box enlarged and relabelled "Bigger language model". Same footer as slide 37.

## Slide 39 — A New Paradigm: Test-Time Scaling

![Slide 39 — A New Paradigm: Test-Time Scaling](../images/10-rl-for-reasoning/slide-39.png)

Two diagrams side by side. Left: the "Bigger language model" diagram of slide 38 (five peach input boxes labelled "input problem (x)", four blue generated boxes, five blue output boxes on top). Right: a "Language model" box of normal height but wider, with the same five peach input boxes labelled "input problem (x)", eight blue boxes of generated tokens beside them and nine blue output boxes above, joined by dotted grey arrows; a double-headed arrow over the top row carries the label "longer responses, “more thinking”, self-correction, etc." Same footer as slide 37.

## Slide 40 — Results: Thinking Models Top the Leaderboards!

![Slide 40 — Results: Thinking Models Top the Leaderboards!](../images/10-rl-for-reasoning/slide-40.jpg)

Three bar charts. Footer: "OpenAI. **o3 and o4-mini blog post.** 2025."

Left, "Codeforces Competition Code" (y-axis "ELO", no numeric ticks, bar values labelled), one series of four yellow bars (the two o3 / o4-mini "with terminal" bars are a darker yellow, the other two paler): o1 1891; o3-mini 2073; o3 (with terminal) 2706; o4-mini (with terminal) 2719.

Middle, "Humanity's Last Exam, Expert-Level Questions Across Subjects" (y-axis "Accuracy (%)"), one series of seven bars (darker yellow for the o3 and o4-mini rows, paler for the others): o1-pro 8.12; o3-mini 13.40; o3 (no tools) 20.32; o3 (python + browsing** tools) 24.90 (printed so); o4-mini (no tools) 14.28; o4-mini (with python + browsin… (label cut off) 17.70; Deep research 26.60.

Right, a bar chart on AIME 2024 ("*Pass@1*") with y-axis "Accuracy / Percentile (%)" from 0 to 100 (ticks 0, 20, 40, 60, 80, 100) and legend entries "DeepSeek-R1" (blue hatched), "OpenAI-o1-1217" (grey), "DeepSeek-R1-32B" (blue), "OpenAI-o1-mini" (grey), "DeepSeek-V3" (pale blue). Five bars: DeepSeek-R1 79.8; OpenAI-o1-1217 79.2; DeepSeek-R1-32B 72.6; OpenAI-o1-mini 63.6; DeepSeek-V3 39.2.

## Slide 41 — But Lots of Interesting Questions Remain!

![Slide 41 — But Lots of Interesting Questions Remain!](../images/10-rl-for-reasoning/slide-41.jpg)

Three dashed grey panels, each with an orange heading, a figure and a red-lead summary.

Panel 1, heading "Desiderata and Formulation". A small figure with the red-and-black caption "Learning “how”: Train algorithm $A_\theta(x)$ to spend **extra test compute**, search over **responses** & discover the **final answer**." (the words in bold are purple, green and blue respectively). Beneath it a double-headed arrow "full response $y$, evaluated by $r(x, y)$" spans labelled green boxes "initial response", "more $A_\theta(x)$" and a "final answer" in blue boxes; the row is labelled $A_\theta(x)$ on the left. These sit above a dashed box "Language Model". Below the model, input tokens as peach boxes ("input tokens ($x$)"), then green and purple boxes under the label "extra compute tokens e.g., self-verification of response", then blue boxes; arrows go up into the model. Summary: "*Summary:* Pose it as an adaptation problem".

Panel 2, heading "Ingredient 1: Dense rewards". A plot with y-axis "reward" (ticks 0 and 1) and x-axis "Test compute budget (H)" (double-headed). A blue curve labelled "Dense reward" rises from 0 with a bump, a dip, then rises to about 0.9 near the right end; a black step line labelled "outcome 0/1 reward" stays at 0 and jumps to 1 only at the far right. A row of five blue boxes lies along the bottom. Summary: "*Summary:* More than outcome reward needed".

Panel 3, heading "Ingredient 2: RL >> SFT". A plot with y-axis "Reward" and x-axis "scaling test-time compute $H$, #training data $n = \Omega(H)$". A light-blue curve labelled "RL" with five dark-blue dots rises steeply and concavely; a light-green curve labelled "SFT/distillation" with five green triangles rises slowly and flattens. A vertical dashed arrow between the curves is labelled "Gap grows as $\sqrt{H}$". Summary: "*Summary:* Use reward signals for training".
