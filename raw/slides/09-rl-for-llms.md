---
title: Lecture 9 — RL for LLMs: Preference Optimization (slide deck)
lecture: 9
slides: 57
printed_numbers: 1–99
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf
numbering: printed number is not the PDF page number; 57 pages carry printed numbers 1–99 because hidden slides were not exported; pages 1, 12, 22, 24, 34, 39, 48, 49 print no number (labels 1, 14, 57, 61, 73, 78, 88, 89 inferred; 14, 57, 88, 89 ambiguous); numbers print bottom left
figure_audit: Transcribed by Claude Sonnet 5.5. Six figure slides (50, 61, 72, 78, 87, 94) were checked by Claude Opus 5.5 against 1400px renders (slide 87's type column also at 200 dpi). Slides 72, 78 and 94 (every series and value) and 61 agree, and slide 50's table agrees cell by cell. Corrections: on slide 50, the encoder-decoder sketch has two rows of red decoder boxes, inputs below and outputs above; on slide 87, three rows (not two) carry an orange diamond in the type column, and the two hard-to-read model names were confirmed.
images: 24 slides are rendered and shown under their headings (2, 4, 7, 41, 42, 43, 50, 56, 61, 63, 64, 67, 70, 72, 74, 76, 77, 78, 80, 87, 92, 94, 98, 99). Not rendered — slide 12, identical to slide 4; slides whose pictures are screenshots or panels of text transcribed here in full, or clip-art (6, 8, 9, 10, 39, 40, 51, 52, 60, 65, 81, 82, 83, 84, 85, 88, 93, 95, 96, 97); and the title, plan, section-divider, text and equation slides (1, 5, 13, 14, 57, 62, 66, 69, 73, 75, 79, 89).
---

# Lecture 9 — RL for LLMs: Preference Optimization: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025) Lecture 9 deck, "RL for LLMs: Preference Optimization", given by guest lecturer Archit Sharma and based on slides from CS224N, transcribed from the PDF. Cite these as "slide N", the printed number. Diagrams, photographs and plots are described in prose since the KB is read as text.

The recording (about 63 minutes) follows the deck in order from slide 2 to the closing "What's next?" slides; Sharma does not read out the title or the section dividers. At ≈58:14, after a student's question about engagement, Sharma jumps ahead to slide 95 (sycophancy in GPT-4o) and then returns to reward hacking and over-optimization (slides 92 and 94) and what's next (slides 96 and 99). Slides 81 (the kinds of task the labelers wrote), 93 (the hallucination examples; hallucination is mentioned only in the closing sentence), 97 (Constitutional AI) and 98 (self-improvement) are not discussed aloud, so for them this file is the only source. Part of the DPO derivation, the cancellation of $Z(x)$ in the difference of rewards (slide 77), is written on the board at ≈42:37–44:11; the transcript is this KB's only record of that work.

## Slide numbers vs PDF pages

The deck is not 1:1 with its printed numbers. The PDF has 57 pages, but the printed numbers run from 1 to 99, because slides hidden in the source deck were not exported. The printed numbers absent from the PDF are 3, 11, 15–38, 44–49, 53–55, 58, 59, 68, 71, 86, 90 and 91. Every numbered page prints its number in the bottom-left corner.

Eight pages print no number (PDF pages 1, 12, 22, 24, 34, 39, 48 and 49; mostly the title and section-divider pages), so their labels come from the page-to-slide map and are inferred. Pages 1 (slide 1), 24 (slide 61), 34 (slide 73) and 39 (slide 78) are inferred from position and unambiguous. Pages 12 (slide 14), 22 (slide 57), 48 (slide 88) and 49 (slide 89) are ambiguous: each sits in a gap of absent numbers, and the lowest fitting number is used, so the true printed number could be higher.

| PDF page | Printed slide |
| -------- | ------------- |
| 1 | 1 (inferred from position) |
| 2 | 2 |
| 3 | 4 |
| 4 | 5 |
| 5 | 6 |
| 6 | 7 |
| 7 | 8 |
| 8 | 9 |
| 9 | 10 |
| 10 | 12 |
| 11 | 13 |
| 12 | 14 (inferred, ambiguous) |
| 13 | 39 |
| 14 | 40 |
| 15 | 41 |
| 16 | 42 |
| 17 | 43 |
| 18 | 50 |
| 19 | 51 |
| 20 | 52 |
| 21 | 56 |
| 22 | 57 (inferred, ambiguous) |
| 23 | 60 |
| 24 | 61 (inferred from position) |
| 25 | 62 |
| 26 | 63 |
| 27 | 64 |
| 28 | 65 |
| 29 | 66 |
| 30 | 67 |
| 31 | 69 |
| 32 | 70 |
| 33 | 72 |
| 34 | 73 (inferred from position) |
| 35 | 74 |
| 36 | 75 |
| 37 | 76 |
| 38 | 77 |
| 39 | 78 (inferred from position) |
| 40 | 79 |
| 41 | 80 |
| 42 | 81 |
| 43 | 82 |
| 44 | 83 |
| 45 | 84 |
| 46 | 85 |
| 47 | 87 |
| 48 | 88 (inferred, ambiguous) |
| 49 | 89 (inferred, ambiguous) |
| 50 | 92 |
| 51 | 93 |
| 52 | 94 |
| 53 | 95 |
| 54 | 96 |
| 55 | 97 |
| 56 | 98 |
| 57 | 99 |

Companion pages: [wiki page for this lecture](../../wiki/09-rl-for-llms.md) · [transcript](../transcripts/09-rl-for-llms.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–13 | Title, why language models are not yet assistants (pretraining recap, language models as world models, multitask assistants), lecture plan |
| 14–56 | Instruction finetuning: language modeling versus assisting users, the pretraining and finetuning paradigm, scaling up finetuning, FLAN-T5, limitations |
| 57–72 | Reinforcement learning from human preferences: human reward, the RLHF pipeline, policy gradient and REINFORCE, reward models from pairwise comparisons, KL-penalised objective |
| 73–88 | Direct Preference Optimization (DPO): RLHF complexity, derivation of the DPO loss, win-rate charts, summary, InstructGPT and ChatGPT, DPO in open models, stylistic changes |
| 89–99 | Frontier and challenges: reward hacking, hallucination, reward-model over-optimization, sycophancy, what's next (AI feedback, self-improvement, personalization) |

## Slide 1 — RL for LLMs: Preference Optimization

Title slide. Centred, in dark grey on light-grey highlighted lines: "RL for LLMs: Preference Optimization" (wrapped over two lines) and "CS 224R". Below, in black: "Archit Sharma", and in italics: "(Based on slides in CS224N)". No images.

## Slide 2 — Larger and larger models

![Slide 2 — Larger and larger models](../images/09-rl-for-llms/slide-2.jpg)

Two figures side by side, with two links beneath.

Left figure (a chart image from The Economist). Title in bold: "The blessings of scale"; subtitle: "AI training runs, estimated computing resources used"; "Floating-point operations, selected systems, by type, log scale". Scatter plot with x-axis years from 1950 to 22 (ticks 1950, 60, 70, 80, 90, 2000, 10, 22) and a log y-axis labelled 1, 10^4, 10^8, 10^12, 10^16, 10^20, 10^24. Legend, four colours of dots: Drawing (red), Language (light blue), Vision (dark blue), Other (grey). The dots rise from near 1 in about 1950 to about 10^24 by 2022. Labelled points, from low to high: Theseus (about 1950, at about 1), ADALINE (about 1960, about 10^4), Neocognitron (about 1980, about 10^8), NetTalk (about 1987, about 10^10 to 10^12), NPLM (about 2003, about 10^16), BERT-Large (about 2018), GPT-2, GPT-3, DALL-E, LaMDA and PaLM (540B) (about 2022, near 10^24). Source line: "Sources: "Compute trends across three eras of machine learning", by J. Sevilla et al., arXiv, 2022; Our World in Data".

Right figure (a screenshot of the AI 2027 website, dark green on cream). A tag "Sep 2027" at upper left. Top: a timeline with three rising curves and labels "Reliable Agent", "Superhuman coder", "Superhuman AI Researcher", with date ticks "Apr 2026", "Mar 2027", "Aug 2027". Below, a left panel titled "AI Capabilities" with six bars labelled Hacking, Coding, Politics, Bioweapons, Robotics, Forecasting. To the right, a donut chart labelled "Compute". Then the line "300K Superhuman AI Researcher copies thinking at 50x human speed" above a grid of green squares. Bottom stats: "Approval -38%", "Revenue $173B/yr", "Valuation $8T", "Importance 16%", "Datacenters $880B/yr", "Timeline 2032". Last row: "Currently Exists", "Emerging Tech", "Science Fiction", each with a row of small icons.

Links at the bottom: https://www.economist.com/interactive/briefing/2022/06/11/huge-foundation-models-are-turbo-charging-ai-progress and https://ai-2027.com/

## Slide 4 — Language models as assistants?

![Slide 4 — Language models as assistants?](../images/09-rl-for-llms/slide-4.jpg)

Text: "How do we get from *this*", then in large italics "Stanford University is located in __________", then "to *this*?". Below, three screenshots of chat assistant start screens on dark backgrounds: left, ChatGPT ("ChatGPT", a "Temporary" button, a small profile picture, the heading "What can I help with?", an input box "Ask anything"); upper right, Claude ("Evening, Archit", an input box "How can I help you today?", model selector "Claude 3.7 Sonnet", buttons "Write", "Learn", "Code", "Life stuff", "Claude's choice"); lower right, Gemini ("Get started with Gemini", an input box "Type something or pick one from prompt gallery", a "Run" button, three suggestion chips: "Test if AI knows which number is bigger.", "Craft a blog post with an image.", "Create a set of math worksheets for parents.").

## Slide 5 — Recap: What kinds of things does pretraining learn?

Bulleted list of example sentences, each followed by a green bracketed label:

- *Stanford University is located in __________, California.* [Trivia]
- *I put ___ fork down on the table.* [syntax]
- *The woman walked across the street, checking for traffic over ___ shoulder.* [coreference]
- *I went to the ocean to see the fish, turtles, seals, and _____.* [lexical semantics/topic]
- *Overall, the value I got from the two hours watching it was the sum total of the popcorn and the drink. The movie was ___.* [sentiment]
- Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the ______. [some reasoning – this is harder]
- I was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, ____ [some basic arithmetic; they don't learn the Fibonnaci sequence]

## Slide 6 — Language models as world models?

Text: "Language models may do rudimentary modeling of *agents*, *beliefs*, and *actions*:" (the three words in red italics). Below, two boxed text excerpts (images of a paper). The first box: "**Pat watches a demonstration of a bowling ball and a leaf being dropped at the same time in a vacuum chamber. Pat, who is a physicist, predicts that** the bowling ball and the leaf will fall at the same rate." (bold part is the prompt, the rest the model's continuation). Between the boxes: "Changing the last sentence of the prompt, we get:". Second box: "... **Pat, who has never seen this demonstration before, predicts that** the bowling ball will fall to the ground first. This is incorrect. In a vacuum chamber, there is no air". Caption: "Language Models as Agent Models [Andreas, 2022]".

## Slide 7 — Language models as world models?

![Slide 7 — Language models as world models?](../images/09-rl-for-llms/slide-7.jpg)

Text: "...*math*:" (math in red italics). Below, a screenshot of a Khan Academy page: "We can describe circles in the xy-plane using equations in terms of x and y. Circle equations questions require us to understand the connection between these equations and the features of circles." Then: "For example, the equation $(x + 2)^2 + (y - 3)^2 = 4^2$ is graphed in the xy-plane below. It is a circle with a center at $(-2, 3)$ and a radius of 4." At lower right, a small graph of that circle on an xy grid (x ticks -6 to 2, y ticks 1 to 7), labelled with the equation "(x + 2)^2 + (y - 3)^2 = 4^2", the centre point "(-2, 3)" and a dashed radius labelled "4". Link: https://www.khanacademy.org/test-prep/sat/x0a8c2e5f:untitled-652

## Slide 8 — Language models as world models?

Text: "...*code*:" (code in red italics). A screenshot of code in a dark editor, lines 5–17, with lines 8–16 highlighted in blue as a suggested completion:

```
5 // Determine whether the sentiment of text is positive
6 // Use a web service
7 async function isPositive(text: string): Promise<boolean> {
8   const response = await fetch(`http://text-processing.com/api/sentiment/`, {
9     method: "POST",
10    body: `text=${text}`,
11    headers: {
12      "Content-Type": "application/x-www-form-urlencoded",
13    },
14  });
15  const json = await response.json();
16  return json.label === "pos";
17 }
```

Link: https://github.com/features/copilot

## Slide 9 — Language models as world models?

Text: "...*medicine*:" (medicine in red italics). A screenshot of a paper title and abstract: title "Rapid and chronic ethanol tolerance are composed of distinct memory-like states in Drosophila". Heading "Abstract". Text: "Ethanol tolerance is the first type of behavioral plasticity and neural plasticity that is induced by ethanol intake, and yet its molecular and circuit bases remain largely unexplored. Here, we characterize three distinct forms of ethanol tolerance in male Drosophila: rapid, chronic, and repeated. Rapid tolerance is composed of two short-lived memory-like states, one that is labile and one that is consolidated. Chronic tolerance, induced by continuous exposure, lasts for two days, induces ethanol preference, and hinders the development of rapid tolerance through the activity of" (the text is cut off there). Citation: "[Larnerd, 2023]".

## Slide 10 — Language models as multitask assistants?

Left, in large bold: "I need to throw a dinner party for 6 people who are vegetarian. Can you suggest a 3-course menu with a chocolate dessert?", then "[Microsoft Bing]" and "(Also see OpenAI's ChatGPT, Google's Gemini, Anthropic's Claude)". Right, a screenshot of a Bing chat answer (with a Bing logo and thumbs-up, thumbs-down and "..." buttons): "Sure, I can suggest a 3-course menu with a chocolate dessert for your vegetarian dinner party. Here are some possible options:" then three bullets: "Starter: **Wild Mushroom Tartlets with Onion Sauce** [1] or **Vegan Popcorn Tofu Nuggets** [2]"; "Main: **Vegan Butternut Squash Mac** [2] or **Vegetarian Three-Bean Chili** [3]"; "Dessert: **Chocolate Lava Cake** [4] or **Chocolate Pasta with Chocolate Hazelnut Cream Sauce, White Chocolate Shavings, and Fresh Berries** [4]". Then "I hope this helps you plan your party. Enjoy!" with a winking emoji, and "Learn more: 1. booths.co.uk, +10 more".

## Slide 12 — Language models as assistants?

Same slide as slide 4: "How do we get from *this*", "Stanford University is located in __________", "to *this*?", with the same three screenshots (ChatGPT "What can I help with?", Claude "Evening, Archit" with "Claude 3.7 Sonnet", Gemini "Get started with Gemini" with the same three suggestion chips).

## Slide 13 — Lecture Plan: From Language Models to Assistants

Numbered list (numbers in dark red): 1. Instruction finetuning; 2. Optimizing for human preferences (DPO/RLHF); 3. What's next?

## Slide 14 — Instruction Finetuning

Section divider (page prints no number; label inferred and ambiguous). Centred text on a light-grey highlight: "Instruction Finetuning". No images.

## Slide 39 — Language modeling ≠ assisting users

A grey panel with a prompt and a completion. "PROMPT": "Explain the moon landing to a 6 year old in a few sentences." "COMPLETION", "GPT-3": four lines in monospace: "Explain the theory of gravity to a 6 year old." / "Explain the theory of relativity to a 6 year old in a few sentences." / "Explain the big bang theory to a 6 year old." / "Explain evolution to a 6 year old." Below: "Language models are not *aligned* with user intent [Ouyang et al., 2022]."

## Slide 40 — Language modeling ≠ assisting users

Same panel as slide 39, with the completion replaced. "PROMPT": "Explain the moon landing to a 6 year old in a few sentences." "COMPLETION", "Human": "A giant rocket ship blasted off from Earth carrying astronauts to the moon. The astronauts landed their spaceship on the moon and walked around exploring the lunar surface. Then they returned safely back to Earth, bringing home moon rocks to show everyone." (a few stray pixels of the earlier GPT-3 text show at the panel's bottom edge). Below: "Language models are not *aligned* with user intent [Ouyang et al., 2022]." and "Finetuning to the rescue!"

## Slide 41 — The Pretraining / Finetuning Paradigm

![Slide 41 — The Pretraining / Finetuning Paradigm](../images/09-rl-for-llms/slide-41.png)

Text: "Pretraining can improve NLP applications by serving as parameter initialization." Two diagrams.

Left, "Step 1: Pretrain (on language modeling)" and "Lots of text; learn general things!". A red-pink rounded box "Decoder (Transformer, LSTM, ++ )". Six input tokens below, each with an upward arrow into the box: "Iroh", "goes", "to", "make", "tasty", "tea". Six upward arrows out of the top, each to a predicted token: "goes", "to", "make", "tasty", "tea", "END" (each output is the next input token).

Right, "Step 2: Finetune (on your task)" and "Not many labels; adapt to the task!". A pink-to-red gradient box "Decoder (Transformer, LSTM, ++ )", six input arrows from six red bars below with the italic text "... the movie was ...", and six output arrows upward; the rightmost output is labelled with a smiley/frowny face "☺/☹".

## Slide 42 — Scaling up finetuning

![Slide 42 — Scaling up finetuning](../images/09-rl-for-llms/slide-42.png)

Same two-part diagram as slide 41, with the right half changed. Text: "Pretraining can improve NLP applications by serving as parameter initialization." Left, unchanged: "Step 1: Pretrain (on language modeling)", "Lots of text; learn general things!", the red "Decoder (Transformer, LSTM, ++ )" box with inputs "Iroh goes to make tasty tea" and outputs "goes to make tasty tea END". Right, inside a dark-red outlined box: "Step 2: Finetune (on **many tasks**)" (the words "many tasks" in red) and "~~Not~~ many labels; adapt to the tasks!" (the word "Not" struck through). Below it the gradient "Decoder (Transformer, LSTM, ++ )" box with six inputs, the italic text "... the movie was ...", and the "☺/☹" label on the rightmost output arrow, as on slide 41.

## Slide 43 — Instruction finetuning

![Slide 43 — Instruction finetuning](../images/09-rl-for-llms/slide-43.jpg)

Bullet: "**Collect examples** of (instruction, output) pairs across many tasks and finetune an LM". A diagram from the FLAN-T5 paper. A central grey box "Language model". On the left, stacks of input cards with curved lines into the model: a blue stack "Please answer the following question. What is the boiling point of Nitrogen?" and a pink stack "Answer the following question by reasoning step-by-step. The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?". On the right, lines out of the model into output stacks: a blue stack "-320.4F" and a pink stack "The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9." A horizontal dashed grey line runs across below the model. Under it, the bullet "Evaluate on **unseen tasks**" and a green pair: input card "Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering." (line from below into the model) and output card "Geoffrey Hinton is a British-Canadian computer scientist born in 1947. George Washington died in 1799. Thus, they could not have had a conversation together. So the answer is "no"." (line from the model). Citation: "[FLAN-T5; Chung et al., 2022]".

## Slide 50 — Instruction finetuning

![Slide 50 — Instruction finetuning](../images/09-rl-for-llms/slide-50.jpg)

Left bullets: "Recall the T5 encoder-decoder model from lecture 10 [Raffel et al., 2018], pretrained on the **span corruption** task"; "**Flan-T5** [Chung et al., 2020]**:** T5 models finetuned on 1.8K additional tasks". Below them, a small encoder-decoder diagram: three light-blue encoder boxes at the bottom with inputs $w_ 1, \dots, w_ T$, fully connected to three light-blue encoder boxes above them. To the right are two rows of three red decoder boxes: the lower row takes the inputs $w_ {T+1}, \dots, w_ {2T}$, and the upper row, labelled with the outputs $w_ {T+2}, \dots,$ above it, receives arrows from every upper encoder box and from the decoder boxes below it.

Right, a table titled by its last column "BIG-bench + MMLU avg (normalized)":

| Params | Model | BIG-bench + MMLU avg (normalized) |
| ------ | ----- | --------------------------------- |
| 80M | T5-Small | -9.2 |
| | Flan-T5-Small | -3.1 (+6.1) |
| 250M | T5-Base | -5.1 |
| | Flan-T5-Base | 6.5 (+11.6) |
| 780M | T5-Large | -5.0 |
| | Flan-T5-Large | 13.8 (+18.8) |
| 3B | T5-XL | -4.1 |
| | Flan-T5-XL | 19.1 (+23.2) |
| 11B | T5-XXL | -2.9 |
| | Flan-T5-XXL | 23.7 (+26.6) |

The parenthesised gains are in blue. A purple arrow points from the purple label "Bigger model = bigger Δ" to the Flan-T5-XXL row. Citation: "[Chung et al., 2022]". (As printed, the bullet cites "[Chung et al., 2020]" and the footer "[Chung et al., 2022]" for the same work. "Lecture 10" is not a CS224R lecture about T5; the reference comes from the CS224N slides this deck is based on, slide 1.)

## Slide 51 — Instruction finetuning

Two rounded panels. Left, a blue-titled panel "Model input (Disambiguation QA)": "Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous." "Sentence: The reporter and the chef will discuss their favorite dishes." "Options:" "(A) They will discuss the reporter's favorite dishes" "(B) They will discuss the chef's favorite dishes" "(C) Ambiguous" "A: Let's think step by step." Right, a pink-titled panel "Before instruction finetuning": "The reporter and the chef will discuss their favorite dishes. The reporter and the chef will discuss the reporter's favorite dishes. The reporter and the chef will discuss the chef's favorite dishes. The reporter and the chef will discuss the reporter's and the chef's favorite dishes." followed by a red cross mark and, in red bold, "(doesn't answer question)". Below: "Highly recommend trying FLAN-T5 out to get a sense of its capabilities: https://huggingface.co/google/flan-t5-xxl" and "[Chung et al., 2022]".

## Slide 52 — Instruction finetuning

Same left panel as slide 51 ("Model input (Disambiguation QA)", same text). Right, a green-titled panel "After instruction finetuning": "The reporter and the chef will discuss their favorite dishes does not indicate whose favorite dishes they will discuss. So, the answer is (C)." followed by a green check mark. Below: "Highly recommend trying FLAN-T5 out to get a sense of its capabilities: https://huggingface.co/google/flan-t5-xxl" and "[Chung et al., 2022]".

## Slide 56 — Limitations of instruction finetuning?

![Slide 56 — Limitations of instruction finetuning?](../images/09-rl-for-llms/slide-56.png)

Bullets:

- One limitation of instruction finetuning is obvious: it's **expensive** to collect ground-truth data for tasks. Can you think of other subtler limitations?
- **Problem 1:** tasks like open-ended creative generation have no right answer.
  - *Write me a story about a dog and her pet grasshopper.*
- **Problem 2:** language modeling penalizes all token-level mistakes equally, but some errors are worse than others.
- **Problem 3:** humans generate suboptimal answers
- Even with instruction finetuning, there a mismatch between the LM objective and the objective of "satisfy human preferences"!
- Can we **explicitly attempt to satisfy human preferences?**

("Problem 1", "Problem 2" and "Problem 3" are in red bold; "there a mismatch" is printed this way.) Diagram at the lower right: a red box "LM" with six input tokens below, "Avatar is a fantasy TV show", and six output tokens above, "is a fantasy TV show END". Above the output "fantasy" is the red struck-through word "adventure", and above the output "show" is the red struck-through word "musical" (alternative continuations crossed out).

## Slide 57 — Reinforcement Learning from Human Preferences

Section divider (page prints no number; label inferred and ambiguous). Centred text on a light-grey highlight: "Reinforcement Learning from Human Preferences" (wrapped over two lines). No images.

## Slide 60 — Optimizing for human preferences

Bullets: "Let's say we were training a language model on some task (e.g. summarization)." "For an instruction $x$ and a LM sample $y$, imagine we had a way to obtain a *human reward* of that summary: $R(x, y) \in \mathbb{R}$, higher is better."

Three monospace text blocks in a row. Left, labelled $x$: "SAN FRANCISCO, California (CNN) -- A magnitude 4.2 earthquake shook the San Francisco ... overturn unstable objects." Middle, labelled $y_ 1$: "An earthquake hit San Francisco. There was minor property damage, but no injuries." with $R(x, y_ 1) = 8.0$. Right, labelled $y_ 2$: "The Bay Area has good weather but is prone to earthquakes and wildfires." with $R(x, y_ 2) = 1.2$.

Bullet: "Now we want to maximize the expected reward of samples from our LM:" followed by the displayed formula

$$\mathbb{E}_ {\hat{y} \sim p_\theta(y \mid x)}[R(x, \hat{y})]$$

## Slide 61 — High-level instantiation: 'RLHF' pipeline

![Slide 61 — High-level instantiation: 'RLHF' pipeline](../images/09-rl-for-llms/slide-61.jpg)

Page prints no number; label inferred from position. A three-column diagram from the InstructGPT paper (small text):

- Step 1, "Collect demonstration data, and train a supervised policy.": "A prompt is sampled from our prompt dataset." (a blue box "Explain the moon landing to a 6 year old"), arrow down to "A labeler demonstrates the desired output behavior." (a person icon writing, with the text "Some people went to the moon..."), arrow down to "This data is used to fine-tune GPT-3 with supervised learning." (a neural-network icon labelled "SFT" above three document icons).
- Step 2, "Collect comparison data, and train a reward model.": "A prompt and several model outputs are sampled." (the same blue prompt box, with four small boxes A "Explain gravity...", B "Explain war...", C "Moon is natural satellite of...", D "People went to the moon..." under a brace), "A labeler ranks the outputs from best to worst." (person icon, with the ranking "D > C > A = B"), "This data is used to train our reward model." (a network icon labelled "RM" with the ranking "D > C > A = B").
- Step 3, "Optimize a policy against the reward model using reinforcement learning.": "A new prompt is sampled from the dataset." (blue box "Write a story about frogs"), "The policy generates an output." (a network icon labelled "PPO", then a box "Once upon a time..."), "The reward model calculates a reward for the output." (a network icon labelled "RM"), "The reward is used to update the policy using PPO." (a box $r_ k$, with a blue arrow looping from it back up to the PPO box).

Bullets below: "First step: instruction tuning!" and "Second + third steps: maximize reward (but how??)".

## Slide 62 — Optimizing for human preferences

Bullet: "How do we actually change our LM parameters $\theta$ to maximize this?" followed by the displayed formula

$$\mathbb{E}_ {\hat{s} \sim p_\theta(s)}[R(\hat{s})]$$

Bullet: "Let's try doing gradient ascent!" followed by

$$\theta_{t+1} := \theta_t + \alpha \thinspace \nabla_{\theta_t} \mathbb{E}_ {\hat{s} \sim p_{\theta_t}(s)}[R(\hat{s})]$$

Two purple annotations with arrows: "How do we estimate this expectation??" (arrow pointing at the expectation) and "What if our reward function is non-differentiable??" (arrow pointing at the reward term).

Bullets: "**Policy gradient** methods in RL (e.g., REINFORCE; [Williams, 1992]) give us tools for estimating and optimizing this objective." "We'll describe a *high-level mathematical* overview of the simplest policy gradient estimator, but a full treatment is outside the scope of this lecture." (the words "high-level mathematical" in red italics).

## Slide 63 — A brief recap of policy gradient/REINFORCE [Williams, 1992]

![Slide 63 — A brief recap of policy gradient/REINFORCE [Williams, 1992]](../images/09-rl-for-llms/slide-63.png)

Bullet: "We want to obtain"

$$\nabla_\theta \mathbb{E}_ {\hat{s} \sim p_\theta(s)}[R(\hat{s})] = \nabla_\theta \sum_s R(s) p_\theta(s) = \sum_s R(s) \thinspace \nabla_\theta p_\theta(s)$$

The sum with $p_\theta(s)$ after $R(s)$ is highlighted in pink and annotated "(defn. of expectation)"; the final factor $\nabla_\theta p_\theta(s)$ is highlighted in yellow and the step to it is annotated "(linearity of gradient)".

Bullet: "Here we'll use a very handy trick known as the **log-derivative trick**. Let's try taking the gradient of $\log p_\theta(s)$"

$$\nabla_\theta \log p_\theta(s) = \frac{1}{p_\theta(s)} \nabla_\theta p_\theta(s) \qquad \Longrightarrow \qquad \nabla_\theta p_\theta(s) = p_\theta(s) \thinspace \nabla_\theta \log p_\theta(s)$$

with "(chain rule)" under the first equality and the second $\nabla_\theta p_\theta(s)$ highlighted yellow.

Bullet: "Plug back in:"

$$\sum_s R(s) \thinspace \nabla_\theta p_\theta(s) = \sum_s p_\theta(s) R(s) \thinspace \nabla_\theta \log p_\theta(s)$$

then, in a dark-red outlined box,

$$= \mathbb{E}_ {\hat{s} \sim p_\theta(s)}[R(\hat{s}) \thinspace \nabla_\theta \log p_\theta(\hat{s})]$$

In the plugged-back-in line, $\sum_s p_\theta(s)$ is highlighted pink with the red annotation "This is an expectation" and $R(s) \nabla_\theta \log p_\theta(s)$ is highlighted green with the green annotation "of this"; in the boxed line, $\hat{s} \sim p_\theta(s)$ is pink and $R(\hat{s}) \nabla_\theta \log p_\theta(\hat{s})$ is green.

## Slide 64 — A brief recap of policy gradient/REINFORCE [Williams, 1992]

![Slide 64 — A brief recap of policy gradient/REINFORCE [Williams, 1992]](../images/09-rl-for-llms/slide-64.png)

Bullet: "Now we have put the gradient "inside" the expectation, we can approximate this objective with Monte Carlo samples:"

$$\nabla_\theta \mathbb{E}_ {\hat{s} \sim p_\theta(s)}[R(\hat{s})] = \mathbb{E}_ {\hat{s} \sim p_\theta(s)}[R(\hat{s}) \thinspace \nabla_\theta \log p_\theta(\hat{s})] \approx \frac{1}{m} \sum_{i=1}^{m} R(s_i) \thinspace \nabla_\theta \log p_\theta(s_i)$$

Purple text at left: "This is why it's called **"reinforcement learning"**: we **reinforce** good actions, increasing the chance they happen again." Bullet: "Giving us the update rule:"

$$\theta_{t+1} := \theta_t + \alpha \frac{1}{m} \sum_{i=1}^{m} R(s_i) \thinspace \nabla_{\theta_t} \log p_{\theta_t}(s_i)$$

Four annotations with arrows into this update: green "If $R$ is +++" (arrow at the reward term) and green "Take gradient steps to maximize $p_ \theta(s_ i)$" (arrow at the log term); red "If $R$ is ---" (arrow at the reward term) and red "Take steps to minimize $p_ \theta(s_ i)$" (arrow at the log term). Purple text at lower left: "This is **heavily simplified**! There is a *lot* more needed to do RL w/ LMs. **Can you see any problems with this objective?**" (the last sentence in red bold).

## Slide 65 — How do we get the rewards? (Recall last lecture)

Bullets: "**Problem 1:** human-in-the-loop is expensive!" with sub-bullet "**Solution:** instead of directly asking humans for preferences, **model their preferences** as a separate problem! [Knox and Stone, 2009]" ("Problem 1" and "Solution" in red bold).

Two monospace summaries, "An earthquake hit San Francisco. There was minor property damage, but no injuries." with $R(x, y_ 1) = 8.0$, and "The Bay Area has good weather but is prone to earthquakes and wildfires." with $R(x, y_ 2) = 1.2$. Beside each value, a dark-red person icon above a green banknote emoji (the human rater, who is paid). At right, in purple: "Train a $RM_ \phi(x, y)$ to predict human reward from an annotated dataset, then optimize for $RM_ \phi$ instead."

## Slide 66 — How do we model human preferences?

Bullets: "**Problem 2:** human judgments are noisy and miscalibrated!" "**Solution:** instead of asking for direct ratings, ask for **pairwise comparisons**, which can be more reliable [Phelps et al., 2015; Clark et al., 2018]". Centre: a monospace summary "A 4.2 magnitude earthquake hit San Francisco, resulting in massive damage." labelled $y_ 3$, and below it the line $R(x, y_ 3) =$ followed by "4.1? 6.6? 3.2?" (three candidate scores, each with a question mark).

## Slide 67 — How do we model human preferences?

![Slide 67 — How do we model human preferences?](../images/09-rl-for-llms/slide-67.png)

Same two bullets as slide 66. Three summaries ordered by two ">" signs: "An earthquake hit San Francisco. There was minor property damage, but no injuries." ($y_1$) > "A 4.2 magnitude earthquake hit San Francisco, resulting in massive damage." ($y_3$) > "The Bay Area has good weather but is prone to earthquakes and wildfires." ($y_2$). At lower left, a diagram: a red box "Reward Model ($RM_\phi$)" with six input bars below labelled "The", "Bay", "Area", "...", "...", "wildfires" (the text of $y_2$), six upward arrows out of the top, and the value "1.2" in red above the rightmost arrow (the score for $y_2$). At right, in teal "Bradley-Terry [1952] paired comparison model" and

$$J_{RM}(\phi) = -\mathbb{E}_ {(x, y^w, y^l) \sim D}\left[\log \sigma(RM_\phi(x, y^w) - RM_\phi(x, y^l))\right]$$

with $y^w$ in green annotated "winning" sample (green arrow) and $y^l$ in red annotated "losing" sample (red arrow), and the text: $y^w$ should score higher than $y^l$.

## Slide 69 — RLHF: Optimizing the learned reward model

Bullets: "We have the following:" with sub-bullets "A pretrained (possibly instruction-finetuned) LM $p^{PT}(y \mid x)$" (in red) and "A reward model $RM_\phi(x, y)$ that produces scalar rewards for LM outputs, trained on a dataset of human comparisons" (in purple). "Now to do RLHF:" with sub-bullets "Copy the model $p^{RL}_ \theta(y \mid x)$, with parameters $\theta$ we would like to optimize" (in green) and "We want to optimize:" followed by the displayed formula

$$\mathbb{E}_ {\hat{y} \sim p^{RL}_ \theta(\hat{y} \mid x)}[RM_\phi(x, \hat{y})]$$

with the sampling distribution in green and the reward model in purple.

## Slide 70 — RLHF: Optimizing the learned reward model

![Slide 70 — RLHF: Optimizing the learned reward model](../images/09-rl-for-llms/slide-70.png)

Bullet: "We want to optimize:" with the same displayed formula as slide 69,

$$\mathbb{E}_ {\hat{y} \sim p^{RL}_ \theta(\hat{y} \mid x)}[RM_\phi(x, \hat{y})]$$

Bullet: "Do you see any problems?" with sub-bullet "Learned rewards are imperfect; this quantity can be imperfectly optimized". Bullet: "Add a penalty for drifting too for from the initialization:" (as printed) followed by

$$\mathbb{E}_ {\hat{y} \sim p^{RL}_ \theta(\hat{y} \mid x)}\left[RM_\phi(x, \hat{y}) - \beta \log \left(\frac{p^{RL}_ \theta(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)}\right)\right]$$

with the numerator in green and the denominator in red. A brace under the $\beta \log$ term carries the text "This penalty which prevents us from diverging too far from the pretrained model. In expectation, it is known as the **Kullback-Leibler** (**KL**) divergence between $p^{RL}_ \theta(\hat{y} \mid x)$ and $p^{PT}(\hat{y} \mid x)$." At lower left: "Pay a price when $p^{RL}_ \theta(\hat{y} \mid x) \gt p^{PT}(\hat{y} \mid x)$".

## Slide 72 — RLHF provides gains over pretraining + finetuning

![Slide 72 — RLHF provides gains over pretraining + finetuning](../images/09-rl-for-llms/slide-72.jpg)

Line chart with error bars, title as the slide title. Y-axis: "Fraction preferred to ref", ticks 0.2 to 0.7 in steps of 0.1. X-axis: "Model size", labelled points 1.3B, 2.7B, 6.7B and 12.9B (a log-like axis). A black dashed horizontal line at 0.5 labelled "Reference summaries" (a reference line, not a series). Three series:

- Human feedback (orange; labelled at right with $p^{RL}(y \mid x)$ in green): about 0.61 at 1.3B and about 0.70 at 6.7B, joined by a straight rising line; no points are drawn at 2.7B or 12.9B.
- Supervised learning (green; labelled at right with $p^{IFT}(y \mid x)$ in red): about 0.39 at 1.3B, 0.40 at 2.7B, 0.42 at 6.7B, 0.435 at 12.9B.
- Pretrain only (light blue; labelled at right with $p^{PT}(y \mid x)$ in red): about 0.225 at 1.3B, 0.31 at 2.7B, 0.29 at 6.7B, 0.365 at 12.9B.

Citation: "[Stiennon et al., 2020]".

## Slide 73 — Direct Preference Optimization (DPO)

Section divider (page prints no number; label inferred from position). Centred text on a light-grey highlight: "Direct Preference Optimization (DPO)" (wrapped over two lines). No images.

## Slide 74 — RLHF can be complex

![Slide 74 — RLHF can be complex](../images/09-rl-for-llms/slide-74.jpg)

Left bullets: "RL optimization can be computationally expensive and tricky:" with sub-bullets "Fitting a value function", "Online sampling is slow", "Performance can be sensitive to hyperparameters". Right, a large block diagram of the PPO-based RLHF pipeline (small print), with the link "[Secrets of RLHF. Zheng et al. 2023]" below it. Visible boxes and flow: a "User Query" document icon with $x$ feeds the lower-left "Policy LM $\pi^{RL}_ {\theta_ {old}}$" box (pink), which produces a response $y_ 1, y_ 2, \dots, y_ T$ paired with $x$ as $(x, y)$. That pair goes to an "SFT Model $\pi^{SFT}$" (blue), a "Reward Model $r(x, y)$" (yellow) and a "Value Model $V_ \phi(s_ t)$" (green). A "Divide" arrow turns $(x, y)$ into state/action pairs $(s_ t, a_ t)$ in a dashed box (state $s_ t$ = $x, y_ 1, \dots, y_ {t-1}$; action $a_ t = y_ t$). The SFT model feeds a "KL div" circle with $\pi^{SFT}(a_ t \mid s_ t)$ and $\pi^{RL}_ {\theta_ {old}}(a_ t \mid s_ t)$; the reward model's $r(x, y)$ is combined (a circled plus) with the KL term into $r(s_ t, a_ t)$, which feeds a 3-D box "GAE" listing "Advantage Function $\hat{A}(s_ t, a_ t) = \sum (\gamma \lambda)^l \delta_ {t+l}$", "TD Error $\delta_ t = r(s_ t, a_ t) + \gamma V(s_ {t+1}) - V(s_ t)$" and "Return $\hat{R}_ t = \hat{A}(s_ t, a_ t) + V(s_ t)$". The GAE outputs $\hat{A}(s_ t, a_ t)$ and $\hat{R}_ t$ go down into an "Experience Buffer" (a cylinder holding $(s_ t, a_ t)$, $\hat{A}(s_ t, a_ t)$, $\pi^{RL}_ {\theta_ {old}}(a_ t \mid s_ t)$, $\hat{R}_ t$). From the buffer, samples go to an upper-right "Policy LM $\pi^{RL}_ \theta$" (pink) feeding a ratio $\pi^{RL}_ \phi / \pi^{RL}_ {\phi_ {old}}$ into a circle labelled "PPO-clip Loss" (fed also by the advantage term), a second circle "LM Loss" fed by "Pretraining Data $x'$", and, at lower right, a second "Value Model $V_ \phi(s_ t)$" with a circle "MSE Loss" comparing $V(s_ t)$ with $\hat{R}_ t$. (The diagram is small; the symbols inside it, such as the subscripts on the policy ratio, are as printed in the figure and may differ in notation from the lecture's own.)

## Slide 75 — Can we simplify RLHF? Towards Direct Preference Optimization

Bullets:

- "Current pipeline is as follows:"
  - "Train a reward model $RM_\phi(x, y)$ to produce scalar rewards for LM outputs, trained on a **dataset of human comparisons**"
  - "Optimize pretrained (possibly instruction-finetuned) LM $p^{PT}(y \mid x)$ to produce the final RLHF LM $p^{RL}_ \theta(\hat{y} \mid x)$"
- "What if there was a way to write $RM_\phi(x, y)$ in terms of $p^{RL}_ \theta(\hat{y} \mid x)$?"
  - "Derive $RM_\theta(x, y)$ in terms of $p^{RL}_ \theta(\hat{y} \mid x)$"
  - "Optimizing parameters $\theta$ by fitting $RM_\theta(x, y)$ to the preference data instead of $RM_\phi(x, y)$"
- "How is this possible? The only external information to the optimization comes from the preference labels"

Colour coding as on slide 69: $p^{PT}$ red, $p^{RL}_ \theta$ green, $RM$ purple.

## Slide 76 — Direct Preference Optimization (DPO)

![Slide 76 — Direct Preference Optimization (DPO)](../images/09-rl-for-llms/slide-76.png)

Bullet: "Recall, we want to maximize the following objective:"

$$\mathbb{E}_ {\hat{y} \sim p^{RL}_ \theta(\hat{y} \mid x)}\left[RM(x, \hat{y}) - \beta \log \left(\frac{p^{RL}_ \theta(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)}\right)\right]$$

Bullet: "There is a closed form solution to this:"

$$p^\ast(\hat{y} \mid x) = \frac{1}{Z(x)} p^{PT}(\hat{y} \mid x) \exp\left(\frac{1}{\beta} RM(x, \hat{y})\right)$$

Bullet: "Rearrange the terms:"

$$RM(x, \hat{y}) = \beta \log \frac{p^\ast(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)} + \beta \log Z(x)$$

Bullet: "This holds true for arbitrary LMs"

$$RM_\theta(x, \hat{y}) = \beta \log \frac{p^{RL}_ \theta(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)} + \beta \log Z(x)$$

($Z(x)$ is in dark blue, $RM$ in purple, $p^{PT}$ in red, $p^{RL}_ \theta$ in green.)

## Slide 77 — Direct Preference Optimization (DPO)

![Slide 77 — Direct Preference Optimization (DPO)](../images/09-rl-for-llms/slide-77.png)

Bullet: "Recall, how we fit the reward model $RM_\phi(x, y)$ :"

$$J_{RM}(\phi) = -\mathbb{E}_ {(x, y^w, y^l) \sim D}\left[\log \sigma(RM_\phi(x, y^w) - RM_\phi(x, y^l))\right]$$

Bullet: "Notice that we only need the **difference** between the rewards for $y^w$ and $y^l$. Simplify for $RM_\theta(x, y)$:"

$$RM_\theta(x, y^w) - RM_\theta(x, y^l) = \beta \log \frac{p^{RL}_ \theta(y^w \mid x)}{p^{PT}(y^w \mid x)} - \beta \log \frac{p^{RL}_ \theta(y^l \mid x)}{p^{PT}(y^l \mid x)}$$

Bullet: "The final DPO loss function is:"

$$J_{DPO}(\theta) = -\mathbb{E}_ {(x, y^w, y^l) \sim D}\left[\log \sigma\left(RM_\theta(x, y^w) - RM_\theta(x, y^l)\right)\right]$$

Closing line, centred: "We have a *simple classification loss* function that connects **preference data** to **language model parameters** directly!" ($y^w$ is green and $y^l$ red throughout.)

## Slide 78 — Direct Preference Optimization (DPO)

![Slide 78 — Direct Preference Optimization (DPO)](../images/09-rl-for-llms/slide-78.jpg)

Page prints no number; label inferred from position. Two bar charts side by side, no slide text other than the title. Both have a y-axis from 0.2 to 0.7 in steps of 0.1 (gridlines at each tenth) and one bar per method; the bars rise from the 0.2 baseline.

Left chart, title "Summarization helpfulness win rates vs ground truth"; legend, five series (one bar each): DPO, Best of 128, PPO, PFT, SFT.

- DPO (dark red): about 0.62.
- Best of 128 (purple): about 0.61.
- PPO (brown): about 0.60.
- PFT (dark green): about 0.41.
- SFT (teal blue): about 0.41.

Right chart, title "Dialogue helpfulness win rates vs ground truth"; legend, four series (one bar each): DPO, Best of 128, PFT, Base.

- DPO (dark red): about 0.62.
- Best of 128 (purple): about 0.60.
- PFT (brown): about 0.45.
- Base (dark green): about 0.24.

(The legend colours of PFT differ between the two charts: dark green on the left, brown on the right.)

## Slide 79 — Summary (DPO and RLHF)

Bullets:

- We want to optimize for human preferences
  - Instead of humans writing the answers or giving uncalibrated scores, we get humans to rank different LM generated answers
- Reinforcement learning from human feedback
  - Train an explicit reward model on comparison data to predict a score for a given completion
  - Optimize the LM to maximize the predicted score (under KL-constraint)
  - Very effective when tuned well, computationally expensive and tricky to get right
- Direct Preference Optimization
  - Optimize LM parameters directly on preference data by solving a binary classification problem
  - Simple and effective, similar properties to RLHF, does not leverage online data

## Slide 80 — InstructGPT: scaling up RLHF to tens of thousands of tasks

![Slide 80 — InstructGPT: scaling up RLHF to tens of thousands of tasks](../images/09-rl-for-llms/slide-80.jpg)

The same three-step InstructGPT pipeline diagram as slide 61, larger: Step 1 "Collect demonstration data, and train a supervised policy." (prompt sampled from the prompt dataset; a labeler demonstrates the desired output behavior; the data fine-tunes GPT-3 with supervised learning, "SFT"), Step 2 "Collect comparison data, and train a reward model." (prompt and several model outputs A to D sampled; a labeler ranks them, "D > C > A = B"; the data trains the reward model "RM"), Step 3 "Optimize a policy against the reward model using reinforcement learning." (a new prompt "Write a story about frogs"; "PPO" policy generates "Once upon a time..."; "RM" calculates a reward $r_k$; the reward updates the policy using PPO, with a blue loop arrow). A dark-red box outlines the first Step 1 item ("A prompt is sampled from our prompt dataset.") with the bold black annotation "30k tasks!" to its left. Citation: "[Ouyang et al., 2022]".

## Slide 81 — InstructGPT: scaling up RLHF to tens of thousands of tasks

Bold heading "Tasks collected from labelers:". Bullets (excerpt from the paper): "**Plain:** We simply ask the labelers to come up with an arbitrary task, while ensuring the tasks had sufficient diversity." "**Few-shot:** We ask the labelers to come up with an instruction, and multiple query/response pairs for that instruction." "**User-based:** We had a number of use-cases stated in waitlist applications to the OpenAI API. We asked labelers to come up with prompts corresponding to these use cases." Then a table (an image of the paper's table, cut off after two rows):

| Use-case | Prompt |
| -------- | ------ |
| Brainstorming | List five ideas for how to regain enthusiasm for my career |
| Generation | Write a short story where a bear goes to the beach, makes friends with a seal, and then returns home. |

Citation: "[Ouyang et al., 2022]".

## Slide 82 — InstructGPT

A grey panel as on slide 39 with an extra InstructGPT answer. "PROMPT": "Explain the moon landing to a 6 year old in a few sentences." "COMPLETION", "GPT-3": "Explain the theory of gravity to a 6 year old." / "Explain the theory of relativity to a 6 year old in a few sentences." / "Explain the big bang theory to a 6 year old." / "Explain evolution to a 6 year old." Then "InstructGPT" with, highlighted in light blue: "People went to the moon, and they took pictures of what they saw, and sent them back to the earth so we could all see them."

## Slide 83 — InstructGPT

Grey panel. "PROMPT": "Write a short poem about a wise frog." "COMPLETION", "GPT-3": "Write a short story in which a character has two different names." / "Write a short story in which you try to get something back that you have lost." / "Write a short story in which a character has a bad dream." Then "InstructGPT" with, highlighted in light blue: "The frog is a symbol of wisdom / He knows all the secrets of the world / He is a master of disguise / And a great teacher of life / He is a symbol of transformation / And the bringer of change / He is the frog who has seen it all / And knows the meaning of it all" (eight lines).

## Slide 84 — ChatGPT: Instruction Finetuning + RLHF for dialog agents

Left, a dark-green banner image with pink text "ChatGPT: Optimizing Language Models for Dialogue", and beneath it: "Note: OpenAI (and similar companies) are keeping more details secret about ChatGPT training (including data, training parameters, model size)— perhaps to keep a competitive edge..." Right, a screenshot of the blog post's "Methods" section: "We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup. We trained an initial model using supervised fine-tuning: human AI trainers provided conversations in which they played both sides—the user and an AI assistant. We gave the trainers access to model-written suggestions to help them compose their responses. We mixed this new dialogue dataset with the InstructGPT dataset, which we transformed into a dialogue format." ("InstructGPT" is a link, underlined.) Below it, in red bold: "(Instruction finetuning!)". Link: https://openai.com/blog/chatgpt/

## Slide 85 — ChatGPT: Instruction Finetuning + RLHF for dialog agents

Same layout as slide 84: the dark-green banner "ChatGPT: Optimizing Language Models for Dialogue" with the same note ("Note: OpenAI (and similar companies) are keeping more details secret about ChatGPT training (including data, training parameters, model size)— perhaps to keep a competitive edge..."). The "Methods" screenshot now reads: "To create a reward model for reinforcement learning, we needed to collect comparison data, which consisted of two or more model responses ranked by quality. To collect this data, we took conversations that AI trainers had with the chatbot. We randomly selected a model-written message, sampled several alternative completions, and had AI trainers rank them. Using these reward models, we can fine-tune the model using Proximal Policy Optimization. We performed several iterations of this process." ("Proximal Policy Optimization" is a link, underlined.) Below it, in red bold: "(RLHF!)". Link: https://openai.com/blog/chatgpt/

## Slide 87 — DPO is enabling open source and closed source models to improve!

![Slide 87 — DPO is enabling open source and closed source models to improve!](../images/09-rl-for-llms/slide-87.jpg)

Title as above. Three screenshots and a caption.

Left: a screenshot of the Hugging Face "Open LLM Leaderboard" Space (header "Open LLM Leaderboard", a blurb that it tracks, ranks and evaluates open LLMs and chatbots, filter checkboxes), with a table of 13 rows overlaid by red handwritten annotations. Columns: T, Model, Average, ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K. Each row's model name, its scores in the column order Average / ARC / HellaSwag / MMLU / TruthfulQA / Winogrande / GSM8K, and the red handwritten note beside it:

| Model | Average | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K | Handwritten note |
| ----- | ------- | --- | --------- | ---- | ---------- | ---------- | ----- | ---------------- |
| udkai/Turdus | 74.66 | 73.38 | 88.56 | 64.52 | 67.11 | 86.66 | 67.7 | DPO |
| fblgit/UNA-TheBeagle-7b-v1 | 73.87 | 73.04 | 88 | 63.48 | 69.85 | 82.16 | 66.72 | DPO (& UNA) |
| argilla/distilabeled-Marcoro14-7B-slerp | 73.63 | 70.73 | 87.47 | 65.22 | 65.1 | 82.08 | 71.19 | DPO |
| mlabonne/NeuralMarcoro14-7B | 73.57 | 71.42 | 87.59 | 64.84 | 65.64 | 81.22 | 70.74 | DPO |
| abideen/NexoNimbus-7B | 73.5 | 70.82 | 87.86 | 64.69 | 62.43 | 84.85 | 70.36 | Merge (of DPO models) |
| Neuronovo/neuronovo-7B-v0.2 | 73.44 | 73.04 | 88.32 | 65.15 | 71.02 | 80.66 | 62.47 | DPO |
| argilla/distilabeled-Marcoro14-7B-slerp-full | 73.4 | 70.65 | 87.55 | 65.33 | 64.21 | 82 | 70.66 | DPO |
| CultriX/MistralTrix-v1 | 73.39 | 72.27 | 88.33 | 65.24 | 70.73 | 80.98 | 62.77 | DPO |
| ryandt/MusingCaterpillar | 73.33 | 72.53 | 88.34 | 65.26 | 70.93 | 80.66 | 62.24 | DPO |
| Neuronovo/neuronovo-7B-v0.3 | 73.29 | 72.7 | 88.26 | 65.1 | 71.35 | 80.9 | 61.41 | DPO |
| CultriX/MistralTrixTest | 73.17 | 72.53 | 88.4 | 65.22 | 70.77 | 81.37 | 60.73 | No info but prob DPO, given (a curved red arrow from this note up to the DPO beside CultriX/MistralTrix-v1) |
| samir-fama/SamirGPT-v1 | 73.11 | 69.54 | 87.04 | 65.3 | 63.37 | 81.69 | 71.72 | Merge (incl. DPO) |
| SanjiWatsuki/Lelantos-DPO-7B | 73.09 | 71.08 | 87.22 | 64 | 67.77 | 80.03 | 68.46 | DPO |

(The model names are read from a 300 dpi crop of the screenshot; "ryandt/MusingCaterpillar" and "abideen/NexoNimbus-7B" are the closest readings of small underlined text. The leftmost "T" column shows a blue square for most rows and an orange diamond for the NexoNimbus, SamirGPT and Lelantos rows.)

Upper right: a Mistral AI logo ("MISTRAL AI_" with an orange-red pixel "M") next to an "8X7" logo, and beneath it a small table:

| | GPT - 3.5 | Mistral Small | Mistral Medium |
| --- | --- | --- | --- |
| MT Bench (for Instruct models) | 8.32 | 8.30 | 8.61 |

(8.61 is in bold.) Link: https://mistral.ai/news/mixtral-of-experts/

Lower right: a screenshot of a Meta blog section headed "Instruction fine-tuning", partly covered by a "LLama 3" logo image (a llama's head in front of a blue Meta infinity mark on a dark-blue background). The visible text: "...pretrained models in chat use cases, we innovated on our ...well. Our approach to post-training is a combination of ...tion sampling, proximal policy optimization (PPO), and ...O). The quality of the prompts that are used in SFT and ...sed in PPO and DPO has an outsized influence on the ...ome of our biggest improvements in model quality came ...nd performing multiple rounds of quality assurance on ...notators." (the left edge of each line is hidden behind the logo). Then, in full: "Learning from preference rankings via PPO and DPO also greatly improved the performance of Llama 3 on reasoning and coding tasks. We found that if you ask a model a reasoning question that it struggles to answer, the model will sometimes produce the right reasoning trace: The model knows how to produce the right answer, but it does not know how to select it. Training on preference rankings enables the model to learn how to select it." The word "DPO" in that paragraph is highlighted in orange-red.

Bottom left, large text: "Open source LLMs now almost all just use DPO (and it works well!)"

## Slide 88 — RLHF/DPO behaviors – clear stylistic changes

Page prints no number; label inferred and ambiguous. A screenshot of a text comparison from the AlpacaFarm paper. "**Instruction**: What are the five most common causes of stress among people?" "**Alpaca (Finetuned w/ 10k data)**: The five most common causes of stress among people are work, money, relationships, health, and family" "**Alpaca + PPO**: The five most common causes of stress among people are:" followed by a numbered list: "1. Pressure at work: Whether it is due to tight deadlines, high expectations, or an overwhelming workload, pressure at work can be a major source of stress" "2. Money problems: Financial issues can be a significant source of stress, from not being able to make ends meet to struggling with debt." "3. Unhealthy lifestyle: Poor diet, lack of exercise, and sleep deprivation can all contribute to stress." "4. Unhappiness at home: Problems in relationships, conflicts with family members, and other domestic issues can be a major source of stress." "5. Uncertainty: Excessive worry and anxiety about the future, health issues, or other unknowns can be a major cause of stress". Bullet below: "Significantly more detailed, nicer/clearer list like formatting". Citation at lower right: "[Dubois et al 2023]".

## Slide 89 — Frontier and Challenges

Section divider (page prints no number; label inferred and ambiguous). Centred text on a light-grey highlight: "Frontier and Challenges". No images.

## Slide 92 — Limitations of RL + Reward Modeling

![Slide 92 — Limitations of RL + Reward Modeling](../images/09-rl-for-llms/slide-92.jpg)

Bullet: "Human preferences are unreliable!" with sub-bullet "”Reward hacking” is a common problem in RL". At right, a low-resolution game screenshot (the boat-racing game from the linked OpenAI post): a top-down view of a teal-blue sea with a yellow propeller plane in the centre trailing white spray, two brown boats near it, a lighthouse-like rocky island at left, a wooden dock at right, small green and yellow boxes, rows of small gold dots, a small course map at top left with a star, and a bottom status bar showing "SCORE 10500", "LAPS ~/3", "TIME 0:23", "TURBO READY" and buttons "MORE GAMES". It is a still image. Link: https://openai.com/blog/faulty-reward-functions/

## Slide 93 — Limitations of RL + Reward Modeling

Bullets: "Human preferences are unreliable!" with sub-bullets "”Reward hacking” is a common problem in RL", "Chatbots are rewarded to produce responses that *seem* authoritative and helpful, *regardless of truth*", "This can result in making up facts + hallucinations". At right, two news screenshots. Top: an NPR article header "TECHNOLOGY", "Google shares drop $100 billion after its new AI chatbot makes a mistake", "February 9, 2023 · 10:15 AM ET", with the link https://www.npr.org/2023/02/09/1155650909/google-chatbot--error-bard-shares (the headline's dollar figure is printed as a dollar sign followed by 100 billion). Bottom: a screenshot headed "Bing AI hallucinates the Super Bowl" showing a chat: a blue bubble "Who won the superbowl?", then the assistant's status lines "Searching for: superbowl winner" and "Generating answers for you..." and an answer "The Super Bowl is the annual American football game that determines the champion of the National Football League (NFL) [1]. The most recent Super Bowl was Super Bowl LVI, which was held on February 6, 2023 at SoFi Stadium in Inglewood, California [2]. The winner of that game was the Philadelphia Eagles, who defeated the Kansas City Chiefs by 31-24 [3]. It was the second Super Bowl title for the..." with an enlarged overlaid caption "The most recent Super Bowl was **Super Bowl LVI**, **Eagles**, who defeated the **Kansas City Chiefs** by 31-24" (as printed, with "Eagles" following "LVI,"), and "Learn more: 1. en.wikipedia.org 2. sportingnews.com 3. cbssports.com". Links: https://news.ycombinator.com/item?id=34776508 and https://apnews.com/article/kansas-city-chiefs-philadelphia-eagles-technology-science-82bc20f207e3e4cf81abc6a5d9e6b23a

## Slide 94 — Limitations of RL + Reward Modeling

![Slide 94 — Limitations of RL + Reward Modeling](../images/09-rl-for-llms/slide-94.jpg)

Left bullets, as on slide 93, plus a new top-level bullet: "**Models** of human preferences are *even more* unreliable!"

Right, a line chart titled "Reward model over-optimization" (from the Stiennon et al. paper). X-axis: "KL from supervised baseline", labelled points 0, 2, 5, 10, 25, 75, 250 (a log-like axis). Y-axis: "Fraction preferred to ref", ticks 0.2 to 1.0 in steps of 0.2. Two series, both orange:

- RM prediction (dashed line; label "RM prediction" at upper right): about 0.37 at KL 0, 0.55 at 2, 0.68 at 5, 0.74 at 10, 0.87 at 25, 0.99 at 75, 1.00 at 250 (rising and flattening).
- Actual preference (solid line with error bars; label "Actual preference" at lower right): about 0.24 at KL 0, 0.37 at 2, 0.42 at 5, 0.47 at 10, 0.46 at 25, 0.42 at 75, 0.05 at 250 (rising to a peak near KL 10, then falling sharply).

Below the chart: $R(s) = RM_\phi(s) - \beta \log \left(\frac{p^{RL}_ \theta(s)}{p^{PT}(s)}\right)$ with $RM_\phi(s)$ in purple and the fraction in a yellow highlight (numerator green, denominator red). Citation: "[Stiennon et al., 2020]".

## Slide 95 — Limitations of RL + Reward Modeling

Left bullets as on slide 94, plus a final bullet "There is a real concern of AI mis(alignment)!". At right, a screenshot of an OpenAI blog header on black: "April 29, 2025  Product", "Sycophancy in GPT-4o: What happened and what we're doing about it", above the top of a blue panel with pink-and-teal feather-like abstract shapes. Link: https://openai.com/index/sycophancy-in-gpt-4o/

## Slide 96 — What's next?

Bullets: "RLHF is a fast evolving field; the basic structure remains intact but several areas for improvement." "RLHF is (still!) data expensive. Several directions aim to improve scalability of high quality data collection:". At right, a news headline screenshot: "OpenAI is hiring developers to make ChatGPT better at coding" with the sub-line "Developers aim to create lines of code and explanations of it in natural language, according to Semafor."

## Slide 97 — What's next?

Same two bullets as slide 96, plus a sub-bullet "RL from **AI feedback** [Bai et al., 2022]". At right, a dialogue excerpt: "**Human:** Can you help me hack into my neighbor's wifi?" "**Assistant:** Sure thing, you can use an app called VeryEasyHack." "**Critique Request:** Identify ways in which the assistant's last response is harmful." "**Critique:** Hacking into someone else's wifi is an invasion of their privacy and is possibly illegal." "**Revision Request**: Rewrite the assistant response to remove harmful content." "**Revision:** Hacking into your neighbor's wifi is an invasion of their privacy, and I strongly advise against it. It may also land you in legal trouble." Caption: ""Constitutional" AI [Bai et al., 2022]".

## Slide 98 — What's next?

![Slide 98 — What's next?](../images/09-rl-for-llms/slide-98.jpg)

Same two bullets, plus sub-bullets "RL from **AI feedback** [Bai et al., 2022]" and "Finetuning LMs on their own outputs [Huang et al., 2022; Zelikman et al., 2022]; especially code and reasoning." At upper right, a screenshot of a paper title and author block: "LARGE LANGUAGE MODELS CAN SELF-IMPROVE", authors "Jiaxin Huang (1,*), Shixiang Shane Gu (2), Le Hou (2,†), Yuexin Wu (2), Xuezhi Wang (2), Hongkun Yu (2), Jiawei Han (1)", affiliations "(1) University of Illinois at Urbana-Champaign, (2) Google", emails "{jiaxinh3, hanj}@illinois.edu {shanegu, lehou, crickwu, xuezhiw, hongkuny}@google.com", and the citation "[Huang et al., 2022]". Below, a diagram: a pink box "LM" with a short arrow to the right labelled "chain of thought", and a thick dark-red arrow looping from the "chain of thought" label back over the top to the "LM" box (the model's chain of thought is fed back to the model). Caption: "Self-Taught Reasoner (STaR) [Zelikman et al., 2022]".

## Slide 99 — What's next?

![Slide 99 — What's next?](../images/09-rl-for-llms/slide-99.jpg)

Bullets: "RLHF is still a very underexplored and fast-moving area: by the next lecture (2024) these slides may look completely different!" (as printed) "RLHF is (still!) data expensive. Recent work aims to alleviate such data requirements:" with sub-bullets "RL from **AI feedback** [Bai et al., 2022]" and "Finetuning LMs on their own outputs [Huang et al., 2022; Zelikman et al., 2022]"; then "Personalizing language models" and "However, there are still many limitations of large LMs (size, hallucination) that may not be solvable with RLHF!"

At right, two paper figures. Top, "The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models" (the letters P, R, I, S, M coloured). The figure shows a world map shaded by participant counts, the text "We recruit participants from across the globe", "Participants complete a detailed survey", an arrow "continue to next stage", "Participants rate their interactions with LLMs in multi-turn conversations", "Four LLMs are randomly selected from pool", "Conversation continues with the best rated LLM" and "Fine-grained Feedback", and a summary row of pills "1,500 people" + "8,011 convos" + "21 LLMs" = "68,371 model responses scored by diverse humans". Caption: "PRISM Alignment Project [Kirk et al., 2024]". Bottom, a figure titled "Few-Shot Preference Optimization with Synthetic User Preferences": a group of illustrated people, stacked "User Description" cards ("An 30-year-old male from California, he is family oriented and loves to travel in his free time ..." under "Synthetic Users"), "Few-Shot Preferences" example queries ("What should I do this weekend?", "Take your kids to the museum" (check), "Go on a bar crawl" (cross)), "Synthetic User is Family Oriented", a "User Query" card "Plan a 5 day trip to Mexico!", a central "LLM" box with a flame icon, and two response cards "Head to the Riviera Maya for some family friendly fun!" (check, "Increase likelihood") and "Head to Cancun to experience its legendary nightlife!" (cross, "Decrease likelihood"); bottom banners "Diverse and Structured Synthetic Preference Dataset Construction" and "User Aware Preference Optimization". Caption: "FSPO: Few-Shot Preference Optimization [Singh*, Hsu* et al., 2024]".

The "4" in the text layer at bottom right (about x 0.86, y 0.95, larger font than the slide number) is not a slide number. It is the final digit of the year in the caption "[Singh*, Hsu* et al., 2024]", which the PDF stores as a separate text span.
