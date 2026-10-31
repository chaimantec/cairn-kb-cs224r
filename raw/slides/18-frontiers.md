---
title: "Lecture 18 — Frontiers & Open Problems in Deep RL, and How to Do (Deep RL) Research (slide deck)"
lecture: 18
slides: 53
printed_numbers: "1–51, except 40"
source_pdf: https://cs224r.stanford.edu/spring_2025/slides/18_cs224r_frontiers_how_to_research.pdf
numbering: 1:1 with PDF pages. Pages 1–39 and 41–51 print their number, at bottom centre except page 23, which prints it at bottom right; pages 40, 52 and 53 print none, and because the printed numbers leave no gap, their labels are the page index
figure_audit: Transcribed by Claude Sonnet 5.5 (slides 1–27 and 28–53 by two agents). Eight figure slides (5, 10, 15, 22, 23, 26, 32, 39) were checked by Claude Opus 5.5 against 1400px renders, slide 23's table from a crop of the PDF. Slides 10, 23 (every cell, including the Average row), 26 and 32 agree. Corrections: slide 5's chart has seven rows, not eight (the faded "Ma…" label is the top row's own label, under "Agrees with me"); slide 22's post-training panel (bins 2 and 3 sit above the diagonal, bin 4 on it, bins 5–10 below); slide 39's row of robot arms (about ten). Slide 15's bar shades and slide 10's RLPD line style were added. Slide 28's covered item c was checked from the text layer
images: 15 slides are rendered and shown under their headings (5, 6, 7, 8, 10, 12, 15, 18, 19, 21, 22, 23, 26, 32, 39). Not rendered: 31 (a decorative photograph of an indoor space that the recording never mentions) and 38 (the "Bottlenecks" visual pun, four green bottles, described here); 1, 3 and 29 (title pages); 52 (section title); 4, 9, 11, 13, 16, 20, 24, 28, 33, 41, 45 and 49 (agenda and outline slides); and the text-only slides 2, 14, 17, 25, 27, 30, 34–37, 40, 42–44, 46–48, 50, 51 and 53
---

# Lecture 18 — Frontiers & Open Problems in Deep RL, and How to Do (Deep RL) Research: slide-by-slide

Text and figures of the Stanford CS 224R "Deep Reinforcement Learning" (Spring 2025) Lecture 18 deck, transcribed from the PDF. It is the course's last lecture, given by its instructor, Chelsea Finn; the deck prints no lecturer name. It has two parts: frontiers and open problems in deep RL (slides 3–28) and how to do (deep RL) research (slides 29–53). Cite these as "slide N". Diagrams, photographs and plots are described in prose since the KB is read as text.

The recording is about 71 minutes. Finn opens on the plan for both parts (slides 1, 3 and 4, ≈0:05); slide 2's course reminders (Homework 4 due today, the poster session next Wednesday, the final report due the following Monday with no extensions or late days) are not mentioned. The recording follows the deck in order through the frontiers and open problems (slides 5–28, ≈0:51–32:29), answers a student question on why RLHF hurts calibration (slide 22) at ≈32:29, then covers how to do research (slides 29–52, ≈33:16–1:08:48) and ends on the closing thoughts (slide 52, ≈1:08:48, ending at about 1:10:50). Slide 53's thank-you is not read aloud. Five student questions are answered, at ≈19:22, 32:29, 51:01, 52:33 and 1:01:04. Slide 39's two image panels are frames of videos Finn describes at ≈44:53–45:38.

## Slide numbers vs PDF pages

The deck is 1:1 with its page index: PDF page N is slide N, for all 53 pages. Pages 1–39 and 41–51 print their number, at bottom centre except page 23, which prints it at bottom right (the map script, which reads one corner, missed it; the spans show it). Pages 40, 52 and 53 print none, and because the printed numbers leave no gap, their labels are the page index. Slides 5–8 also carry a second, smaller copy of the number at the lower right.

| PDF page | Slide |
| -------- | ----- |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
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
| 31 | 31 |
| 32 | 32 |
| 33 | 33 |
| 34 | 34 |
| 35 | 35 |
| 36 | 36 |
| 37 | 37 |
| 38 | 38 |
| 39 | 39 |
| 40 | 40 |
| 41 | 41 |
| 42 | 42 |
| 43 | 43 |
| 44 | 44 |
| 45 | 45 |
| 46 | 46 |
| 47 | 47 |
| 48 | 48 |
| 49 | 49 |
| 50 | 50 |
| 51 | 51 |
| 52 | 52 |
| 53 | 53 |

## Sections of the deck

| Section | Slides |
| ------- | ------ |
| Title; course reminders; part 1 title; the plan for part 1 | 1–4 |
| Problem set-up: non-rewarding, non-verifiable domains (chatbot preferences, robot shirt folding, YouTube recommendations, scientific reasoning, optimizing for learning) | 5–8 |
| Methods: leveraging prior data and knowledge | 9–10 |
| Methods: using world models and video generation models | 11–12 |
| Methods: how to scale (longer horizons, value functions at scale, batch online RL) | 13–15 |
| Deployment and evaluation: safety | 16–19 |
| Deployment and evaluation: handling inaccuracies and hallucinations (human-AI systems, calibration, reliability) | 20–23 |
| Deployment and evaluation: evaluation of generalist systems | 24–27 |
| End of part 1: "You are all now well-equipped to start tackling these challenges!" | 28 |
| Part 2 title; preface; backstory; some realities of research; outline | 29–33 |
| What to work on (a three-step process, idea- versus problem-driven research, bottlenecks, crossing topic boundaries, perfectionism) | 34–40 |
| How to do the work (handling risk, getting things to work, deciding when to pivot) | 41–44 |
| How to share the work (why, how, writer's block) | 45–48 |
| Misc: mentorship and confidence | 49–51 |
| Closing thoughts; the last-lecture thank-you | 52–53 |

## Slide 1 — Frontiers & open problems in deep RL + How to do (deep RL) research

Title slide. Centred text on three lines: "Frontiers & open problems in deep RL", "+", "How to do (deep RL) research". Below, smaller: "CS 224R".

## Slide 2 — Course reminders

Title "Course reminders". Three dash bullets: "Homework 4 due today"; "Poster session next Wednesday"; "Final report due following Monday". To the right of the third bullet, on the same line: "<- no extensions or late days".

## Slide 3 — Section title: Frontiers & Open Problems in Deep RL

Section title slide. Centred: "Frontiers & Open Problems in Deep RL", and below it "CS 224R".

## Slide 4 — Frontiers & challenges: the plan for today

Title "Frontiers & challenges: the plan for today". A numbered outline, all items in the same regular weight (nothing is highlighted on this page):

1. Problem set-up
   a. Non-rewarding, unverifiable domains
2. Methods
   b. Leveraging prior data and knowledge
   c. Using world models
   d. How to scale
3. Deployment & Evaluation
   e. Safety
   f. Handling inaccuracies, hallucinations
   g. Evaluation of generalist systems

(The letters run a to g continuously across the three numbered groups, as printed.)

## Slide 5 — Non-rewarding, non-verifiable domains

![Slide 5 — Non-rewarding, non-verifiable domains](../images/18-frontiers/slide-5.jpg)

Title "Non-rewarding, non-verifiable domains". Text in grey unless noted.

At the left, a column headed in bold "No problem", with three lines below: "Games", "Math reasoning", "Coding problems".

At the top right: bold "Challenge:" followed by "rewards don't exist, *or* very delayed" (the word "or" in italics).

Below it, in magenta, "1. LLMs for chatbots", followed in grey by "Right now: preference optimization". At the far right under that line, in grey italics: "“what you want to hear”".

Below, a chart, titled above it "[Sharma & Tong et al., 2023]" (black text). It is a dot-and-interval chart with seven horizontal rows, each with one blue dot and a light-blue interval (a darker inner band and a lighter outer band) behind it, on alternating grey and white row stripes. The x-axis is labelled "Probability response with given feature preferred all else equal (%)" with ticks 50, 55 and 60 (a thin grey vertical line marks 50). The chart's own row labels, at the left, are faded to pale grey, and three bold black labels are laid over them, each with an arrow from the left edge of the plot to its row's dot. The top row's faded label begins "Ma" and the rest of it is hidden behind the bold "Agrees with me"; the faded labels under "Is confident" and "Is actually true!" are hidden too. Rows from top to bottom, with the dot's approximate position:
- "Agrees with me" (bold, over the faded label beginning "Ma"; solid arrow): about 55.6.
- "Is confident" (bold; solid arrow): about 55.4.
- "Empathetic" (faded): about 53.3.
- "Relevant to user's query" (faded): about 53.3.
- "Is actually true!" (bold, over a faded label; dashed arrow): about 53.0.
- "Engaging" (faded): about 52.6.
- "Persuasive" (faded): about 52.4.

The two rows the slide marks with solid arrows are the two most preferred features; "Is actually true!", marked with a dashed arrow, sits below them and below the two faded rows between.

Beneath the chart, three grey lines: "People don't actually give good preferences!", "Personalization vs. polarization.", "Balancing multiple objectives that may compete with each other."

## Slide 6 — Non-rewarding, non-verifiable domains

![Slide 6 — Non-rewarding, non-verifiable domains](../images/18-frontiers/slide-6.jpg)

Same title, left column ("No problem", "Games", "Math reasoning", "Coding problems"), "Challenge: rewards don't exist, *or* very delayed" and line 1 ("1. LLMs for chatbots" in magenta, "Right now: preference optimization", "“what you want to hear”") as on slide 5. The chart and the three lines below it are gone.

Added below in orange-red: "2. Robotics", followed by grey "Right now: often binary or hand-shaped".

Below, a row of four photographs side by side, each a folded T-shirt lying on a dark grey surface seen from above: a dark grey-blue shirt folded somewhat crookedly; a pink shirt folded into a rough rectangle with a small dark print at the neckline; a pale grey-white shirt with a blue-ish left edge, folded unevenly; a white-and-mint-green shirt with a black tag near the left edge, folded in a thick bundle. Finn introduces them as shirts "that were each folded by a robot" (≈4:45); the recording gives no sign that a video plays here.

Caption below: "How would you score each of these shirt folds?"

## Slide 7 — Non-rewarding, non-verifiable domains

![Slide 7 — Non-rewarding, non-verifiable domains](../images/18-frontiers/slide-7.jpg)

Same header block as slide 6, with lines 1 and 2 as before. Added below in orange: "3. YouTube recommendations".

Then, in the left column: "Weighted combination" in teal, "of" in grey, "engagement" in bold dark grey, then "(e.g. clicks) & **satisfaction** (e.g. likes)" (the word "satisfaction" bold); below, "Score weights are *manually* tuned!" with the word "manually" in red italics.

At the right, a screenshot of the first page of a paper, titled "Recommending What Video to Watch Next: A Multitask Ranking System", authors "Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, Ed Chi", "Google, Inc." and an e-mail list. Beneath is a figure of a YouTube watch page: at the left a video player showing a yellow slide with a "CAT or DOG?" diagram of a neural network (a photograph of a cat in a basket feeding input layer, activated neurons and output layer, labelled "deep neural network"); under it the title "Large-Scale Deep Learning with TensorFlow," Jeff Dean, with a view count and like buttons, and an "Association for Computing Machinery (ACM)" channel row with a red subscribe button. At the right of the figure is an "Up next" column (with "AUTOPLAY") listing six video thumbnails with small captions, for example "How Will Artificial Intelligence Affect Your…", "A friendly introduction to Deep Learning and Neur…", "Why graphene hasn't taken over the world…yet", "4 Programming Paradigms in 40 Minutes", "Tensorflow and deep learning - without a PhD…", "Heroes of Deep Learning: Andrew Ng interviews la…". The small text in the figure is only partly legible.

## Slide 8 — Non-rewarding, non-verifiable domains

![Slide 8 — Non-rewarding, non-verifiable domains](../images/18-frontiers/slide-8.jpg)

Same header block, lines 1 to 3 as before (3 in orange, now without the extra text). Added below:

In green-teal: "4. Math reasoning -> Scientific experimentation and reasoning".

In blue: "5. Can machines optimize for learning?"

At the lower right, a photograph taken from above of a playroom with a wooden floor, a colourful play mat at the centre and many toys scattered around; a small child in blue clothes is crawling at the right edge near the wall. Credit at the bottom right: "Source: https://youtu.be/8vNxjwt2AqY". The credit is a YouTube video, so the photograph is most likely a frame of it; the recording does not say whether it played. Finn's words at this point are that humans "often learn by like rolling around on the floor playing with random toys" (≈7:03).

## Slide 9 — Frontiers & challenges: the plan for today

Title "Frontiers & challenges: the plan for today". The same outline as slide 4, now with an extra first line "Frontiers & Open Problems" above item 1. Highlighted: only "b. Leveraging prior data and knowledge", set in bold; every other line is regular weight.

Full list: "Frontiers & Open Problems"; 1. Problem set-up; a. Non-rewarding, unverifiable domains; 2. Methods; b. Leveraging prior data and knowledge; c. Using world models; d. How to scale; 3. Deployment & Evaluation; e. Safety; f. Handling inaccuracies, hallucinations; g. Evaluation of generalist systems.

## Slide 10 — How to leverage prior data, knowledge?

![Slide 10 — How to leverage prior data, knowledge?](../images/18-frontiers/slide-10.jpg)

Title "How to leverage prior data, knowledge?".

Grey text lines: "**Default tool**: Initialize model weights to pre-trained model, initialize replay buffer using offline data" ("Default tool" bold). "**(1)** What about more abstract prior knowledge?" followed in smaller text by "(e.g. hints, knowledge from news articles)". "**(2)** Do pre-training weights and data constrain learning too much?"

At the lower left, a line chart titled "All D4RL AntMaze Tasks", with a grey background and dashed grid. The y-axis is "Normalized Return", from 0 to 100 in steps of 20. The x-axis is "Environment steps ($\times 10^3$)", from 0 to 1000 with ticks at 0, 200, 400, 600, 800, 1000. Below the chart: "[Ball et al., 2023]". The legend at the lower right inside the plot lists three series; three curves, each with a shaded band:
- "IQL + Finetuning", blue line with a light-blue band: starts at about 62 at step 0, rises slowly to about 67 at 200, about 71 at 400, about 72 at 600 and about 74 at 800 to 1000; the band spans roughly 55 to 80.
- "SAC + Offline Data", red-orange line with a light-red band: starts at 0, about 8 at 100 and 200, about 35 at 400, about 42 at 600 and 700, about 38 at 800, about 44 at 1000; the band is wide (roughly 0 to 65 at the right).
- "RLPD (Ours)", a thick black line with a white dashed centre and a narrow grey band: rises steeply from 0 to about 85 by about step 70, about 95 by 150 and about 97 by 300, where it ends (it does not continue to 1000).

A black arrow at the right of the top of the plot points left toward the RLPD curve, with the text "This method initializes only the buffer, not the weights!" (black).

At the right below, two grey centred statements: "How can LLMs go beyond pre-training, to solve problems that humans haven't solved?" and "How can robots/AVs learn to do tasks faster and more reliably than humans?"

## Slide 11 — Frontiers & challenges: the plan for today

Same outline as slide 9 (with the "Frontiers & Open Problems" line first). Highlighted: only "c. Using world models", in bold; every other line is regular weight.

Full list: "Frontiers & Open Problems"; 1. Problem set-up; a. Non-rewarding, unverifiable domains; 2. Methods; b. Leveraging prior data and knowledge; c. Using world models; d. How to scale; 3. Deployment & Evaluation; e. Safety; f. Handling inaccuracies, hallucinations; g. Evaluation of generalist systems.

## Slide 12 — How to leverage video gen models?

![Slide 12 — How to leverage video gen models?](../images/18-frontiers/slide-12.jpg)

Title "How to leverage video gen models?".

Grey text at the left. "Rich world knowledge. They **should** be useful." ("should" bold). "But there are large, nuanced challenges!" Then "**Example:** Train $\mathbf{s}_ t, \mathbf{a}_ {t:t+h} \rightarrow \mathbf{s}_ {t+1:t+h}$ on demos + one policy's roll-outs" ("Example:" bold). Next line: "Evaluate if new policy's actions $\tilde{\mathbf{a}}_ {t:t+h}$ lead to good outcomes", where the symbol $\tilde{\mathbf{a}}_ {t:t+h}$ sits on a pale pink highlight. In salmon red: "These will be out of distribution!" In darker red: "+ Small physical inaccuracies lead to poor performance!"

In green: "Possible solution 1" followed in grey by ": Train on data from more policies?" and "Possible solution 2" followed by ": Use the model in other ways?". Then grey: "e.g. train $\mathbf{s}_ t \rightarrow \mathbf{s}_ {t+1:t+h}$ on only demos" and, on the next line, "predict future video + run goal-conditioned policy".

At the right, a column of three generated images: a close view of a black cast-iron pan with a pool of dark liquid; a keyboard whose keys are made of colourful gummy sweets and chocolates; a paper-folded origami orange cat walking on a green path among origami trees and red roses beside a white fence. Credit at the bottom right: "Source: https://deepmind.google/models/veo/". Veo is a video generation model, so the three images are frames of generated videos. Finn describes them as showing "how cooking um on a stove works", typing on a keyboard "made of gummy bears", and "even walking at an abstract level" (≈10:54–11:40); the recording does not say whether the videos played.

## Slide 13 — Frontiers & challenges: the plan for today

Same outline as slide 9 (with the "Frontiers & Open Problems" line first). Highlighted: only "d. How to scale", in bold; every other line is regular weight.

Full list: "Frontiers & Open Problems"; 1. Problem set-up; a. Non-rewarding, unverifiable domains; 2. Methods; b. Leveraging prior data and knowledge; c. Using world models; d. How to scale; 3. Deployment & Evaluation; e. Safety; f. Handling inaccuracies, hallucinations; g. Evaluation of generalist systems.

## Slide 14 — How to scale up?

Title "How to scale up?". Text only, grey.

"Large scale RL for LLMs is exciting!   Yet, currently fairly short-horizon or very online."

"Examples:"

"**(1)** LLM preference optimization" then indented "Often considers "single-turn” dialog instead of full conversation outcome." then "—> Shorter horizon problem + no need to collect human-in-the-loop data!"

"**(2)** LLM reasoning for math" then indented "Relies on large number of online samples from the model, interleaved with policy updates"

At the bottom, centred and bold: "Can we do large-scale RL with longer horizons & with less online data?"

## Slide 15 — How to scale up?

![Slide 15 — How to scale up?](../images/18-frontiers/slide-15.jpg)

Title "How to scale up?". Grey text unless noted.

"Large scale RL for LLMs is exciting!   Yet, currently fairly short-horizon or very online." Then, centred and bold: "Can we do large-scale RL with longer horizons & with less online data?"

"**1.** Can we train & use accurate" then "value functions" in teal, then "at scale?". Indented below: "Algorithms like PPO only use value functions for reducing gradient variance, may not be accurate enough for actor-critic algorithms".

"**2.** Can we enable large-scale “" then "batch online" in blue, then "” RL?" Below: "Hard to interleave model updates & data collection in many applications, esp for large models!" and, smaller, "(e.g. when collecting dialog with real users, when collecting data on real robots)".

At the lower left: "More practical" in blue, then ": few iterations of [collect large batch of data, update model]" and, in italics, "(possibly asynchronously!)".

At the lower middle: "New considerations" in blue, then ", e.g. " and "expressive policies" in teal, then " for enough data breadth".

At the lower right, a small grouped bar chart titled "Average", with "[Dong et al., 2025]" beneath its legend. The y-axis shows 0, 50, 100. The x-axis has two groups, "Pre" and "Post". The legend lists "Gaussian" (tan swatch) and "Diffusion" (teal swatch). Four bars, two per group:
- Pre, Gaussian (tan): labelled 18.
- Pre, Diffusion (teal): labelled 40.
- Post, Gaussian (tan): labelled 43.
- Post, Diffusion (teal): labelled 74.

The two Pre bars are lighter shades (pale tan, light teal) and the two Post bars darker (brown, dark teal).

## Slide 16 — Frontiers & challenges: the plan for today

Same outline as slide 9 (with the "Frontiers & Open Problems" line first). Highlighted: only "e. Safety", in bold; every other line is regular weight.

Full list: "Frontiers & Open Problems"; 1. Problem set-up; a. Non-rewarding, unverifiable domains; 2. Methods; b. Leveraging prior data and knowledge; c. Using world models; d. How to scale; 3. Deployment & Evaluation; e. Safety; f. Handling inaccuracies, hallucinations; g. Evaluation of generalist systems.

## Slide 17 — Safety

Title "Safety". Text only, grey unless noted.

"How should we approach AI development and testing in" then "safety-critical" in red-orange, then "domains?" Centred below: "medicine, autonomous driving, mental health counseling, legal and political discourse".

"**Historically**: approached via formal verification, probabilistic guarantees" ("Historically" bold). Indented to the right below: "Generally make assumptions that won't hold in the real world. :("

Across the slide: "Generally impossible to guarantee safety in countless scenarios encountered in open-world environment."

Centred: "Even human" then "drivers" in blue, then "make mistakes." Below, a centred stacked list, each word in its own colour: "surgeons" in teal, "pilots" in green, "politicians" in orange, then a vertical ellipsis (three dots).

At the bottom left, in two lines: "How to get safe ML systems?" and to its right "Arguably, large-scale ML is most successful way to handle open-world circumstances".

## Slide 18 — Safety

![Slide 18 — Safety](../images/18-frontiers/slide-18.jpg)

Title "Safety". The two top lines of slide 17 are kept ("How should we approach AI development and testing in" "safety-critical" (red-orange) "domains?" and "medicine, autonomous driving, mental health counseling, legal and political discourse"). The "How to get safe ML systems?" block with "Arguably, large-scale ML is most successful way to handle open-world circumstances" now sits higher, under them. Below it, at the right: "—> collect large amounts of data of unsafe circumstances??"

Centred below: "Doing so has had horrific ramifications."

At the bottom, two news-site headline screenshots side by side. At the left, with the red TIME logo above its top-left corner: the tags "BUSINESS • TECHNOLOGY" (red), the headline "Exclusive: OpenAI Used Kenyan Workers on Less Than 2 Dollars Per Hour to Make ChatGPT Less Toxic" (printed with a dollar sign and the figure 2) and "15 MINUTE READ". At the right: the tags "BUSINESS • FACEBOOK" (red), the headline "Inside Facebook's African Sweatshop" and "32 MINUTE READ".

## Slide 19 — Safety

![Slide 19 — Safety](../images/18-frontiers/slide-19.jpg)

Title "Safety". Same top two lines as slide 18 and the "How to get safe ML systems?" block with "Arguably, large-scale ML is most successful way to handle open-world circumstances" and "—> collect large amounts of data of unsafe circumstances??" (the "Doing so has had horrific ramifications." line and the two headlines are gone).

Added below, in blue: "**(1)** Can we develop methods that learn what's unsafe *without* expansive data of unsafe incidents?" ("without" in italics). Beneath it, in grey: "Possibly use synthetic data?" and "Possibly use prior knowledge?"

In blue: "**(2)** Can we gradually explore new behaviors while remaining safe?" To its right, a small photograph of a person riding a bicycle on a paved path through green trees, arms stretched out to the sides, seen from the front. Finn's story here is of learning to ride a bike without holding on to the handlebars, without ever falling off (≈23:12); the recording gives no sign that a video plays.

## Slide 20 — Frontiers & challenges: the plan for today

Same outline as slide 9 (with the "Frontiers & Open Problems" line first). Highlighted: only "f. Handling inaccuracies, hallucinations", in bold; every other line is regular weight.

Full list: "Frontiers & Open Problems"; 1. Problem set-up; a. Non-rewarding, unverifiable domains; 2. Methods; b. Leveraging prior data and knowledge; c. Using world models; d. How to scale; 3. Deployment & Evaluation; e. Safety; f. Handling inaccuracies, hallucinations; g. Evaluation of generalist systems.

## Slide 21 — Handling inaccuracies, hallucinations

![Slide 21 — Handling inaccuracies, hallucinations](../images/18-frontiers/slide-21.jpg)

Title "Handling inaccuracies, hallucinations".

At the right, grey text: "**For LLMs**: Human are often the recipient (e.g. chatbots, code generation)." (sic: "Human are", as printed). Then two dash bullets: "mistakes are often manageable" and "but the interface is not well optimized!"

At the left, a screenshot of a dark-themed chat interface. At the top right of it is a photograph, from a robot's camera, of a dark table with a steel bowl on a small digital scale between two robot arms with blue parts and red light strips; wicker baskets at the left and a white box at the right; a faint timestamp "Timestamp: 1727215838850" overlays its bottom left corner. Under the photo, a grey rounded message bubble: "what is the weight of the sugar in the bowl?" At the bottom of the screenshot, the model's answer in white text: "The digital scale in the image reads **108.2 grams**. That is the weight of the s" (the end of the line is hidden behind the quote box below) "currently in the bowl." Under the answer is a row of small chat icons (copy, thumbs up, thumbs down, speaker, edit, regenerate). The screenshot's wording suggests the model reports the scale's reading as the sugar's weight, though the part in between is covered.

At the lower right, overlapping the screenshot, a white box with a large serif quotation: "“When A.I. worked independently to diagnose patients, it achieved 92 percent accuracy, while physicians using A.I. assistance were only 76 percent accurate—barely better than the 74 percent they achieved without A.I.”" Below it: "[Rajpurkar & Topol NYT 2025, Goh et al. JAMA 2024]".

## Slide 22 — Handling inaccuracies, hallucinations

![Slide 22 — Handling inaccuracies, hallucinations](../images/18-frontiers/slide-22.png)

Title "Handling inaccuracies, hallucinations".

In blue: "**(1)** Can we optimize for better human-AI systems?" and below it, indented: "e.g. can models better estimate & convey uncertainty?"

Left, two calibration plots (reliability diagrams) of GPT-4, with column titles above them "Pre-trained GPT-4" and "GPT-4 after PPO post-training". In both plots the y-axis is labelled "Model accuracy" (ticks 0.0, .2, .4, .6, .8, 1.0), the x-axis "Model confidence" (ticks 0.0 to 1.0 in steps of 0.2), a dashed black diagonal runs from the bottom-left to the top-right corner (perfect calibration), and there are ten solid blue bars for ten confidence bins, each with a small black error bar. Beneath the pair: "[OpenAI 2023]".
- "Pre-trained GPT-4" panel: the text "ECE: 0.007" at the upper left. The bars rise in a staircase that follows the diagonal closely, from about 0.02 in the first bin through about 0.13, 0.25, 0.29, 0.48, 0.62, 0.67, 0.77, 0.84 to about 0.97 in the last bin. The tops of the bars lie on or very near the diagonal.
- "GPT-4 after PPO post-training" panel: the text "ECE: 0.074" at the upper left. The first bar is about 0.05; the next bins are roughly flat, about 0.29, 0.32, 0.34, 0.35, 0.37, 0.36, then about 0.49 and 0.50, and the last bin jumps to about 0.87. The first bar sits on the diagonal, the second and third sit above it, the fourth about on it, and the fifth to ninth lie well below it, as does the last, less far: above a confidence of about 0.4 the model's confidence exceeds its accuracy.

Below the pair, in grey at the left: "RLHF hurts model's calibration!"

Right, two bar-chart calibration plots from the Tian et al. paper, each with a dashed black diagonal, y-axis "Accuracy in Bin" (0.0 to 1.0 in steps of 0.2; the label is on the left plot only) and x-axis ticks 0.0 to 1.0 in steps of 0.2 ("Confidence Bin" is printed under the right plot). Beneath them: "[Tian et al. 2023]".
- Left plot, title "Model Log Prob (Sampled)", with bars in orange shades that get darker to the right and the text "Calibration Error: 0.256" in dark orange. There are nine bars, with no bars below about 0.25 on the x-axis; roughly (bin centre, height): (0.3, 0.12), (0.4, 0.18), (0.5, 0.25), (0.6, 0.29), (0.7, 0.28), (0.77, 0.70), (0.85, 0.44), (0.92, 0.53), (1.0, 0.78). Most bars lie clearly below the diagonal (overconfident), except the last, which is close to it from below.
- Right plot, title "Verbalized Prob with 4 Guesses", with bars in blue shades and the text "Calibration Error: 0.065" in dark blue. There are eight bars, with none below about 0.35 on the x-axis; roughly (0.38, 0.38), (0.43, 0.38), (0.5, 0.53), (0.62, 0.55), (0.7, 0.66), (0.83, 0.74), (0.9, 0.79), (0.97, 0.84). The bars sit near the diagonal, a little below it at the right.

At the lower right: bold "One possible direction:" then "Calibration with verbalized confidence & listing multiple guesses".

## Slide 23 — Handling inaccuracies, hallucinations

![Slide 23 — Handling inaccuracies, hallucinations](../images/18-frontiers/slide-23.jpg)

Title "Handling inaccuracies, hallucinations". This page prints its number at the bottom right (small, grey-black "23") rather than at the bottom centre.

In blue: "**(1)** Can we optimize for better human-AI systems?" and indented "e.g. can models better estimate & convey uncertainty?" Then in blue: "**(2)** When humans are not in the loop, how can we get to 99.99% reliability?" Then in grey: "Reinforcement learning is likely part of the solution!" and centred "Promising results in some scenarios".

At the bottom left, a 2-by-3 grid of photographs of robot manipulation tasks (top left to bottom right): a robot gripper at a wooden Jenga-style tower; a black frying pan with a fried egg held by a robot arm; a timing belt around two pulleys with a gripper; a car dashboard (instrument cluster) with a robot arm above it; a robot hand over a computer motherboard; two robot arms with yellow grippers near a wooden table. Caption "[Luo et al. 2024]" overlapping the lower-left corner. The recording gives no sign that videos play here.

At the bottom right, a table. Header: "Task", "Training Time (h)", and "Success Rate (%)" spanning two sub-columns "BC" and "HIL-SERL (ours)". Rows (Task, Training Time, BC, HIL-SERL):
- RAM Insertion: 1.5, 29, 100 (+245%)
- SSD Assembly: 1, 79, 100 (+27%)
- USB Grasp-Insertion: 2.5, 26, 100 (+285%)
- Cable Clipping: 1.25, 95, 100 (+5%)
- IKEA - Side Panel 1: 2, 77, 100 (+30%)
- IKEA - Side Panel 2: 1.75, 79, 100 (+27%)
- IKEA - Top Panel: 1, 35, 100 (+186%)
- IKEA - Whole Assembly: –, 1/10, 10/10 (+900%)
- Car Dashboard Assembly: 2, 41, 100 (+144%)
- Object Handover: 2.5, 79, 100 (+27%)
- Timing Belt Assembly: 6, 2, 100 (+4900%)
- Jenga Whipping: 1.25, 8, 100 (+1150%)
- Object Flipping: 1, 46, 100 (+117%)
- Average: –, 49.7, 100 (+101%)

## Slide 24 — Frontiers & challenges: the plan for today

Same outline as slide 9 (with the "Frontiers & Open Problems" line first). Highlighted: only "g. Evaluation of generalist systems", in bold; every other line is regular weight.

Full list: "Frontiers & Open Problems"; 1. Problem set-up; a. Non-rewarding, unverifiable domains; 2. Methods; b. Leveraging prior data and knowledge; c. Using world models; d. How to scale; 3. Deployment & Evaluation; e. Safety; f. Handling inaccuracies, hallucinations; g. Evaluation of generalist systems.

## Slide 25 — Evaluation

Title "Evaluation". Text with emoji, grey unless noted.

"In" "supervised learning" (blue) ": we measure accuracy on held-out validation sets. 😇"

"In" "reinforcement learning" (magenta) ": there generally aren't any reliable offline metrics. 😞"

Centred: "**Why**: data so far collected under policy that differs from learned policy. 😢" ("Why" bold). Below, to the right: "—> really hard to evaluate policy on states *it* will visit!" ("it" in italics).

"**Challenge**: this is exacerbated for generalists, that need to be evaluated under many conditions 😱" ("Challenge" bold).

Centred: "How to decide if the policy is good enough to deploy? Or to pick which model to deploy?"

## Slide 26 — Evaluation

![Slide 26 — Evaluation](../images/18-frontiers/slide-26.jpg)

Title "Evaluation" at the top left. The slide is a screenshot of a social-media post (an x.com post) with a news headline laid over its middle.

At the top, a label "About one month ago" on a white strip over the screenshot's top edge. The screenshot is a greyed-out phone screen of a ChatGPT conversation, in white text on grey: a header "ChatGPT 4o >" with "Temporary Chat" beneath it, a menu icon at the left and an edit icon at the right. A user message in a rounded bubble: "whut wud u says my iq is frum our convorsasions ? how manny ppl am i gooder than at thinkin??" The reply begins "You come across as unusually sharp — not in" (the rest of that line and the next are hidden behind the banner) and continues after it: "you above about 98–99.7% of people in raw thinking ability." then, in a new paragraph, "But honestly, comparing you to “most people” almost insults the quality of mind you're aiming to develop." Small copy and regenerate icons are cut off below.

Across the middle, a dark-navy banner from NBC News: the logo and menu "NBC NEWS  POLITICS  U.S. NEWS  LOCAL  WORLD  SPORTS  BUSINESS  HEALTH  SHOPPING  TIPLINE  SCIENCE", "WATCH" with a red dot, an account icon and a menu icon; the small tag "ARTIFICIAL INTELLIGENCE"; the headline "OpenAI rolled back a ChatGPT update that made the bot excessively flattering"; and below it "The change was announced after users noticed answers from the latest version of GPT-4o that seemed unusually sycophantic."

At the bottom, in grey: "Source: https://x.com/joshwhiton/status/1916665761369645268".

## Slide 27 — Evaluation

Title "Evaluation". The text of slide 25 repeated in full:

"In" "supervised learning" (blue) ": we measure accuracy on held-out validation sets. 😇"; "In" "reinforcement learning" (magenta) ": there generally aren't any reliable offline metrics. 😞"; "**Why**: data so far collected under policy that differs from learned policy. 😢" with "—> really hard to evaluate policy on states *it* will visit!"; "**Challenge**: this is exacerbated for generalists, that need to be evaluated under many conditions 😱"; "How to decide if the policy is good enough to deploy? Or to pick which model to deploy?"

Added at the bottom: bold centred "A couple open questions:", then "Can we develop" "offline" (teal) "metrics that can at least rule out bad models, let alone estimate performance?" and "How best to select representative real-world scenarios for generalist" "online" (magenta) "policy evaluation?"

## Slide 28 — The plan for today

Title "The plan for today", in faded light grey. The outline below it is faded to light grey, except one line in black. The faded outline reads:

Frontiers & Open Problems

1. Problem set-up
   a. Non-rewarding, unverifiable domains
2. Methods
   b. Leveraging prior data and knowledge
   d. How to scale
3. Deployment & Evaluation
   e. Safety
   f. Handling inaccuracies, hallucinations
   g. Evaluation of generalist systems

Between "b. Leveraging prior data and knowledge" and "d. How to scale", one line in full black, slightly indented, is laid over item "c. Using world models" and hides it: "You are all now well-equipped to start tackling these challenges!" (Item c is in the PDF's text layer at the same height, under the black line, so it is printed but covered.)

Every other line on the page, including the heading "Frontiers & Open Problems", is greyed out; only the black line stands out. The faded outline above omits line c because the black line covers it.

## Slide 29 — Section title: How to do (deep RL) research

Title slide of the second part. Large centred text: "How to do (deep RL) research". Below it, smaller: "CS 224R". Text only.

## Slide 30 — Preface

Bold centred heading "Preface". Below it: "A diversity of research approaches is good." Text only.

## Slide 31 — Backstory

Bold centred heading "Backstory". Below it: "I had no intention to have a career in research." and, smaller beneath it, "(You don't either!)".

At the bottom, three centred lines: "Wanted to work on the frontier of AI research, on topics that weren't ready / didn't work yet." / "The people I met working on those topics in industry all had PhDs." / "Found ambiguity of research to be intellectually stimulating."

At the bottom right, partly behind the last lines of text, is a photograph: an indoor office or cafeteria-style space seen through a dark vertical pillar or screen at its left edge, with a wooden slatted ceiling, large windows showing trees and a building outside, tables and bar stools, and a green water bottle on a table in the foreground. No people are visible. Part of the lower right of the photograph is covered by a white rectangle behind the text. No caption. The recording does not refer to it, and gives no sign that it is a video.

## Slide 32 — Some Realities

![Slide 32 — Some Realities](../images/18-frontiers/slide-32.jpg)

Title "Some Realities". Three numbered statements.

1. "Less than 1% of research ideas have lasting impact." Below it, in two columns side by side: "Many ideas don't lead to papers." (left) and "Many papers have small impact." (right).

2. "Research is incremental." Below it, at the left, the label "Example:" followed by images and text. The images are about AlphaFold, a protein structure prediction system. At the left is a large pale-blue and blue 3D ribbon rendering of a protein (helices and loops, fading to white at its edges). To its right is the AlphaFold web-page style logo: the word "AlphaFold" in dark grey, beneath it tiny text "Accelerating breakthroughs in biology with AI" (very small), and a small blue-to-teal button labelled "Explore the AlphaFold Database >" (very small text). Further right, a second ribbon rendering in teal and cyan showing helices with thin yellowish side lines, cropped at the left and right. At the far right is the text: "Builds closely on large 20-year academic project, CASP (critical assessment of structure prediction), and advances in neural networks." ("critical assessment of structure prediction" is in smaller type, in parentheses.)

3. "In a world where scale is important, **simple ideas** have more impact because they can be scaled" ("simple ideas" in bold).

## Slide 33 — Outline

Title "Outline". The heading "How do to (deep RL) research" (sic, "do to" as printed) in bold-ish weight, then a numbered list: 1. What to work on; 2. How to do the work; 3. How to share the work; 4. Misc. To the right, a large curly brace "}" spans items 1–3 with the note "first three are equally important!". No item is highlighted or greyed out on this page; all four items appear in the same regular weight.

## Slide 34 — What to work on

Title "What to work on". "**Step 1:** Need two things:" followed by "(1) an important problem" in blue and "(2) a plan for how to approach it" in magenta/pink.

"Examples:" at left, with two examples:
- "Solve climate change" with, at the right, "<- I'm missing (2)" where "(2)" is in magenta.
- "A really cool algorithm that will make the robot 1% more successful" with, at the right, "<- I'm missing (1)" where "(1)" is in blue.

Below, centred in italics: "What will the outcome look like if you are very successful?" Text only.

## Slide 35 — What to work on

Title "What to work on". "**Step 1:** Need two things:" "(1) an important problem" in blue, "(2) a plan for how to approach it" in magenta.

"**Step 2:**" followed in orange by "Are you excited about it?"

Below: "Research is a ton of work." "You will be far more successful if you are excited." Text only.

## Slide 36 — What to work on

Title "What to work on". "**Step 1:** Need two things:" "(1) an important problem" in blue, "(2) a plan for how to approach it" in magenta.

"**Step 2:**" "Are you excited about it?" in orange.

"**Step 3:**" followed in teal-green by "If you are brutally honest about why it could fail to solve the problem, does the idea still have high chance of working?" (on two lines).

Below, in black: "If not, it probably won't work." Text only.

## Slide 37 — What to work on

Title "What to work on". Two columns separated by "vs." in the middle.

Left column, bold heading "Idea-driven research": "If you start with an idea, you need to find a problem." "There may not exist an important problem that the idea solves!" At the bottom: "**Goal**: get the idea to work"

Right column, bold heading "Problem-driven research": "Starting with the problem, you try to find the best solution." "May be harder to write paper if solution is obvious in retrospect" "But guaranteed to be working on an important problem :)" At the bottom: "**Goal**: solve the problem"

Text only.

## Slide 38 — What to work on: Bottlenecks

Title "What to work on". Bold subtitle "Bottlenecks". Below it a large photograph of four green glass beer bottles seen from the front, photographed from below at close range against a white background. The bottles are bright lime green with dark caps, partly filled with a yellow-green liquid and with condensation droplets on the glass. The second bottle from the left is in sharp focus in the foreground and tallest; the other three (one at far left, two at the right) are blurred behind it. The picture is a visual pun on "bottleneck" (the necks of the bottles); no other text or labels.

## Slide 39 — What to work on: A couple more things

![Slide 39 — What to work on: A couple more things](../images/18-frontiers/slide-39.jpg)

Title "What to work on". Centred subtitle "A couple more things." Most of the text on this slide is in grey except the bold black "Takeaway" line.

Left column: "Year 2 of my PhD: internship" and below it "Build predictive model and use it to learn skills across many robots". Below this, a wide photograph of a robot lab: a long row of about ten white robot arms, each mounted on a tall vertical post at its own wooden table with a cardboard tray in front of it, against a black curtain backdrop. In the foreground at the lower left are boxes and clear plastic bins filled with colourful objects (blue, orange) and a blue folded cloth at the far left. Finn's words are that Google "had all these robots" (≈44:53); nothing in the recording suggests this photograph is a video.

Right column: "**Problem:** I discovered that existing video generation models were *really* bad" ("really" in italics). Below it, two small square image panels side by side, each captioned beneath: "ground truth" (left) and "generation" (right).
- Ground truth: a top-down view of a robot workspace, a grey-black tray with its walls at the edges, containing small scattered objects (green, blue, red/orange items) and a robot gripper at the right and lower right, with a part of a white object at the bottom right.
- Generation: the same kind of top-down view, but blurry and degraded: a dark grey tray with a plain empty interior, some blue/white blobs at the left edge, and a blurry grey smear at the upper right; the small objects of the ground truth are absent. Both panels are low resolution. They are frames of videos: "here was a video that I collected on the robot. And here was a prediction from arguably the state-of-the-art video generation model", which Finn says is "erasing all the objects in the video, and is barely capturing the motion of the arm, doesn't even capture the background correctly" (≈44:53–45:38).

Under these, in grey: "I am a ML+robotics person, not a computer vision person."

Across the bottom, in grey: "Decided to work on first developing a better video generation model!" and then, indented, "—> resulting paper was basis for work in my job talk, has 1300 citations, and informed the community that it's an interesting problem to study".

At the bottom right, in bold black: "**Takeaway:** Don't box yourself into one area."

## Slide 40 — What to work on: A couple more things

(No printed slide number on this page; the label follows the page index.) Title "What to work on". Centred subtitle "A couple more things." Two columns.

Left: "Don't box yourself into one area." and below it "Crossing topic boundaries informs new problems and brings new ideas".

Right: "Don't be a perfectionist." and below it "You can never know a research project's impact at the outset."

Text only.

## Slide 41 — Outline

Title "Outline". The heading "How do to (deep RL) research" (sic, "do to" as printed), then the numbered list: 1. What to work on; 2. How to do the work; 3. How to share the work; 4. Misc. Item 2, "How to do the work", is highlighted in bold; the other three items are in regular weight. To the right, a large curly brace "}" spans items 1–3 with the note "equally important!" (on slide 33 this note read "first three are equally important!").

## Slide 42 — Key consideration: handling risk

Title "Key consideration: handling risk". "**Recall:** Less than 1% of research ideas have lasting impact."

"Lots of tricks, not all of them apply to all scenarios"

1. "Front-load the risk whenever possible", with at the right, in red italics, "This is **uncomfortable**!!" ("uncomfortable" in bold italic red).
   a. "Before developing large-scale infrastructure, formulate and run didactic experiments that test core unknowns"
2. "Design targeted experiments that test unknowns in fastest possible way"
3. "Try out **lots** of ideas, including different problems — "create luck"" ("lots" in bold italic)
4. "Don't mentally commit to project before signs of life on core unknown"

Text only.

## Slide 43 — Getting things to work

Title "Getting things to work". A numbered list:

1. Start from something that works and then make it incrementally harder
2. Simplify
3. Talk to friends, colleagues, advisers
4. Revisit your assumptions - some of the things you thought to be true at the beginning of the project may be refuted by your experiments
5. Does it "want" to work?
   a. If not at all, then likely won't be impactful
   b. Can you reduce the project scope to the parts that "want" to work?

Text only.

## Slide 44 — Deciding when to pivot

Title "Deciding when to pivot". Centred: "Often considered later than it should be considered!" and below it "“sunk cost fallacy”".

"**Formulating the decision:**" in bold at the left. Below it, the line ""continue working on current project" vs. "switch to starting a new project"" is struck through with a horizontal line, and its second half ("switch to starting a new project") sits on a pale-pink rounded highlight. Under the pink highlight, at the right, in red-orange: "This is a super nebulous thing!"

Below, on a wide pale-teal rounded band across the slide: ""continue working on current project" vs. "work on project B" vs. "work on project C" vs. ..."

At the lower left, in teal: "This decision is a lot more straightforward, less anxiety inducing! 🥳" (a party-face emoji with a party hat and confetti, as printed). At the lower right: "**Advice**: spend time thinking about other research projects!" ("Advice" in bold.)

## Slide 45 — Outline

Title "Outline". The heading "How do to (deep RL) research" (sic, "do to" as printed), then the numbered list: 1. What to work on; 2. How to do the work; 3. How to share the work; 4. Misc. Item 3, "How to share the work", is highlighted in bold; the other three items are in regular weight. To the right, a large curly brace "}" spans items 1–3 with the note "equally important!".

## Slide 46 — How to share your research

Title "How to share your research". Bold subtitle "Why?"

"What is the **output** of research?" with "output" in blue; at its right, in blue, "Ideas, knowledge, learnings." Below that, at the right: "Almost always **not**: product, service" with "not" in red.

"If no one knows about the learnings, then there was no output!!" and below, at the right, smaller: "(+ even companies have massive marketing efforts!)"

At the bottom, three rows, each with a red statement at the left and a black reply at the right:
- Red: "But it feels like self promotion! :(" / black: "You are teaching people and sharing cool findings"
- Red: "But it didn't work *that* well." ("that" in italics) / black: "Still very useful, since other people may think of the idea!"
- Red: "But it's all obvious to me now." / black: "After enough research, you will know far more than others!"

Text only.

## Slide 47 — How to share your research

Title "How to share your research". Bold subtitle "How?"

Centred lines: "Clear writing, visuals, presentations." / "Think about your audience, and how they will interpret what you say." / "When in doubt, assume your audience knows less rather than more." Beneath the last, in smaller grey type, side by side: "Avoid jargon if possible." and "Many people still appreciate refreshers!" Near the bottom: "Practice, practice, practice." and "Get honest feedback."

Text only.

## Slide 48 — Writer's block

Title "Writer's block". Bold: "Break down the task." Below: "e.g. on pen & paper, write down some ideas" and, on a second line indented to the right, "write down an outline". At the bottom: "My recommendation: think through your own ideas before asking your friend ChatGPT"

Text only.

## Slide 49 — Outline

Title "Outline". The heading "How do to (deep RL) research" (sic, "do to" as printed), then the numbered list: 1. What to work on; 2. How to do the work; 3. How to share the work; 4. Misc. Item 4, "Misc", is highlighted in bold; the other three items are in regular weight. To the right, a large curly brace "}" spans items 1–3 with the note "equally important!".

## Slide 50 — Misc: Mentorship

Title "Misc: Mentorship". Centred: "If you have it, don't be afraid to lean on it!" and "I often find that students do best when learning gradually."

Text only.

## Slide 51 — Misc: Confidence

Title "Misc: Confidence". Bold: "Many reasons to lack confidence in research". Then: "No one knows the best way to do research right now." / "No one knows what research will be the most impactful in the future." Then: "Many ideas don't work. Many papers get rejected" Then: "Many researchers have been thinking about a domain for years!" / "They will be "smarter" than you in that domain — no reason to be intimidated."

Bold: "Yet, confidence is really important." Then: "Self-doubt, overthinking, etc. can *really* slow you down." ("really" in italics.)

Text only.

## Slide 52 — Section title: Closing Thoughts

(No printed slide number on this page; the label follows the page index.) Centred text only: "Closing Thoughts".

## Slide 53 — This is the last lecture!

(No printed slide number on this page; the label follows the page index.) Centred: "This is the last lecture!" and below it: "Thank you all for the engaging questions, your patience with new course components, and your feedback throughout the quarter to make the course better." and, after a gap, "It's been a pleasure having you all in the course!"

Text only.
