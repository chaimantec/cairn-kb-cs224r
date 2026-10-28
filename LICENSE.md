# Licence and copyright

This repository is a **study aid compiled from Stanford CS224R course material**. It is not a
work with a single owner, so it carries no single repository-wide licence. Three layers of
authorship sit here, with three different copyright positions. Read this before reusing
anything from it.

The short version: **the explanatory writing is yours to reuse with attribution; the course
material and the third-party images in the slides are not mine to give you.**

---

## Layer 1 — the compilation's own editorial work

Copyright (c) 2026 chaimantec.

Licensed under **Creative Commons Attribution 4.0 International** (`CC-BY-4.0`),
<https://creativecommons.org/licenses/by/4.0/>. You may share and adapt it, including
commercially, with attribution.

This covers only the material original to this compilation:

- `INDEX.md`, `AGENTS.md`, `BUILD_LOG.md`, `TODO.md`, `kb.json`, `sources.md`, this file;
- the original explanatory prose in `wiki/`: the explanations, the connective argument, the
  organisation of the material into pages, and the editorial judgements about what is worth
  recording.

It does **not** extend to the lecture content those pages describe, nor to any passage in
`wiki/` that quotes the lecturer or a slide. Those are quotations, marked as such, and belong to
Layer 2 or Layer 3.

### Not offered under CC BY 4.0, even though I wrote the words

`raw/transcripts/` and `raw/slides/` are derivative works. The copy-editing, the `[Ed:]` notes
and the figure descriptions are mine; the lecture being transcribed and the slides being
described are not. I cannot license the underlying expression, so these directories are not
offered under CC BY 4.0. Read them, cite them, and quote them as you would quote the lecture
itself, but do not treat them as freely relicensable text.

The verbatim auto-captions these transcripts were edited from are **deliberately not published
here**. They would be a complete reproduction of each lecture with no editorial contribution.
They are regenerable from the public video with the `cairn-kb` skill's `fetch_transcript.py`,
using the video id in each transcript's front matter, so anyone can re-derive them and audit an
edit.

---

## Layer 2 — Stanford CS224R course material

**No licence exists, and none is granted here.**

The lectures and slide decks are the work of Chelsea Finn, the guest lecturers and the CS224R
course staff (Stanford University, Spring 2025). They are published through the course website,
<https://cs224r.stanford.edu/spring_2025/>, with the lecture videos on YouTube.

As checked on 2026-10-28, no licence, copyright or reuse statement appears on the course
website, and none is printed on any page of the Lecture 1, 2 or 3 decks. Publicly downloadable is not the
same as licensed. Absent an express grant the default applies: copyright arises automatically,
and no reuse rights pass to anyone.

The images in `raw/images/` and the material described in `raw/slides/` are reproduced here for
study and commentary, relying on fair use (17 U.S.C. § 107) rather than on any permission. Only
slides that carry a figure are rendered. Title slides, text slides, section dividers and
photographs of course staff are not. Every deck is linked at its canonical URL beside the
material drawn from it. **That reliance is mine and does not transfer to you.** Fair use is a
defence to infringement, not a licence. It can justify a particular use by a particular user for
a particular purpose, and gives me nothing to pass downstream. If you want to reuse Layer 2
material, make your own assessment or ask the course staff.

**Takedown.** If the instructor or any rights holder asks for an image or a transcription to be
removed, it will be deleted along with every link to it.

---

## Layer 3 — third-party images reproduced inside the decks

The decks reproduce images that are not Stanford's: video frames, figures from research papers,
screenshots and photographs. They belong to their respective owners, Stanford could not license
them to anyone, and nothing here does either. The attributions below are taken from citations,
logos, captions and URLs visible on the slides. Where nothing on the slide identifies the source,
the entry says so.

### Lecture 1 — Class Intro

Deck: [01_cs224r_intro_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf).
Rendered: 16 slides, in `raw/images/01-class-intro/`.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 19 | Video stills of a quadruped robot (captioned "Unitree") and a humanoid. The humanoid's source is not credited. |
| 20 | Video stills captioned "Physical Intelligence π0" and "Google Gemini Robotics". |
| 21 | Screenshot of a broadcast of the AlphaGo–Ke Jie match, with a "Future of Go Summit in Wuzhen" banner and a Google logo. The broadcaster is not credited. |
| 23 | Traffic-simulation screenshots, "Source: https://mit-wu-lab.github.io/automatic_vehicular_control/". |
| 24 | Generated-image strips, "Source: https://rl-diffusion.github.io/". |
| 25 | Chip-placement diagram, "Source: https://research.google/blog/chip-design-with-deep-reinforcement-learning/". |
| 26 | Video still of a PR2 robot at UC Berkeley, cited on the slide as Levine et al., ICRA '15. |
| 27, 28 | Video stills of a PR2 robot, cited on the slides as Levine, Finn et al., JMLR '16. Slide 28 shows a person's hand holding the cube. |
| 30 | Video still of a PR2 robot with the overlay "MDGPS network policy". Not credited. |
| 32, 34 | A photograph of the towel-and-hook robot setup and two chatbot screenshots. Not credited. |
| 35 | A highway photograph, a restaurant-booking web page, a poker hand and clip-art grass. Not credited. |
| 36 | The policy-network diagram, with photographs of the towel-and-hook robot setup. Not credited. |
| 37 | A highway photograph inside the deck's own graphical model. Not credited. |
| 33 | No third-party content identified: the deck's own graphical model. |

Not rendered from this deck: the photographs of course staff (slide 3), a meme (slide 13), clip-art
icons (slide 8) and product logos (slide 22).

### Lecture 2 — Imitation Learning

Deck: [02_cs224r_imitation_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/02_cs224r_imitation_2025.pdf).
Rendered: 15 slides, in `raw/images/02-imitation-learning/`.

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 3 | The policy-network diagram from Lecture 1, with photographs of the towel-and-hook robot setup. Not credited. |
| 5, 7 | A highway photograph (the one used in Lecture 1); slide 7 adds the deck's own arrows and chart. Not credited. |
| 8 | A Super Mario game screenshot and a photograph from a driver's seat. Not credited. |
| 9 | A random-noise image. Not credited. |
| 10 | A photograph of a robot arm pushing a T-shaped block. Not credited. |
| 12 | A simulation render of two robot arms and a photograph of a four-arm robot workspace, beside the deck's bar charts. The slide gives the sources "https://diffusion-policy.cs.columbia.edu/" and "https://aloha-unleashed.github.io/". |
| 13 | Video stills captioned "Physical Intelligence π0", "NVIDIA Gr00t N1" (with a "Fourier GR-1" banner) and "Figure Helix", and robot-view photographs captioned "OpenVLA". |
| 14 | An image captioned "Waymo EMMA" and a video frame captioned "Wayve LINGO-2", carrying a Wayve logo. |
| 20 | A clip-art drawing of a car with a girl and a robot driver. Not credited. |
| 23 | Photographs of kinesthetic teaching with a PR2 robot, of a game controller, a 3D mouse and a VR controller, and of a person puppeteering leader arms. Not credited. |
| 24 | Video frames of a person and of simulated characters, cited on the slide as Peng, Kanazawa, Malik, Abbeel, Levine, "SFV: Reinforcement Learning of Physical Skills from Videos", SIGGRAPH Asia 2018. |
| 18, 19, 21 | No third-party content identified: the deck's own diagrams. |

Not rendered from this deck: slides 1 and 15, whose only images are SpongeBob SquarePants cartoon
stills; the slides print "SpongeBob image rights & credits to Viacom International".

### Lecture 3 — Policy Gradients

Deck: [03_cs224r_policy_gradients_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf).
Rendered: 11 slides, in `raw/images/03-policy-gradients/`. Slides 7–10 print "Slide adapted from
Sergey Levine".

| Slide | Third-party content (as identifiable from the slide) |
| --- | --- |
| 5 | A photograph of a humanoid robot, captioned "Source: Unitree", and a screenshot of a chat interface showing code. The screenshot is not credited. |
| 7, 10, 11 | 3D plots of trajectories over a gridded surface (slide 11's heat-mapped). The slides they appear on are "adapted from Sergey Levine" (7, 10); slide 11 credits nothing. |
| 12, 14 | A still of a simulated humanoid. Not credited. |
| 17 | A photograph of a humanoid robot holding a jacket on a table. Not credited. |
| 6, 19, 24, 26 | No third-party content identified: the deck's own loop diagrams and trajectory sketches. |

Not rendered from this deck: slide 27, which repeats slide 5; slides 13, 15 and 16, whose only
picture is a thumbnail of the slide 12 humanoid.
