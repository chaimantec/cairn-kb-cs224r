# See also — related knowledge bases

Other Cairn knowledge bases whose material genuinely bears on CS224R. Each entry gives the repo URL to
pass as `kb` to `kb_read` / `kb_list`, and says what that KB is good for and what it does not cover, so
it is worth deciding before spending a turn on it. This file is the only place the chat learns that
these siblings exist.

The extra links point at `raw.githubusercontent.com` rather than `github.com/.../blob/...`, because a
raw URL returns plain markdown and so also works as a `web_fetch` target.

- **CS224N — Natural Language Processing with Deep Learning** (Stanford, Christopher Manning, Spring
  2024).
  KB: `https://github.com/chaimantec/cairn-kb-cs224n` — pass this as `kb` to `kb_read` / `kb_list`.
  CS224R's Lecture 9 deck says it is "Based on slides in CS224N", and CS224N's Lecture 11
  (Post-training) is an earlier version of the same deck by the same lecturer, Archit Sharma. It
  holds the parts CS224R's deck leaves out: prompting, zero-shot and few-shot in-context learning, and
  chain-of-thought, before the instruction finetuning, RLHF and DPO material the two share. Reach for it
  when a question about Lecture 9 touches what comes before instruction finetuning, or for a second
  telling of the same derivations. **Its slide numbers are its own deck's**: the CS224R deck was revised
  and renumbered, so a "slide N" there is not slide N here. It covers all 23 of its lectures, and
  says nothing about RL outside language models.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/INDEX.md) ·
  [Lecture 11](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/11-post-training.md) ·
  [DPO](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs224n/main/wiki/direct-preference-optimization.md)

- **CS336 — Language Modeling from Scratch** (Stanford, Percy Liang and Tatsunori Hashimoto, Spring
  2026).
  KB: `https://github.com/chaimantec/cairn-kb-cs336` — pass this as `kb` to `kb_read` / `kb_list`.
  Post-training as it is built: its Lecture 15 covers supervised fine-tuning and RLHF, and its Lecture
  16 reinforcement learning from verifiable rewards for reasoning models (PPO, GRPO and open recipes),
  the direction Lecture 9 closes on and the subject of Lecture 10. Reach for it when a question is
  about how RLHF or DPO is done in practice, or about GRPO and open reasoning recipes, which Lecture 10
  only names, rather than about this course's derivations. It covers all 18 of
  its lectures; it does not teach RL outside language models.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/INDEX.md) ·
  [Lecture 15](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/wiki/15-mid-post-training.md) ·
  [DPO](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/wiki/dpo.md) ·
  [Lecture 16](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/wiki/16-post-training-rlvr.md)

- **CS329A — Self-Improving AI Agents** (Stanford, Aakanksha Chowdhery and Azalia Mirhoseini, Autumn
  2025).
  KB: `https://github.com/chaimantec/cairn-kb-cs329a` — pass this as `kb` to `kb_read` / `kb_list`.
  A paper-reading seminar on the neighbourhood of Lecture 10: scaling test-time compute (its Lecture
  2), verifiers, including outcome and process reward models and the PRM800K dataset (Lecture 3), and
  train-time scaling with RL, from STaR's filtered self-training to DeepSeekMath and the reasoning
  models (Lecture 6). Reach for it when a question about Lecture 10 turns to test-time scaling,
  verifiers or the papers behind thinking models, which Lecture 10 covers only in its slides or in
  passing. It is a different course's reading of different papers, so it does not explain Lecture
  10's own methods (rejection finetuning's scaling, per-step DPO, process advantage verifiers). It
  covers all 9 of its lectures.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs329a/main/INDEX.md) ·
  [Lecture 3](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs329a/main/wiki/03-robust-verification.md) ·
  [Lecture 6](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs329a/main/wiki/06-train-time-scaling-scaling-rl.md)

- **Princeton ROB 345/549 — Introduction to Robotics** (Princeton, Anirudha Majumdar, Fall 2026).
  KB: `https://github.com/chaimantec/cairn-kb-princeton-rob-345` — pass this as `kb` to `kb_read` /
  `kb_list`. The control side of Lecture 11's planning: trajectory optimization with *known* dynamics,
  solved as a quadratic program or by sequential quadratic programming (its Lecture 6), and receding
  horizon control, its name for model predictive control, presented as replanning that turns an open-loop
  plan into feedback (its Lecture 7 and its feedback-control page). Reach for it when a question about
  Lecture 11's model predictive control or open- and closed-loop execution wants the control-theory
  framing, or how a plan is optimized when the dynamics are given. It does not learn dynamics models, and
  it does not cover random shooting, the cross-entropy method or model-based RL. It covers its 11 recorded
  lectures, of the 24 its site schedules.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-princeton-rob-345/main/INDEX.md) ·
  [Lecture 7](https://raw.githubusercontent.com/chaimantec/cairn-kb-princeton-rob-345/main/wiki/07-stability-and-pd-control.md) ·
  [Feedback control](https://raw.githubusercontent.com/chaimantec/cairn-kb-princeton-rob-345/main/wiki/feedback-control.md) ·
  [Trajectory optimization](https://raw.githubusercontent.com/chaimantec/cairn-kb-princeton-rob-345/main/wiki/trajectory-optimization.md)
