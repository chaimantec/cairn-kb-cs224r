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
  the direction Lecture 9 closes on. Reach for it when a question is about how RLHF or DPO is done in
  practice, or about RL for reasoning, rather than about this course's derivations. It covers all 18 of
  its lectures; it does not teach RL outside language models.
  Start at [INDEX](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/INDEX.md) ·
  [Lecture 15](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/wiki/15-mid-post-training.md) ·
  [DPO](https://raw.githubusercontent.com/chaimantec/cairn-kb-cs336/main/wiki/dpo.md)
