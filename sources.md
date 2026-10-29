# Sources — CS224R

Every slide deck, handout and reading that the course website
(<https://cs224r.stanford.edu/spring_2025/>) links to, with its canonical URL, organised by
lecture. The PDFs are deliberately **not** committed to this repo. This knowledge base is read by
an agent that navigates markdown, so the URL is what is useful, and the 19 decks alone total
about 190MB. Cite these URLs directly.

Crawled 2026-10-28. The student final-project archive (`projects/`) and the 2023 offering's
page were excluded. The site has no `robots.txt` and states no licence; see
[`LICENSE.md`](LICENSE.md).

## Lectures

Dates are the Spring 2025 schedule on the site. "Video" is the recording in the Cairn catalog.
"In this KB" says what has been built for that lecture so far.

| # | Date | Lecture (site title) | Video | Deck | Optional reading (as listed on the site) | In this KB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Wed Apr 2 | Course intro + MDPs | [EvHRQhMX7_w](https://www.youtube.com/watch?v=EvHRQhMX7_w) | [01_cs224r_intro_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/01_cs224r_intro_2025.pdf) | none | [wiki](wiki/01-class-intro.md) · [slides](raw/slides/01-class-intro.md) · [transcript](raw/transcripts/01-class-intro.md) |
| 2 | Fri Apr 4 | Imitation Learning | [WxRDyObrm_M](https://www.youtube.com/watch?v=WxRDyObrm_M) | [02_cs224r_imitation_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/02_cs224r_imitation_2025.pdf) | Chi et al. (2024), [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137v5) | [wiki](wiki/02-imitation-learning.md) · [slides](raw/slides/02-imitation-learning.md) · [transcript](raw/transcripts/02-imitation-learning.md) |
| 3 | Wed Apr 9 | Policy Gradients | [KCAOXd4IO9o](https://www.youtube.com/watch?v=KCAOXd4IO9o) | [03_cs224r_policy_gradients_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/03_cs224r_policy_gradients_2025.pdf) | Williams (1992), [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696) | [wiki](wiki/03-policy-gradients.md) · [slides](raw/slides/03-policy-gradients.md) · [transcript](raw/transcripts/03-policy-gradients.md) |
| 4 | Fri Apr 11 | Actor-Critic Methods | [oejFZShW9hU](https://www.youtube.com/watch?v=oejFZShW9hU) | [04_cs224r_actor_critic_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/04_cs224r_actor_critic_2025.pdf) | Sutton et al. (1999), [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf); Schulman et al. (2017), [Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347) | [wiki](wiki/04-actor-critic.md) · [slides](raw/slides/04-actor-critic.md) · [transcript](raw/transcripts/04-actor-critic.md) |
| 5 | Wed Apr 16 | Off-Policy Actor Critic | [cRGKc-nAWho](https://www.youtube.com/watch?v=cRGKc-nAWho) | [05_cs224r_offpolicy_actor_critic_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/05_cs224r_offpolicy_actor_critic_2025.pdf) | Mnih et al. (2013), [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) | [wiki](wiki/05-off-policy-actor-critic.md) · [slides](raw/slides/05-off-policy-actor-critic.md) · [transcript](raw/transcripts/05-off-policy-actor-critic.md) |
| 6 | Fri Apr 18 | Q-learning | [-7kv6jf0isQ](https://www.youtube.com/watch?v=-7kv6jf0isQ) | [06_cs224r_qlearning_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/06_cs224r_qlearning_2025.pdf) | Hasselt et al. (2015), [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461); Bellemare et al. (2017), [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1806.10293) | [wiki](wiki/06-q-learning.md) · [slides](raw/slides/06-q-learning.md) · [transcript](raw/transcripts/06-q-learning.md) |
| 7 | Wed Apr 23 | Offline RL | [lRDaXnPIzks](https://www.youtube.com/watch?v=lRDaXnPIzks) | [07_cs224r_offline_rl_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/07_cs224r_offline_rl_2025.pdf) | Kumar et al. (2020), [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779); Kostrikov et al. (2021), [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/1906.08253) — the site gives this the same arXiv id as Janner et al. (2019) under lecture 11 | [wiki](wiki/07-offline-rl.md) · [slides](raw/slides/07-offline-rl.md) · [transcript](raw/transcripts/07-offline-rl.md) |
| 8 | Fri Apr 25 | Reward Learning | [PDIxDhA9Z6Y](https://www.youtube.com/watch?v=PDIxDhA9Z6Y) | [08_cs224r_reward_learning_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/08_cs224r_reward_learning_2025.pdf) | Christiano et al. (2017), [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) | not yet |
| 9 | Wed Apr 30 | Guest lecture: RL for LLMs: Preference Optimization (Archit Sharma) | [XKLGuwvSKvI](https://www.youtube.com/watch?v=XKLGuwvSKvI) | [09_cs224r-2025-rlhf.pdf](https://cs224r.stanford.edu/spring_2025/slides/09_cs224r-2025-rlhf.pdf) | Rafailov et al. (2023), [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | not yet |
| 10 | Fri May 2 | Guest lecture: RL for LLMs: Reasoning (Aviral Kumar) | [O2VpNnwB4lM](https://www.youtube.com/watch?v=O2VpNnwB4lM) | [10_cs224r-rl_for_reasoning_lecture.pdf](https://cs224r.stanford.edu/spring_2025/slides/10_cs224r-rl_for_reasoning_lecture.pdf) | none | not yet |
| 11 | Wed May 7 | Model-based RL | [PvqyGnOirgA](https://www.youtube.com/watch?v=PvqyGnOirgA) | [11_cs224r_mbrl_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/11_cs224r_mbrl_2025.pdf) | Janner et al. (2019), [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253) | not yet |
| 12 | Fri May 9 | Multi-Task and Goal-Conditioned RL | [qNdsI_4AQJw](https://www.youtube.com/watch?v=qNdsI_4AQJw) | [12_cs224r_mtrl_gcrl_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/12_cs224r_mtrl_gcrl_2025.pdf) | Andrychowicz et al. (2018), Hindsight Experience Replay; Rakelly et al. (2019). The site gives one link, [arXiv 1903.08254](https://arxiv.org/abs/1903.08254), spanning both entries | not yet |
| 13 | Wed May 14 | Meta-RL | [wSiyEpvoGkA](https://www.youtube.com/watch?v=wSiyEpvoGkA) | [13_cs224r_metarl_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/13_cs224r_metarl_2025.pdf) | Duan et al. (2016), [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779) | not yet |
| 14 | Fri May 16 | Exploration | [4tlSKdi8teU](https://www.youtube.com/watch?v=4tlSKdi8teU) | [14_cs224r_exploration_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/14_cs224r_exploration_2025.pdf) | Liu et al. (2021), [Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices](https://arxiv.org/abs/2008.02790) | not yet |
| 15 | Wed May 21 | Hierarchical RL and IL | [iKWYLSVAtfM](https://www.youtube.com/watch?v=iKWYLSVAtfM) | [15_cs224r_hierarchy_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/15_cs224r_hierarchy_2025.pdf) | Ahn et al. (2022), [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691) | not yet |
| 16 | Fri May 23 | RL for Robots: Autonomous Learning | [rbaWQQLrzl0](https://www.youtube.com/watch?v=rbaWQQLrzl0) | [16_cs224r_autonomy_2025.pdf](https://cs224r.stanford.edu/spring_2025/slides/16_cs224r_autonomy_2025.pdf) | none | not yet |
| 17 | Wed May 28 | Guest lecture: RL for Robots: Sim-to-Real Transfer (Ashish Kumar); catalog title "Advancing Robot Intelligence" | [Hp1WBWghrak](https://www.youtube.com/watch?v=Hp1WBWghrak) | [17_rl_for_robotics.pdf](https://cs224r.stanford.edu/spring_2025/slides/17_rl_for_robotics.pdf) — match to the video unconfirmed | none | not yet |
| 18 | Fri May 30 | Frontiers | [FacJ_1tTSx4](https://www.youtube.com/watch?v=FacJ_1tTSx4) | [18_cs224r_frontiers_how_to_research.pdf](https://cs224r.stanford.edu/spring_2025/slides/18_cs224r_frontiers_how_to_research.pdf) | none | not yet |
| 19 | Fri Apr 18 (TA session) | Extra section on Q-learning | [07MQNMcxhZU](https://www.youtube.com/watch?v=07MQNMcxhZU) | [section_q_learning_tutorial.pdf](https://cs224r.stanford.edu/spring_2025/slides/section_q_learning_tutorial.pdf) | none | not yet |

The "#" column follows the Cairn catalog's playlist order, which puts the Q-learning TA session
last although it was held in week 3.

## Assignments and project

Due dates are 11:59 pm PT on Gradescope, per the site.

| Document | Out | Due | URL |
| --- | --- | --- | --- |
| Homework 1 (imitation learning) | Fri Apr 4 | Fri Apr 18 | <https://cs224r.stanford.edu/spring_2025/material/CS224R_2025_Homework_1.pdf> |
| Homework 2 (online RL) | Fri Apr 18 | Fri May 2 | <https://cs224r.stanford.edu/spring_2025/material/CS224R_2025_Homework_2.pdf> |
| Homework 3 (offline RL) | Fri May 2 | Fri May 16 | <https://cs224r.stanford.edu/spring_2025/material/CS224r_Homework3_updated.pdf> |
| Homework 4 | Fri May 16 | Fri May 30 | <https://cs224r.stanford.edu/spring_2025/material/CS224R_2025_Homework_4.pdf> |
| Default final project guidelines | | | <https://cs224r.stanford.edu/spring_2025/material/CS224R_Default_Project_Guidelines.pdf> |
| Custom final project guidelines | | | <https://cs224r.stanford.edu/spring_2025/material/CS224R_Custom_Project_Guidelines.pdf> |

The homework topics in brackets are from lecture 1's slide 12, which also names Homework 4.
The site also links LaTeX templates and starter-code zips beside each homework.

## Background reading the site recommends

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  <http://incompleteideas.net/book/RLbook2020.pdf>. The site recommends chapters 3 and 4 for
  MDPs.
- CS221's modules on MDPs and RL, <https://stanford-cs221.github.io/autumn2022/modules/>.
