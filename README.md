# Exploring the Landscape of Differentiable Architecture Search with Reinforcement Learning

In recent times, differentiable architecture search (DARTS) has gained significant attention as an effective approach for Neural Architecture Search (NAS). However, DARTS is susceptible to the collapse problem, wherein the search algorithm selects non-parameteric operations, resulting in suboptimal architectures. To overcome this limitation, previous studies have employed gradient-based techniques. In this research, we extensively investigate the application of reinforcement learning (RL) to DARTS, exploring various combinations of state space, discrete and continuous action spaces, and three distinct reward functions: `accuracy`, `loss`, and `nwot`. Through experimental evaluations, our RL-based approach achieves a test error of 3.26% on CIFAR-10 within the DARTS search space and demonstrates its robustness in the S2 and S3 RobustDARTS search spaces.

The results of the experiments are as follows:
| Name                       | Mean            | Max            |
| -------------------------- | --------------- | -------------- |
| `disc-nwot`                | 96.74 ± 0.30    | 97.01%         |
| `disc-acc`                 | 93.78 ± 0.12    | 93.94%         |
| `disc-loss`                | 95.43 ± 0.06    | 95.49%         |
| `disc-ckpt-acc`            | 95.74 ± 0.14    | 95.87%         |
| `disc-ckpt-loss`           | 92.55 ± 0.39    | 92.98%         |
| `cont-acc-step`            | 96.08 ± 0.19    | 96.35%         |
| `cont-acc-grad-step`       | 96.24 ± 0.32    | 96.42%         |
| `cont-loss-step`           | 95.52 ± 0.16    | 95.70%         |
| `cont-loss-grad-step`      | 95.67 ± 0.29    | 96.00%         |

---
Requirements:
- `torch == 2.0.1`
- `stable_baselines3 >= 2.0.0a9`
- `tensorboard`
