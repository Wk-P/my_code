# DQN — Deep Q-Network for Service Deployment

## Overview

This module trains a **Deep Q-Network (DQN)** agent to solve the service-to-ECU deployment task.  
Unlike the PPO-based approaches (Problems 3–5), DQN uses an **off-policy, experience-replay** training loop and enforces constraints via **reward shaping with hard episode termination** — no action masking is applied.

---

## Problem Definition

| Symbol | Meaning                         |
| ------ | ------------------------------- |
| $N$    | Number of ECUs                  |
| $M$    | Number of services ($M \leq N$) |
| $e_j$  | VM capacity of ECU $j$          |
| $n_i$  | VM requirement of service $i$   |

**Objective:** $\max AR = \frac{1}{M}\sum_{i,j} x_{ij} \frac{n_i}{e_j}$

---

## MDP Formulation

### State Space

$$s_t = \bigl[\, d_t,\; AR_t,\; r_0^t,\; \ldots,\; r_{N-1}^t \,\bigr] \in \mathbb{R}^{N+2}$$

| Index          | Content                             | Range   |
| -------------- | ----------------------------------- | ------- |
| 0              | Current service demand (normalised) | $[0,1]$ |
| 1              | Cumulative AR                       | $[0,1]$ |
| $2 \ldots N+1$ | Remaining capacity fraction per ECU | $[0,1]$ |

### Action Space

$a_t \in \{0, \ldots, N-1\}$ — discrete selection of an ECU.

### Reward Function (Dense + Penalty)

```
Infeasible assignment (capacity exceeded OR duplicate ECU):
    reward = -1.0,  episode terminates immediately

Valid assignment (not the last step):
    reward = ru / M   (immediate, proportional contribution to AR)

Valid assignment (last step, all M services placed):
    reward = ru / M   (same as intermediate)
```

The cumulative reward over a complete $M$-step episode with zero violations equals the final AR.

| Scenario            | Reward                        | Done |
| ------------------- | ----------------------------- | ---- |
| Invalid ECU         | $-1$                          | ✓    |
| Valid, intermediate | $n_{a_t} / (M \cdot e_{a_t})$ | ✗    |
| Valid, last step    | $n_{a_t} / (M \cdot e_{a_t})$ | ✓    |

### Key Difference from PPO Approaches

| Aspect              | DQN (this module)     | P3 PPO            | P4 PPO + Masking | P5 PPO + Lagrangian |
| ------------------- | --------------------- | ----------------- | ---------------- | ------------------- |
| Reward              | **Dense** (per step)  | Sparse (final AR) | Sparse           | Dense (Lagrangian)  |
| Constraint handling | Termination + penalty | None              | Hard mask        | Soft penalty        |
| On/off policy       | **Off-policy**        | On-policy         | On-policy        | On-policy           |
| Replay buffer       | ✓                     | ✗                 | ✗                | ✗                   |

---

## Algorithm & Hyperparameters

**Algorithm:** DQN (Stable-Baselines3)

| Hyperparameter           | Value                       |
| ------------------------ | --------------------------- |
| Learning rate            | `1e-3`                      |
| Replay buffer size       | `100,000`                   |
| Learning starts          | `2,000`                     |
| Batch size               | `64`                        |
| Target network $\tau$    | `1.0` (hard update)         |
| Discount factor $\gamma$ | `0.99`                      |
| Training frequency       | every `4` steps             |
| Gradient steps           | `1`                         |
| Target update interval   | every `500` steps           |
| Exploration fraction     | `0.5` (50 % of total steps) |
| Final $\epsilon$         | `0.05`                      |
| Q-network hidden layers  | `[128, 128]`                |
| Total training steps     | `500,000`                   |

The exploration schedule linearly decays $\epsilon$ from `1.0` to `0.05` over the first 50 % of training steps.

---

## Files

| File          | Description                                               |
| ------------- | --------------------------------------------------------- |
| `env.py`      | `DQNEnv` — dense reward, episode termination on violation |
| `train.py`    | DQN training loop, training curve plot, log               |
| `evaluate.py` | Load model, evaluate AR and success rate                  |
| `run_all.py`  | Full pipeline: ILP → random → DQN train → evaluate → plot |
| `config.py`   | All hyperparameters and paths                             |
| `results/`    | Models, training logs, plots, summary CSV                 |

---

## Usage

```bash
# Full pipeline
python dqn/run_all.py

# Train only
python dqn/train.py

# Evaluate only
python dqn/evaluate.py
```

---

## Design Notes

- **Dense rewards** give DQN an immediate signal at every step, which can accelerate early learning compared to sparse-reward PPO.
- **Episode termination on violation** is a simple, effective hard-constraint mechanism that does not require `sb3-contrib` or Lagrangian tuning.
- Because DQN uses an **experience replay buffer**, it can be more sample-efficient than on-policy PPO for small discrete action spaces.
- The Q-network with `[128, 128]` hidden layers is deliberately smaller than the PPO architectures, since DQN update targets are noisier and larger networks can overfit to stale buffer transitions.
- Total training budget (`500,000` steps) is larger than P3/P4 (`200,000`) to compensate for the off-policy sample correlation at the start of training (before the buffer is sufficiently diverse).
