# Problem 5 — PPO with Lagrangian Constraint Relaxation

## Overview

Problem 5 trains a standard **PPO agent** with **Lagrangian constraint relaxation** to handle the deployment constraints as **soft penalties** rather than hard blocks.  
A Lagrangian multiplier $\lambda$ is maintained and updated via **dual ascent** throughout training, adaptively increasing the penalty for constraint violations until the agent learns to avoid them.

---

## Problem Definition

Same optimisation objective as Problems 3 and 4:

$$\max \; AR = \frac{1}{M} \sum_{i,j} x_{ij} \cdot \frac{n_i}{e_j}$$

Subject to:

1. Each service is assigned to exactly one ECU.
2. Each ECU hosts at most one service.
3. $\sum_i x_{ij} n_i \leq e_j$ for all $j$ (capacity constraint).

In Problem 5, these constraints are **not hard-blocked** but penalised via a Lagrangian term in the reward.

---

## MDP Formulation

### State Space

$$s_t = \bigl[\, d_t,\; AR_t,\; r_0^t,\; \ldots,\; r_{N-1}^t \,\bigr] \in \mathbb{R}^{N+2}$$

| Index          | Content                                                  | Range     |
| -------------- | -------------------------------------------------------- | --------- |
| 0              | Service demand (normalised)                              | $[0,1]$   |
| 1              | Cumulative AR                                            | $[0,1]$   |
| $2 \ldots N+1$ | Remaining capacity fraction (can be $< 0$ if overloaded) | $[-2, 1]$ |

### Action Space

$a_t \in \{0, \ldots, N-1\}$ — no masking; all ECUs are always selectable.

### Reward Function (Lagrangian)

At each step $t$, a violation indicator is computed:

$$c_t = \begin{cases} 1 & \text{if remaining\_vms}[a_t] < n_t \;\text{OR}\; \text{ECU}\;a_t\;\text{already assigned} \\ 0 & \text{otherwise} \end{cases}$$

The per-step reward is:

$$r_t = \frac{n_{a_t}}{M \cdot e_{a_t}} - \lambda \cdot c_t$$

where $\lambda \geq 0$ is the **Lagrangian multiplier** (dual variable).

A complete $M$-step episode with zero violations yields cumulative reward equal to AR.

### Episode Structure

Episodes always run for exactly $M$ steps. `remaining_vms` can go negative (overloaded ECU).

---

## Lagrangian Multiplier Update (Dual Ascent)

$\lambda$ is updated by a training callback every `LAMBDA_UPDATE_WINDOW` episodes:

$$\lambda \leftarrow \text{clip}\!\bigl(\lambda + \alpha \cdot (\bar{v} - v^*),\; 0,\; \lambda_{\max}\bigr)$$

| Symbol           | Meaning                                  | Default value   |
| ---------------- | ---------------------------------------- | --------------- |
| $\alpha$         | Dual ascent step size (`LAMBDA_LR`)      | `0.05`          |
| $\bar{v}$        | Recent average per-step violation rate   | measured online |
| $v^*$            | Target violation rate (`LAMBDA_TARGET`)  | `0.0`           |
| $\lambda_{\max}$ | Maximum allowed $\lambda$ (`LAMBDA_MAX`) | `10.0`          |
| Window           | Episodes used to estimate $\bar{v}$      | `30`            |

When $\bar{v} > v^*$, $\lambda$ increases, making violations more costly and steering the policy away from infeasible actions.

---

## Algorithm & Hyperparameters

**Algorithm:** PPO (Stable-Baselines3, no masking)

| Hyperparameter    | Value        |
| ----------------- | ------------ |
| Learning rate     | `3e-4`       |
| `n_steps`         | `2048`       |
| Batch size        | `256`        |
| Epochs            | `10`         |
| $\gamma$          | `0.99`       |
| GAE $\lambda$     | `0.95`       |
| Clip range        | `0.2`        |
| Actor network     | `[128, 128]` |
| Critic network    | `[256, 256]` |
| Total steps       | `300,000`    |
| Initial $\lambda$ | `0.0`        |

---

## Comparison with Other Approaches

|                         | P3 (PPO, no constraint) | P4 (PPO + masking) | **P5 (PPO + Lagrangian)** |
| ----------------------- | ----------------------- | ------------------ | ------------------------- |
| Constraint handling     | None                    | Hard (mask)        | **Soft (penalty)**        |
| Violation rate          | > 0 %                   | 0 %                | ≈ 0 % (adaptive)          |
| Dense reward            | No (sparse)             | No (sparse)        | **Yes**                   |
| Requires `sb3-contrib`  | No                      | Yes                | No                        |
| $\lambda$ tuning needed | No                      | No                 | **Yes**                   |

---

## Files

| File          | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `env.py`      | `LagrangeEnv` — per-step Lagrangian reward, `set_lambda()` API |
| `train.py`    | PPO training with dual-ascent callback                         |
| `evaluate.py` | Load model, evaluate AR and violation rate                     |
| `run.py`      | Full pipeline                                                  |
| `config.py`   | All hyperparameters, Lagrangian settings, and paths            |
| `results/`    | Models, logs, plots, summary CSV                               |

---

## Usage

```bash
python problem5_ppo_lagrangian/run.py
```

---

## Design Notes

- **Dense reward** (`ru/M` at every step) provides richer gradient signal than the sparse AR-at-end reward in P3/P4, which can accelerate early-stage learning.
- **Adaptive $\lambda$** avoids the need to hand-tune a fixed penalty coefficient; the dual ascent loop finds the minimal $\lambda$ that drives violations to the target rate.
- Unlike P4, the agent has access to infeasible ECUs; this can lead to higher AR in unconstrained scenarios but requires $\lambda$ to be large enough to deter violations.
- If $\lambda$ grows too slowly, the agent may lock in a high-AR but violation-prone policy before the penalty becomes significant.
