# Problem 3 (Optimised Variant) — PPO for Service Deployment (No Constraint Enforcement)

## Overview

`problem3_ppo_opt` is an **experimental copy** of `problem3_ppo` that shares the same environment design and algorithm.  
It was created to allow parallel hyperparameter experiments or ablation studies without touching the original Problem 3 directory.  
The underlying MDP formulation, reward design, and evaluation protocol are **identical** to Problem 3.

For the full theoretical background, refer to [problem3_ppo/docs_en.md](../problem3_ppo/docs_en.md).

---

## Differences from `problem3_ppo`

| Aspect           | `problem3_ppo`           | `problem3_ppo_opt`               |
| ---------------- | ------------------------ | -------------------------------- |
| Environment      | `P3Env` (no constraint)  | `P3Env` (no constraint) — same   |
| Network arch     | `[256, 256]`             | `[256, 256]` — same              |
| Training steps   | `200,000`                | `200,000` — same                 |
| Saved model path | `results/ppo_p3_model`   | `results/ppo_p3_model` — same    |
| Purpose          | Reference implementation | Experimentation / comparison run |

> If further optimisations are applied (e.g., different network size, reward shaping, curriculum), the differences should be documented here.

---

## MDP Formulation (summary)

**Objective:** Maximise AR = $\frac{1}{M}\sum x_{ij}\frac{n_i}{e_j}$.

**State:** $s_t = [d_t,\; AR_t,\; r_0^t, \ldots, r_{N-1}^t]$ — shape $(N+2)$.

**Action:** $a_t \in \{0, \ldots, N-1\}$ — select ECU for current service.

**Reward:**
$$r_t = \begin{cases} AR_M & t = M-1 \\ 0 & \text{otherwise} \end{cases}$$

No penalty for violations. Violations are recorded in `info` only.

---

## Algorithm & Hyperparameters

**Algorithm:** PPO (Stable-Baselines3)

| Hyperparameter | Value        |
| -------------- | ------------ |
| Learning rate  | `3e-4`       |
| `n_steps`      | `1024`       |
| Batch size     | `128`        |
| Epochs         | `10`         |
| $\gamma$       | `0.999`      |
| GAE $\lambda$  | `0.95`       |
| Clip range     | `0.2`        |
| Net arch       | `[256, 256]` |
| Total steps    | `200,000`    |

---

## Files

| File             | Description                       |
| ---------------- | --------------------------------- |
| `env_p3.py`      | Same `P3Env` as in `problem3_ppo` |
| `train_p3.py`    | Training loop                     |
| `evaluate_p3.py` | Evaluation script                 |
| `run_all.py`     | Full pipeline                     |
| `config.py`      | Hyperparameters and paths         |
| `docs.md`        | Chinese design document           |
| `results/`       | Outputs                           |

---

## Usage

```bash
python problem3_ppo_opt/run_all.py
```
