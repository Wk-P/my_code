# Problem 3 — PPO for Service Deployment (No Constraint Enforcement)

## Overview

Problem 3 trains a **Proximal Policy Optimisation (PPO)** agent on the service deployment task **without enforcing any constraints**.  
Constraint violations are **recorded** for analysis but never penalised, and episodes always run for the full $M$ steps.  
The goal is to measure how well an unconstrained RL agent maximises AR and to quantify the resulting violation rate as a baseline for Problems 4 and 5.

---

## Problem Definition

| Symbol | Meaning                         |
| ------ | ------------------------------- |
| $N$    | Number of ECUs                  |
| $M$    | Number of services ($M \leq N$) |
| $e_j$  | VM capacity of ECU $j$          |
| $n_i$  | VM requirement of service $i$   |

**Objective — maximise Average Resource Utilisation (AR):**

$$\max \; AR = \frac{1}{M} \sum_{j} \sum_{i} x_{ij} \cdot \frac{n_i}{e_j}$$

---

## MDP Formulation

### State Space

$$s_t = \bigl[\, d_t,\; AR_t,\; r_0^t,\; \ldots,\; r_{N-1}^t \,\bigr] \in \mathbb{R}^{N+2}$$

| Index          | Content                                                             | Range     |
| -------------- | ------------------------------------------------------------------- | --------- |
| 0              | Current service demand (normalised by $\max e_j$)                   | $[-1, 1]$ |
| 1              | Cumulative AR at step $t$                                           | $[0, 1]$  |
| $2 \ldots N+1$ | Remaining capacity fraction per ECU (can be negative if overloaded) | $[-1, 1]$ |

### Action Space

$a_t \in \{0, \ldots, N-1\}$ — select the ECU for the current service.

### Reward Function (Sparse, No Constraint Penalty)

$$r_t = \begin{cases} AR_M & t = M-1 \text{ (final step)} \\ 0 & \text{otherwise} \end{cases}$$

No penalty is applied for capacity violations or duplicate ECU assignments.

### Episode Structure

Every episode runs exactly $M$ steps regardless of violations.  
Infeasible assignments are executed anyway (`remaining_vms` can go negative).

---

## Constraint Tracking (Evaluation Only)

During evaluation, the environment records:

| Metric                      | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `capacity_violations`       | Steps where demand exceeded ECU remaining capacity     |
| `single_service_violations` | Steps where an already-assigned ECU was selected again |
| `violation_rate`            | Total violations / $M$                                 |

These metrics are **not fed into the reward**, only logged in `info`.

---

## Algorithm & Hyperparameters

**Algorithm:** PPO (Stable-Baselines3)

| Hyperparameter             | Value                                                      |
| -------------------------- | ---------------------------------------------------------- |
| Learning rate              | `3e-4`                                                     |
| Rollout length (`n_steps`) | `1024`                                                     |
| Batch size                 | `128`                                                      |
| Epochs per update          | `10`                                                       |
| Discount factor $\gamma$   | `0.999` (high; final reward must back-propagate $M$ steps) |
| GAE $\lambda$              | `0.95`                                                     |
| Clip range                 | `0.2`                                                      |
| Network architecture       | `[256, 256]` (shared MLP for actor and critic)             |
| Total training steps       | `200,000`                                                  |

---

## Comparison with Other Approaches

|                        | P2 (ILP)            | **P3 (PPO, no constraint)** | P4 (PPO + masking) | P5 (PPO + Lagrangian) |
| ---------------------- | ------------------- | --------------------------- | ------------------ | --------------------- |
| Constraint enforcement | Hard (guaranteed)   | **None**                    | Hard (action mask) | Soft (penalty)        |
| Violation rate         | 0 %                 | Can be > 0 %                | ≈ 0 %              | ≈ 0 %                 |
| AR performance         | Optimal upper bound | High (but may violate)      | Slightly lower     | Slightly lower        |
| Inference speed        | Slow (NP-hard)      | $O(1)$                      | $O(1)$             | $O(1)$                |

---

## Files

| File             | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `env_p3.py`      | `P3Env` — Gymnasium environment with violation tracking only  |
| `train_p3.py`    | PPO training loop, training curve plot, raw log               |
| `evaluate_p3.py` | Load trained model, evaluate on all 200 scenarios             |
| `run_all.py`     | One-shot pipeline: ILP → random → PPO train → evaluate → plot |
| `config.py`      | All hyperparameters and paths                                 |
| `docs.md`        | Chinese design document                                       |
| `results/`       | Saved models, training logs, plots, summary CSV               |

---

## Usage

```bash
# Full pipeline (recommended)
python problem3_ppo/run_all.py

# Train only
python problem3_ppo/train_p3.py

# Evaluate only (requires trained model)
python problem3_ppo/evaluate_p3.py
```

---

## Expected Behaviour

- The agent will learn to maximise AR and can achieve values close to the ILP upper bound.
- However, because no constraint is enforced, a non-zero violation rate is expected, especially when the agent needs to match large services to large ECUs.
- The violation rate serves as a baseline for evaluating the constraint-handling strategies in Problems 4 and 5.
