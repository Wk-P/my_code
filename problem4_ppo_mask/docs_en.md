# Problem 4 — PPO with Action Masking for Service Deployment

## Overview

Problem 4 trains a **MaskablePPO** agent (PPO extended with action masking) to solve the service deployment task with **hard constraint enforcement**.  
At every step, the environment exposes a boolean mask that marks infeasible ECU assignments as unavailable.  
The agent's policy logits for masked actions are set to $-\infty$ before sampling, so **constraint violations are impossible by design**.

---

## Problem Definition

Same as Problems 3 and 5:

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
| 0              | Service demand (normalised)         | $[0,1]$ |
| 1              | Cumulative AR                       | $[0,1]$ |
| $2 \ldots N+1$ | Remaining capacity fraction per ECU | $[0,1]$ |

### Action Mask

At each step `t`, `action_masks()` returns a boolean array of length $N$:

```
mask[j] = True   iff  (ECU j not yet used)  AND  (remaining_vms[j] >= n_t)
```

MaskablePPO applies the mask before the softmax:

$$\pi(a_t \mid s_t) \propto \exp\!\bigl(\text{logit}(a_t)\bigr) \cdot \mathbf{1}[\text{mask}[a_t]]$$

### Reward Function

$$r_t = \begin{cases} AR_M & t = M-1 \text{ (final step)} \\ \text{AR}_{\text{current}} & \text{if no valid action remains (early stop)} \\ 0 & \text{otherwise} \end{cases}$$

If at any step no valid ECU exists for the current service, the episode terminates early and the agent receives the AR achieved so far.

### Episode Structure

```
reset()
  ├── step 0: service_0 → ECU[a_0]  (mask filters infeasible ECUs)
  ├── step 1: service_1 → ECU[a_1]
  ├── ...
  └── step M-1 or early stop → reward = AR
```

---

## Constraint Guarantee

| Property                 | Guarantee                                          |
| ------------------------ | -------------------------------------------------- |
| Capacity violation       | **Impossible** — mask blocks over-capacity ECUs    |
| Duplicate ECU assignment | **Impossible** — mask blocks already-assigned ECUs |
| Violation rate           | **0 %** by construction                            |

---

## Algorithm & Hyperparameters

**Algorithm:** MaskablePPO (`sb3-contrib`)

| Hyperparameter | Value        |
| -------------- | ------------ |
| Learning rate  | `3e-4`       |
| `n_steps`      | `1024`       |
| Batch size     | `128`        |
| Epochs         | `10`         |
| $\gamma$       | `0.999`      |
| GAE $\lambda$  | `0.95`       |
| Clip range     | `0.2`        |
| Actor network  | `[128, 128]` |
| Critic network | `[256, 256]` |
| Total steps    | `200,000`    |

The actor and critic use **separate** network architectures (`dict(pi=[128,128], vf=[256,256])`).

---

## Comparison with Other Approaches

|                        | P2 (ILP)  | P3 (PPO, no constraint) | **P4 (PPO + masking)**     | P5 (PPO + Lagrangian)  |
| ---------------------- | --------- | ----------------------- | -------------------------- | ---------------------- |
| Constraint enforcement | Hard (LP) | None                    | **Hard (mask)**            | Soft (penalty)         |
| Violation rate         | 0 %       | > 0 %                   | **0 %**                    | ≈ 0 %                  |
| AR performance         | Optimal   | High (but may violate)  | **Slightly lower than P3** | Slightly lower than P3 |
| Infeasible handling    | N/A       | Assignment proceeds     | Episode ends early         | Assignment proceeds    |

---

## Files

| File             | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `env_p4.py`      | `P4Env` — `action_masks()` method, no-violation guarantee |
| `train_p4.py`    | MaskablePPO training loop                                 |
| `evaluate_p4.py` | Evaluation script                                         |
| `run_all.py`     | Full pipeline: ILP → random → PPO train+eval → plot       |
| `config.py`      | Hyperparameters and paths                                 |
| `results/`       | Models, logs, plots, summary CSV                          |

---

## Usage

```bash
# Full pipeline
python problem4_ppo_mask/run_all.py

# Train only
python problem4_ppo_mask/train_p4.py

# Evaluate only
python problem4_ppo_mask/evaluate_p4.py
```

---

## Design Notes

- **Action masking is the cleanest constraint-handling strategy** among the RL approaches: it requires no reward engineering and provably achieves zero violations.
- The trade-off is that early-stop episodes (no feasible ECU remaining) can reduce average AR compared to unconstrained P3.
- Scenarios with tight capacity constraints benefit the most from masking, as random guessing would waste many steps on infeasible ECUs.
