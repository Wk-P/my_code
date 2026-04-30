# Experiment Report: ECU Service Placement Optimisation with Reinforcement Learning

**Date:** 2026-04-30  
**Experiment Series:** `ecu_eq_svc_p` · `ecu_gt_svc_p` · `ecu_lt_svc_p`  
**Branch:** `linux-cuda12.2`

---

## 1. Problem Definition

The task is to assign M software services (SVCs) to N ECUs (Electronic Control Units) in automotive systems. Each service has a resource requirement; each ECU has a finite capacity. The objective is to maximise **Average Resource Utilisation (AR)**:

$$AR = \frac{1}{|A|} \sum_{j \in A} \frac{\sum_{i \to j} \text{req}_i}{\text{cap}_j}$$

where $A$ is the set of active ECUs (those hosting at least one service).

**Constraints:**
- **Capacity:** total service requirements on an ECU must not exceed its capacity
- **Conflict:** services within the same conflict set must not share an ECU

The ILP (Integer Linear Programming) solver provides a provably optimal baseline via Dinkelbach fractional programming.

---

## 2. Experimental Scenarios

Three scenario variants are studied, each with 200 randomly generated scenarios and an 80/20 train/test split:

| Scenario | Variant | N (ECUs) | M (SVCs) | Feasible Scenarios | Key Challenge |
|----------|---------|----------|----------|--------------------|---------------|
| `ecu_eq_svc_p` | Equal | 10 | 10 | 200 / 200 | N = M, one-to-one assignment |
| `ecu_gt_svc_p` | ECU-rich | 15 | 10 | 200 / 200 | N > M, sparse placement |
| `ecu_lt_svc_p` | Service-dense | 10 | 15 | ~160 / 200 | N < M, bin packing + conflict |

The `lt` variant is the hardest: 15 services must be packed into 10 ECUs (~1.5 services per ECU on average), with significantly higher conflict density than the other two variants. Training and evaluation use only the 160 feasible scenarios.

---

## 3. RL Methods

Six RL agents are trained and evaluated per scenario. All share the same MLP policy backbone.

| ID | Method | Constraint Handling | Algorithm |
|----|--------|---------------------|-----------|
| P3 | PPO (unconstrained) | No enforcement — violations recorded only | PPO |
| P4 | PPO-Mask | Hard capacity action masking; conflicts soft-recorded | MaskablePPO |
| P5 | PPO-Lagrange | Hard capacity mask + adaptive Lagrangian conflict penalty | PPO + dual ascent |
| P6 | PPO+Repair | Best-fit repair heuristic on violation; repair step penalised | PPO |
| — | DQN | Violation penalty; episode terminates on violation (eq/gt) | DQN |
| — | DDQN | Same as DQN with double Q-network | DDQN |

### `lt`-Specific Design Differences

Due to the bin packing regime (N < M), `lt` differs from eq/gt in the following ways:

| Aspect | eq / gt | lt |
|--------|---------|----|
| P4 conflict handling | Soft-recorded | Capacity + conflict **dual masking** |
| DQN/DDQN violation termination | Terminate on violation | No early termination (avoids sparse signals) |
| DQN/DDQN terminal bonus | None | ±AR (provides episode-level reward signal) |
| P6 repair heuristic | Best-fit repair | Same as eq/gt |

### Observation Space Dimensions

| Scenario | P3 / P4 / P6 | P5 | Extra Feature |
|----------|--------------|----|---------------|
| eq / gt | `4N + 6 + M` | `4N + 7 + M` | — |
| lt | `5N + 6 + M` | `5N + 7 + M` | `ecu_allowed_frac[j]`: fraction of remaining services placeable on ECU j without conflict |

### Training Hyperparameters

| Parameter | PPO (P3–P6) | DQN / DDQN |
|-----------|------------|------------|
| Total steps (PPO) | 5,000,000 | — |
| Total steps (DQN) | — | 3,000,000 |
| N_ENVS (eq/gt) | 40 | 12 |
| N_ENVS (lt) | 16 | 6 |
| Network arch | [256, 256] | [128, 128] |
| Learning rate | 3e-4 | 1e-3 |
| Batch size | 128 | 64 |
| Discount γ | 0.999 | 0.99 |
| Eval episodes | 300 | 300 |
| Random seed | 42 | 42 |

---

## 4. Results

### 4.1 `ecu_eq_svc_p` (Equal, N=10, M=10)

ILP Optimal AR: **0.5321**

| Method | AR (mean) | AR (std) | Placed | Cap Viol | Conflict Viol | Gap to ILP |
|--------|-----------|----------|--------|----------|---------------|------------|
| ILP (Optimal) | 0.5321 | 0.000 | 10.0 | 0 | 0 | — |
| P3 PPO | 0.8691 | 0.063 | 10.0 | 0 | 1114 | *(infeasible)* |
| P4 PPO-Mask | 0.5220 | 0.070 | 10.0 | 0 | 0 | −1.9% |
| P5 PPO-Lagrange | 0.5230 | 0.070 | 10.0 | 0 | 1 | −1.7% |
| P6 PPO+Repair | 0.5242 | 0.070 | 10.0 | 0 | 8 | −1.5% |
| DQN | 0.5088 | 0.067 | 10.0 | 0 | 0 | −4.4% |
| DDQN | 0.5088 | 0.069 | 10.0 | 0 | 0 | −4.4% |

### 4.2 `ecu_gt_svc_p` (ECU-rich, N=15, M=10)

ILP Optimal AR: **0.6337**

| Method | AR (mean) | AR (std) | Placed | Cap Viol | Conflict Viol | Gap to ILP |
|--------|-----------|----------|--------|----------|---------------|------------|
| ILP (Optimal) | 0.6337 | 0.000 | 10.0 | 0 | 0 | — |
| P3 PPO | 0.8683 | 0.057 | 10.0 | 0 | 860 | *(infeasible)* |
| P4 PPO-Mask | 0.6135 | 0.064 | 9.97 | 0 | 0 | −3.2% |
| P5 PPO-Lagrange | 0.6144 | 0.065 | 10.0 | 0 | 0 | −3.0% |
| P6 PPO+Repair | 0.6174 | 0.065 | 10.0 | 0 | 7 | −2.5% |
| DQN | 0.5820 | 0.070 | 10.0 | 0 | 0 | −8.2% |
| DDQN | 0.5836 | 0.055 | 10.0 | 0 | 0 | −7.9% |

### 4.3 `ecu_lt_svc_p` (Service-dense, N=10, M=15)

ILP Optimal AR: **0.7390**

| Method | AR (mean) | AR (std) | Placed | Cap Viol | Conflict Viol | Gap to ILP |
|--------|-----------|----------|--------|----------|---------------|------------|
| ILP (Optimal) | 0.7390 | 0.000 | 15.0 | 0 | 0 | — |
| P3 PPO | 12.556† | 2.647 | 15.0 | 4495 | 3948 | *(infeasible)* |
| P4 PPO-Mask | 0.6822 | 0.067 | 15.0 | 0 | 0 | −7.7% |
| P5 PPO-Lagrange | 0.7070 | 0.056 | 15.0 | 8 | 364 | −4.3%‡ |
| P6 PPO+Repair | 0.6765 | 0.063 | 12.47 | 27 | 0 | −8.5%‡ |
| DQN | 0.6726 | 0.056 | 15.0 | 248 | 359 | −9.0%‡ |
| DDQN | 0.6852 | 0.060 | 15.0 | 259 | 399 | −7.3%‡ |

† P3 AR > 1 indicates a reward-scaling issue in the lt environment (unnormalised cumulative reward); the agent is infeasible.  
‡ Gap computed for reference only; these methods have non-zero violations and are **not fully feasible** in the lt scenario.

---

## 5. Cross-Scenario Analysis

### 5.1 AR Gap to ILP Optimal

| Method | eq | gt | lt | Feasible in lt? |
|--------|----|----|-----|----------------|
| P4 PPO-Mask | −1.9% | −3.2% | −7.7% | Yes (0 violations) |
| P5 PPO-Lagrange | −1.7% | −3.0% | −4.3% | No (372 violations) |
| P6 PPO+Repair | −1.5% | −2.5% | −8.5% | No (27 cap viol, 12.5/15 placed) |
| DQN | −4.4% | −8.2% | −9.0% | No (607 violations) |
| DDQN | −4.4% | −7.9% | −7.3% | No (658 violations) |

### 5.2 Constraint Satisfaction

| Method | eq conflicts | gt conflicts | lt cap viol | lt conflicts | Notes |
|--------|-------------|-------------|------------|-------------|-------|
| P3 PPO | 1114 | 860 | 4495 | 3948 | Infeasible — no constraint enforcement |
| P4 PPO-Mask | 0 | 0 | 0 | 0 | **Fully satisfied in all scenarios** |
| P5 PPO-Lagrange | 1 | 0 | 8 | 364 | Lagrangian insufficient for dense lt conflicts |
| P6 PPO+Repair | 8 | 7 | 27 | 0 | Repair fails under bin-packing pressure; 12.47/15 placed |
| DQN | 0 | 0 | 248 | 359 | No early-termination in lt → violations accumulate |
| DDQN | 0 | 0 | 259 | 399 | Same |

### 5.3 ILP Optimal AR Across Scenarios

| Scenario | ILP Optimal AR | Notes |
|----------|---------------|-------|
| eq (N=M=10) | 0.5321 | Baseline |
| gt (N=15, M=10) | 0.6337 | ECU-rich allows tighter packing on fewer high-capacity ECUs |
| lt (N=10, M=15) | 0.7390 | Bin packing forces high utilisation on active ECUs; ILP AR is highest of the three |

---

## 6. Conclusions

1. **P4 (PPO-Mask) is the only fully-feasible RL method across all three scenarios**, achieving 98.1–92.3% of ILP optimal AR (eq/gt/lt respectively). The gap widens significantly in lt due to the bin-packing + dense-conflict regime.

2. **The lt scenario breaks every other method's feasibility**: P5's Lagrangian penalty proves insufficient for the high conflict density (364 violations), P6's repair heuristic fails under packing pressure (27 cap violations, only 12.47/15 services placed on average), and DQN/DDQN accumulate hundreds of violations without the early-termination mechanism.

3. **P5 achieves the highest raw AR in lt (0.7070, −4.3% from ILP)** but is not feasible. Among fully-feasible methods, P4 at 0.6822 (−7.7%) is the best.

4. **P3's AR of 12.556 in lt indicates a reward-scaling bug** — the unnormalised cumulative reward exceeds the physical maximum of 1.0, confirming the agent is not computing AR correctly. The eq/gt P3 results (AR ≈ 0.87) do not exhibit this issue, suggesting the lt environment's step reward accumulation differs.

5. **The ILP optimal AR increases monotonically across scenarios (0.5321 → 0.6337 → 0.7390)**: bin-packing pressure in lt forces all 10 ECUs to be heavily loaded, yielding higher utilisation at the ILP optimum.

6. **DQN/DDQN gap to ILP widens with scenario complexity** (eq: −4.4%, gt: −8.2%, lt reference: −7.3–9.0%), and feasibility collapses in lt without early termination. PPO with proper constraint handling is more robust.

7. **PPO-family performance in eq/gt (−1.5% to −3.2% gap)** demonstrates the approach is viable for simpler scenarios; the lt results highlight the need for stronger constraint enforcement or tighter integration of the bin-packing structure into the action space.

---

## 7. Completed Steps

- [x] `ecu_eq_svc_p` and `ecu_gt_svc_p` experiments completed
- [x] `ecu_lt_svc_p` parallel run completed (all 6 problems)
- [x] `ecu_lt_svc_p/run_scripts/results_combined/combined.py` run, plot generated
- [x] All TBD fields in Section 4.3, Section 5, and Section 6 filled in
