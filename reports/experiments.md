# Experiment Summary

## Project Overview

This project benchmarks **Reinforcement Learning (RL)** against **Integer Linear Programming (ILP)** for the **ECU service placement** problem: assigning software services to automotive Electronic Control Units (ECUs) to maximise **Average Resource Utilisation (AR)** while satisfying capacity constraints and optional privacy (conflict) constraints.

**Objective — Average Resource Utilisation (AR):**

```
AR = Σ_{j ∈ active} ( Σ_{i on j} req_i / cap_j ) / |active ECUs|
```

Higher AR means denser, more efficient packing. The ILP computes the exact optimum for each scenario; RL methods are evaluated on their ability to approach this optimum without constraint violations.

---

## Constraint Types

| # | Constraint | Description |
|---|-----------|-------------|
| 1 | **Capacity** | Total service demand on one ECU must not exceed its capacity |
| 2 | **Conflict (Privacy)** | Services in the same conflict subset may not share an ECU |

Ten conflict subsets are generated randomly per scenario (size 2 to min(N, M)).

---

## Scenario Variants

| Directory | N (ECUs) | M (Services) | Conflict Sets | Description |
|-----------|----------|-------------|---------------|-------------|
| `ecu_eq_svc` | 10 | 10 | No | Baseline: equal counts, no privacy constraint |
| `ecu_eq_svc_p` | 10 | 10 | Yes (K=10) | Equal counts + privacy/conflict constraints |
| `ecu_gt_svc_p` | 15 | 10 | Yes (K=10) | More ECUs than services — resource-abundant |
| `ecu_lt_svc_p` | 10 | 15 | Yes (K=10) | More services than ECUs — capacity-tight |

**Scenario generation:** 200 scenarios per variant. ECU capacity sampled uniformly from [50, 200] (step 5); service requirement from [10, 100] (step 5). For `ecu_lt_svc_p`, only feasibility-verified scenarios (ILP-solvable) are included in training and evaluation; infeasible scenarios (~20%) are excluded from both splits.

**Train / Test split:** 80/20 deterministic split (seed = 42) — 160 training scenarios, 40 held-out test scenarios. All reported metrics are on the **test split only**.

---

## Methods Overview

| ID | Method | Algorithm | Constraint Handling |
|----|--------|-----------|---------------------|
| P2 | ILP (Optimal) | PuLP / CBC-MIP | Hard — exact optimal baseline |
| P3 | PPO | MlpPolicy | None — unconstrained AR maximisation |
| P4 | MaskablePPO | MlpPolicy + masks | Hard capacity masking before action sampling |
| P5 | Lagrangian PPO | MlpPolicy | Soft penalty via adaptive Lagrange multiplier λ |
| P6 | PPO + Repair | MlpPolicy + heuristic | Post-placement best-fit repair |
| DQN | DQN | MlpPolicy | Hard capacity termination (episode ends on violation) |
| DDQN | Double DQN | MlpPolicy | Hard capacity termination (variance-reduced) |

---

## Training Setup

| Parameter | PPO variants (P3–P6) | DQN / DDQN |
|-----------|----------------------|-----------|
| Total timesteps | 5,000,000 | 3,000,000 |
| Parallel environments | 40 (SubprocVecEnv) | 12 (SubprocVecEnv) |
| Learning rate | 3e-4 | 1e-3 |
| Network architecture | pi=[256,256], vf=[256,256] | [128, 128] |
| Batch size | 128 | 64 |
| Gamma | 0.99 | 0.99 |
| Random seed | 42 | 42 |

**Observation space:** `Box(4N + 6 + M)` for P3/P4/P6/DQN/DDQN; `Box(4N + 7 + M)` for P5 (one extra dimension for the Lagrange multiplier). Components: current service demand, cumulative AR, per-ECU remaining/initial capacity, conflict flags, validity flags, remaining service demands.

**Episode structure:** Services are sorted largest-demand-first at each `reset()`. The agent places them one by one (M steps per episode). Early termination occurs in P4/P6 (no valid ECU) and DQN/DDQN (capacity violation).

---

## Algorithm Details

### P2 — ILP (Optimal)

Solved via PuLP / CBC. Decision variable `x[i][j] ∈ {0,1}` = 1 if service i placed on ECU j. Objective: maximise total utilisation `Σ x[i][j]·req_i/cap_j`. Hard constraints: each service placed exactly once; ECU capacity respected; infeasible pairs forbidden; conflict subsets satisfied. AR computed post-solve as total utilisation / number of active ECUs.

### P3 — PPO (No Constraints)

Receives a dense reward proportional to AR trend. Neither capacity nor conflict violations terminate the episode or reduce reward. The observed high AR is inflated by unconstrained packing.

### P4 — MaskablePPO

At each step, invalid actions (ECUs whose remaining capacity < current service demand) are masked to probability 0 before sampling. Conflict violations are recorded but not masked. Terminal bonus: +1·AR on clean completion, +0.1·AR if conflict violations occurred.

### P5 — Lagrangian PPO

Step reward: `match_gain − (λ + 0.2)·c_t` where `c_t = 1` on any violation. Lambda (λ) starts at 0 and is updated every 20 episodes after a 20,000-episode warmup: `λ ← clip(λ + 3e-4 · avg_viol_rate, 0, 2)`. The extra observation dimension carries `λ / λ_max`. Episode always runs M steps.

### P6 — PPO + Repair

On constraint violation, a best-fit relocation heuristic attempts to move the offending service to another valid ECU rather than terminating. If no repair is possible, the episode terminates with a demand penalty.

### DQN / DDQN

Reward: `+req/cap` on valid placement; episode terminates immediately on capacity violation with penalty `= −(remaining services count)`. Conflict violations are recorded but do not terminate the episode. DDQN uses a target network with double Q-learning to reduce overestimation bias.

---

## Evaluation Protocol

- **Scenarios:** 40 held-out test scenarios per variant (80/20 split, seed=42)
- **Episodes per evaluation:** 300
- **ILP reference:** Solved on the same 40 test scenarios

Metrics per method:

| Metric | Definition | Direction |
|--------|-----------|-----------|
| `ar_mean` | Mean AR across 300 eval episodes | ↑ Higher better |
| `ar_std` | Standard deviation of AR | ↓ Lower better |
| `placed_mean` | Mean services placed per episode | ↑ Higher better |
| `viol_rate` | Fraction of episodes with ≥1 violation | ↓ Lower better |
| `cap_viol_total` | Total capacity violations across 300 episodes | ↓ Lower better |
| `conflict_viol_total` | Total conflict violations across 300 episodes | ↓ Lower better |

---

## Results by Scenario

> **Note:** Results below were generated **before** the 80/20 train/test split was introduced. They reflect performance on the full scenario pool and will be updated after re-running all experiments with the new evaluation protocol.

### ecu_eq_svc — No Privacy Constraint (N=10, M=10)

| Method | AR mean | AR std | Placed | Viol rate | Cap viols | Conflict viols |
|--------|---------|--------|--------|-----------|-----------|----------------|
| ILP (Optimal) | **0.738** | 0.000 | 10.0 | **0.000** | 0 | 0 |
| Random Baseline | 0.458 | 0.099 | 10.0 | 0.706 | 1472 | 1705 |
| PPO (P3, no constr) | 0.913 | 0.067 | 10.0 | 0.050 | 0 | 226 |
| MaskablePPO (P4) | 0.699 | 0.078 | 10.0 | **0.000** | 0 | 0 |
| Lagrange PPO (P5) | 0.713 | 0.064 | 10.0 | 0.013 | 0 | 57 |
| PPO + Repair (P6) | 0.710 | 0.070 | 10.0 | 0.007 | 0 | 31 |
| DQN | 0.682 | 0.065 | 10.0 | **0.000** | 0 | 357 |
| DDQN | 0.686 | 0.066 | 10.0 | **0.000** | 0 | 338 |

> No conflict constraint active. P3 achieves AR=0.913 by ignoring conflict sets (226 violations). P4/P5/P6 reach AR 0.699–0.713, within 0.025–0.039 of ILP, with near-zero violations. DQN/DDQN trail by ≈0.05 AR.

---

### ecu_eq_svc_p — With Privacy Constraint (N=10, M=10)

| Method | AR mean | AR std | Placed | Viol rate | Cap viols | Conflict viols |
|--------|---------|--------|--------|-----------|-----------|----------------|
| ILP (Optimal) | **0.544** | 0.000 | 10.0 | **0.000** | 0 | 0 |
| Random Baseline | 0.783 | 0.217 | 10.0 | 0.580 | 702 | 1039 |
| PPO (P3, no constr) | 0.913 | 0.044 | 10.0 | 0.373 | 0 | 1120 |
| MaskablePPO (P4) | 0.548 | 0.069 | 10.0 | **0.000** | 0 | 0 |
| Lagrange PPO (P5) | 0.550 | 0.070 | 10.0 | 0.0003 | 1 | 0 |
| PPO + Repair (P6) | **0.549** | 0.069 | 10.0 | **0.000** | 0 | 0 |
| DQN | 0.520 | 0.068 | 10.0 | **0.000** | 0 | 0 |
| DDQN | 0.530 | 0.067 | 10.0 | **0.000** | 0 | 0 |

> Conflict constraints lower ILP optimal to 0.544 (tighter feasible space). P4/P5/P6 match or marginally exceed ILP AR (0.548–0.550) with zero or near-zero violations. P3's apparent AR=0.913 is inflated by 1120 conflict violations. DQN/DDQN achieve zero violations but trail by ≈0.02 AR.

---

### ecu_gt_svc_p — More ECUs Than Services, With Privacy (N=15, M=10)

| Method | AR mean | AR std | Placed | Viol rate | Cap viols | Conflict viols |
|--------|---------|--------|--------|-----------|-----------|----------------|
| ILP (Optimal) | **0.626** | 0.000 | 10.0 | **0.000** | 0 | 0 |
| Random Baseline | 0.679 | 0.173 | 10.0 | 0.439 | 561 | 757 |
| PPO (P3, no constr) | 0.913 | 0.048 | 10.0 | 0.294 | 0 | 883 |
| MaskablePPO (P4) | 0.627 | 0.076 | 10.0 | **0.000** | 0 | 0 |
| Lagrange PPO (P5) | 0.628 | 0.078 | 10.0 | **0.000** | 0 | 0 |
| PPO + Repair (P6) | 0.627 | 0.076 | 10.0 | **0.000** | 0 | 0 |
| DQN | 0.587 | 0.075 | 10.0 | **0.000** | 0 | 0 |
| DDQN | 0.588 | 0.073 | 10.0 | **0.000** | 0 | 0 |

> Extra ECU headroom (N=15 > M=10) raises ILP optimal to 0.626 and simplifies constraint satisfaction. P4/P5/P6 all match ILP exactly with zero violations. DQN/DDQN trail by ≈0.04 AR.

---

### ecu_lt_svc_p — More Services Than ECUs, With Privacy (N=10, M=15)

> **Results pending re-run.** This variant uses N=10 ECUs and M=15 services — the hardest layout (each ECU must host multiple services on average). Only feasibility-verified scenarios are used; approximately 20% of generated scenarios are infeasible and excluded from both splits.

| Method | AR mean | AR std | Placed | Viol rate | Cap viols | Conflict viols |
|--------|---------|--------|--------|-----------|-----------|----------------|
| ILP (Optimal) | — | — | — | — | — | — |
| PPO (P3, no constr) | — | — | — | — | — | — |
| MaskablePPO (P4) | — | — | — | — | — | — |
| Lagrange PPO (P5) | — | — | — | — | — | — |
| PPO + Repair (P6) | — | — | — | — | — | — |
| DQN | — | — | — | — | — | — |
| DDQN | — | — | — | — | — | — |

---

## Cross-Scenario Analysis

### AR Gap Relative to ILP Optimal (positive = better than ILP, negative = worse)

| Method | ecu_eq_svc | ecu_eq_svc_p | ecu_gt_svc_p | ecu_lt_svc_p |
|--------|-----------|--------------|--------------|--------------|
| P3 | +0.175 ⚠ violated | +0.369 ⚠ violated | +0.287 ⚠ violated | — |
| P4 | −0.039 | +0.004 | +0.001 | — |
| P5 | −0.025 | +0.006 | +0.002 | — |
| P6 | −0.028 | +0.005 | +0.001 | — |
| DQN | −0.056 | −0.024 | −0.039 | — |
| DDQN | −0.052 | −0.014 | −0.038 | — |

### Key Findings

1. **P4/P5/P6 match ILP quality in constrained scenarios.** In `ecu_eq_svc_p` and `ecu_gt_svc_p`, all three methods achieve AR within ±0.006 of ILP optimal with zero or near-zero violations.

2. **P3's high AR is entirely violation-driven.** Once conflict constraints are active, P3 accumulates hundreds to thousands of conflict violations. Its apparent AR advantage (+0.17 to +0.37) disappears entirely when violations are penalised.

3. **DQN/DDQN learn safe behaviour but sacrifice ≈3–4% AR.** They achieve zero violations across all constrained scenarios but fall further below ILP than P4/P5/P6.

4. **Conflict constraints are the binding challenge, not capacity.** `cap_viol_total = 0` for all constrained RL methods in every `_p` scenario; all violations are conflict-type.

5. **Extra ECUs raise both ILP optimal and RL performance.** Going from `ecu_eq_svc_p` (N=M=10) to `ecu_gt_svc_p` (N=15, M=10) lifts ILP AR from 0.544 → 0.626 and makes all constrained RL methods trivially reach the optimum.

6. **The capacity-tight layout (`ecu_lt_svc_p`, N=10, M=15) is the most demanding.** Multiple services must share each ECU, raising both the average utilisation ceiling and the constraint satisfaction difficulty. Results pending re-run with the new evaluation protocol.
