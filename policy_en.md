# Policy Overview for All Problems (Code-Aligned)

This document summarizes the strategy (policy design) used in each module based on the current code implementation.

### 1) Problem 1 (`problem1`): Baseline Gym Environment

- Core idea: build a simple MDP for service-to-ECU assignment.
- Action: choose one ECU for the current service.
- Observation: service demand + running AR + remaining ECU capacities.
- Constraint handling: hard fail on capacity violation (episode terminates).
- Reward policy:
    - valid step: directional AR feedback (`+1` if AR improves, `-1` if AR drops, `0` if unchanged)
    - invalid step: `-1`, terminate
- Role in project: educational baseline environment and transition prototype.

### 2) Problem 2 (`problem2_ilp`): ILP Global Optimum

- Core idea: solve the assignment as a binary integer linear program (PuLP/CBC).
- Action/policy form: not RL; direct optimization.
- Constraint handling: strict mathematical constraints (always feasible in final solution).
- Objective policy: maximize utilization objective globally.
- Role in project: upper-bound oracle and reference target for RL methods.

### 3) Problem 3 (`problem3_ppo`): PPO Without Constraint Enforcement

- Core idea: unconstrained PPO learns utilization behavior directly.
- Algorithm: Stable-Baselines3 PPO.
- Action: choose ECU from all `N` options.
- Observation (`N+2`): demand + AR + remaining capacity ratio.
- Constraint handling policy:
    - violations are recorded only (capacity overflow, duplicate ECU), not blocked.
    - episode still runs full `M` steps.
- Reward policy (current code): dense AR-direction signal
    - reward is AR-delta-shaped (`delta * 0.5 + sign term`), encouraging upward AR trend.
- Trade-off:
    - tends to explore high-AR behaviors quickly,
    - but can have high violation rates.

### 4) Problem 4 (`problem4_ppo_mask`): MaskablePPO With Hard Action Mask

- Core idea: enforce feasibility at action-sampling time via mask.
- Algorithm: `sb3_contrib` MaskablePPO.
- Action mask policy:
    - valid action iff ECU is unused AND has enough remaining capacity.
    - masked invalid actions are never sampled.
- Observation (current code, enhanced): `3N+2`
    - demand, AR, remaining ratio, occupied flags, valid-action flags.
- Reward policy (current code, enhanced):
    - base utilization gain `ru`
    - small capacity-waste penalty (encourages tighter packing)
    - demand-weighted early-stop penalty if infeasible future
    - completion bonus based on final AR
- Constraint behavior:
    - effectively zero direct violation actions due to mask,
    - may end early when no valid action remains for next service.

### 5) Problem 5 (`problem5_ppo_lagrangian`): PPO With Adaptive Lagrangian Penalty

- Core idea: soft-constraint PPO via dynamic dual variable `lambda`.
- Algorithm: PPO + callback dual ascent.
- Observation (`3N+3`):
    - demand, AR, normalized lambda, remaining ratio, occupied flags, valid-action flags.
- Constraint policy:
    - all actions remain selectable (no hard mask),
    - violations incur `lambda`-scaled penalty.
- Reward policy:
    - feasible utilization increment minus `lambda * violation_indicator`.
- Lambda update policy (current tuned code):
    - warmup episodes before dual updates,
    - window-averaged violation rate,
    - clipped `lambda` with reduced LR and smaller max to avoid over-penalization collapse.
- Trade-off:
    - smoother than pure hard-mask in some settings,
    - requires careful lambda dynamics tuning.

### 6) Problem 6 (`problem6_ppo_opt`): PPO + Best-Fit Patch Optimization

- Core idea: when selected ECU is already occupied, run a local patch/relocation heuristic.
- Algorithm: PPO (standard SB3).
- Observation (`2N+2`): demand, AR, remaining ratios, load ratios.
- Constraint/repair policy:
    - if target ECU occupied, try relocating one service using best-fit free ECU;
    - fallback enumerates alternatives to minimize violations;
    - extra safety branch handles no-free-ECU case robustly.
- Reward policy:
    - exact increase in total utilization after action.
- Trade-off:
    - keeps learning signal dense,
    - combines learned policy with structured combinational repair.

### 7) DQN (`dqn`): Off-Policy Value Learning With Termination Penalty

- Core idea: use DQN for discrete ECU selection with replay-based learning.
- Algorithm: SB3 DQN.
- Observation (`3N+2`): demand, AR, remaining ratio, occupied flags, valid-action flags.
- Constraint policy:
    - invalid action (overflow or duplicate) => immediate termination with negative penalty.
- Reward policy:
    - valid step reward `ru` (dense),
    - invalid action penalty proportional to unplaced services.
- Trade-off:
    - sample reuse via replay,
    - can be sensitive to exploration schedule and terminal-penalty scale.

---

## Cross-Problem Strategy Map

- `problem2_ilp`: exact optimization baseline (oracle).
- `problem3_ppo`: unconstrained RL baseline (high freedom, higher violation risk).
- `problem4_ppo_mask`: hard feasibility via action masking.
- `problem5_ppo_lagrangian`: soft feasibility via adaptive penalties.
- `problem6_ppo_opt`: RL + local repair heuristic hybrid.
- `dqn`: off-policy value-based alternative with terminal penalty constraints.

If your practical target is "high AR with near-zero violations":

1. Start with `problem4_ppo_mask` (most stable feasibility control).
2. Compare with `problem6_ppo_opt` for potential AR lift from patch optimization.
3. Use `problem2_ilp` as the gap reference.
