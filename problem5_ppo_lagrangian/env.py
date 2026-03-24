"""
P5 Environment — Lagrangian Constraint Relaxation.

Constraints are SOFT:
    - Violations are NOT blocking; episode always runs M steps.
    - Each violated step incurs per-step penalty: lambda_val * c_t.
    - lambda_val (Lagrangian multiplier) is updated externally by the training
        callback via dual ascent: λ ← clip(λ + lr*(avg_viol - target), 0, λ_max).

Violation at step t (c_t):
    c_t = 1.0  if capacity insufficient OR ECU already assigned, else 0.0

Reward per step:
    r_t = match_gain  -  (lambda_val + base_violation_penalty) * c_t  +  terminal_bonus
    where match_gain is the direct per-step placement quality (requirement/capacity)
    for feasible actions and 0.0 for violated actions.

Terminal bonus (terminal only, lightweight):
    +0.1 * final_ar always at episode end,
    and +0.2 * final_ar extra when the episode has zero violations.

Observation now includes the normalised λ value to reduce non-stationarity
from the policy's perspective.
remaining_vms can go negative (overloaded ECU).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import gymnasium as gym
import numpy as np
from problem2_ilp.objects import ECU, SVC


class LagrangeEnv(gym.Env):
    """
        Observation (shape: 4N+7+M):
            Standard PPO state from thinking.md plus one extra scalar:
            [0]                current service demand
            [1]                current cumulative AR
            [2]                sum of remaining usable ECU capacity
            [3]                sum of remaining service demand
            [4]                number of remaining usable ECUs
            [5]                number of remaining services
            [6:6+N]            initial capacity fraction per ECU
            [6+N:6+2N]         remaining capacity fraction per ECU
            [6+2N:6+3N]        ECU occupied flags
            [6+3N:6+4N]        valid-action flags for current service
            [6+4N:6+4N+M]      remaining service demands (sorted descending)
            [6+4N+M]           current λ value, normalised by λ_max

    Services are sorted by descending requirement at each episode reset, enabling
    the agent to use a first-fit-decreasing strategy and plan ahead.

    Reward per step:
        r_t = match_gain  -  (lambda_val + base_violation_penalty) * c_t  +  terminal_bonus
            c_t = 1.0 if capacity insufficient OR ECU already used, else 0.0
            match_gain = requirement/capacity for feasible actions, else 0.0
            terminal_bonus = +1.0 * final_ar (zero-violation episode)
                           or +0.1 * final_ar (episode with violations)

    Lagrangian multiplier λ is updated externally by the training callback via dual ascent:
    λ ← clip(λ + lr*(avg_viol - target), 0, λ_max)
    """

    metadata = {"render_modes": []}

    def __init__(self, ecus: list[ECU], services: list[SVC],
                 scenarios=None, lambda_init: float = 0.0,
                 lambda_max: float = 10.0):
        super().__init__()
        self._scenarios = scenarios
        self.ecus       = ecus
        self.services   = services
        self.N          = len(ecus)
        self.M          = len(services)
        self.lambda_val = float(lambda_init)
        self.lambda_max = max(float(lambda_max), 1e-8)

        self.action_space = gym.spaces.Discrete(self.N)
        # Remaining capacity can go negative in P5 due to soft constraints.
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=1.0, shape=(4 * self.N + 7 + self.M,), dtype=np.float32,
        )

        self.initial_vms = np.array([e.capacity for e in ecus], dtype=np.float32)
        self.remaining_vms:      np.ndarray
        self.ecu_assigned:       np.ndarray
        self.ar:                 float
        self._step:              int
        self.episode_violations: int
        self.reset()

    # ── λ setter — called by training callback after each dual-ascent step ────
    def set_lambda(self, val: float):
        self.lambda_val = float(val)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._scenarios is not None:
            caps, reqs = random.choice(self._scenarios)
            self.ecus     = [ECU(f"ECU{i}", cap) for i, cap in enumerate(caps)]
            self.services = [SVC(f"SVC{i}", req) for i, req in enumerate(reqs)]
            self.initial_vms = np.array([e.capacity for e in self.ecus], dtype=np.float32)
        # Sort services descending by requirement (FFD order improves greedy placement quality)
        self.services = sorted(self.services, key=lambda s: s.requirement, reverse=True)
        self.remaining_vms      = self.initial_vms.copy()
        self.ecu_assigned       = np.zeros(self.N, dtype=bool)
        self.ar                 = 0.0
        self._step              = 0
        self.episode_violations = 0
        return self._obs(), {}

    # ── observation ───────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        if self._step >= self.M:
            service_demand_norm = np.float32(0.0)
        else:
            svc = self.services[self._step]
            service_demand_norm = np.float32(
                svc.requirement / (np.max(self.initial_vms) + 1e-8)
            )
        remaining_pct = self.remaining_vms / (self.initial_vms + np.float32(1e-8))
        initial_cap_pct = self.initial_vms / (np.max(self.initial_vms) + np.float32(1e-8))
        assigned_flag = self.ecu_assigned.astype(np.float32)
        if self._step >= self.M:
            valid_flag = np.zeros(self.N, dtype=np.float32)
        else:
            valid_flag = ((~self.ecu_assigned) & (self.remaining_vms >= svc.requirement)).astype(np.float32)
        max_cap = float(np.max(self.initial_vms) + 1e-8)
        total_cap = float(np.sum(self.initial_vms) + 1e-8)
        usable_mask = ~self.ecu_assigned
        remaining_usable_capacity_sum = np.float32(
            np.sum(np.clip(self.remaining_vms[usable_mask], 0.0, None)) / total_cap
        )
        if self._step >= self.M:
            remaining_service_demand_sum = np.float32(0.0)
            remaining_services_count = np.float32(0.0)
        else:
            remaining_service_demand_sum = np.float32(
                sum(self.services[t].requirement for t in range(self._step, self.M)) / total_cap
            )
            remaining_services_count = np.float32((self.M - self._step) / max(self.M, 1))
        remaining_usable_ecu_count = np.float32(np.sum(usable_mask) / max(self.N, 1))
        remaining_svcs = np.array(
            [self.services[t].requirement / max_cap if t >= self._step else 0.0
             for t in range(self.M)],
            dtype=np.float32,
        )
        lambda_norm = np.float32(self.lambda_val / self.lambda_max)
        return np.concatenate([
            [service_demand_norm],
            np.array([self.ar], dtype=np.float32),
            np.array([remaining_usable_capacity_sum], dtype=np.float32),
            np.array([remaining_service_demand_sum], dtype=np.float32),
            np.array([remaining_usable_ecu_count], dtype=np.float32),
            np.array([remaining_services_count], dtype=np.float32),
            initial_cap_pct,
            remaining_pct,
            assigned_flag,
            valid_flag,
            remaining_svcs,
            np.array([lambda_norm], dtype=np.float32),
        ]).astype(np.float32)

    def _compute_feasible_total_util(self) -> float:
        loads = self.initial_vms - self.remaining_vms
        feasible_loads = np.minimum(loads, self.initial_vms)
        active_mask = self.ecu_assigned.astype(bool)
        if not np.any(active_mask):
            return 0.0
        return float(np.sum(feasible_loads[active_mask] / (self.initial_vms[active_mask] + 1e-8)))

    def _compute_ar(self) -> float:
        active_mask = self.ecu_assigned.astype(bool)
        active_ecus = int(np.sum(active_mask))
        if active_ecus == 0:
            return 0.0
        total_util = self._compute_feasible_total_util()
        return total_util / active_ecus

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        svc = self.services[self._step]
        prev_total_util = self._compute_feasible_total_util()

        # ── Violation check (soft — assignment always proceeds) ───────────────
        cap_violated = bool(self.remaining_vms[action] < svc.requirement)
        dup_violated = bool(self.ecu_assigned[action])
        violated     = cap_violated or dup_violated
        c_t          = 1.0 if violated else 0.0

        # ── Perform assignment regardless of violations ───────────────────────
        self.remaining_vms[action] -= svc.requirement   # may go negative
        self.ecu_assigned[action]   = True

        self.ar = self._compute_ar()
        self._step += 1
        if violated:
            self.episode_violations += 1

        done = self._step >= self.M

        match_gain = 0.0 if violated else float(svc.requirement / (self.initial_vms[action] + 1e-8))
        base_violation_penalty = 0.2
        terminal_bonus = 0.0
        if done:
            if self.episode_violations == 0:
                terminal_bonus = 1.0 * self.ar   # strong signal for clean episodes
            else:
                terminal_bonus = 0.1 * self.ar   # weak consolation for violated episodes
        reward = float(match_gain - (self.lambda_val + base_violation_penalty) * c_t + terminal_bonus)

        return self._obs(), reward, done, False, {
            "ar":              self.ar,
            "violated":        violated,
            "violations_ep":   self.episode_violations,
            "viol_rate_ep":    self.episode_violations / self._step,
            "services_placed": self._step,
            "lambda":          self.lambda_val,
        }

    # ── render ────────────────────────────────────────────────────────────────
    def render(self):
        if self._step < self.M:
            svc = self.services[self._step]
            print(f"  Step {self._step}/{self.M} | need {svc.requirement} VMs"
                  f" | AR={self.ar:.4f} | λ={self.lambda_val:.3f}"
                  f" | violations={self.episode_violations}")
        else:
            print(f"  Done | AR={self.ar:.4f}"
                  f" | violations={self.episode_violations}/{self.M}")


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test
